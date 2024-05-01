import random
from typing import Any, Dict, List, Union

from omegaconf import OmegaConf

from cookiemonster.attribution import LastTouch, LastTouchWithCount
from cookiemonster.budget_accountant import BudgetAccountant
from cookiemonster.events import Conversion, Impression
from cookiemonster.report import HistogramReport, Partition, Report, ScalarReport
from cookiemonster.utils import (
    COOKIEMONSTER,
    COOKIEMONSTER_BASE,
    IPA,
    compute_global_sensitivity,
    maybe_initialize_filters,
)


class ConversionResult:
    def __init__(self, unbiased_final_report: Report, final_report: Report) -> None:
        self.unbiased_final_report = unbiased_final_report
        self.final_report = final_report


class User:

    def __init__(self, id: Any, config: OmegaConf) -> None:
        self.id = id
        self.config = config.user
        self.logging_keys = config.logs.logging_keys
        self.filters_per_origin: Dict[str, BudgetAccountant] = {}
        self.impressions: Dict[int, List[Impression]] = {}
        self.initial_budget = float(self.config.initial_budget)

    def process_event(
        self, event: Union[Impression, Conversion]
    ) -> Union[None, ConversionResult]:
        if isinstance(event, Impression):
            if event.epoch not in self.impressions:
                self.impressions[event.epoch] = []
            self.impressions[event.epoch].append(event)

        elif isinstance(event, Conversion):
            return self.create_report(event)

        else:
            raise ValueError(f"Unsupported event Type: {type(event)}")

    def create_report(self, conversion: Conversion) -> ConversionResult:
        """Searches for impressions to attribute within the attribution window that match
        the keys_to_match. Then creates a report using the attribution logic."""

        # Create partitioning
        partitions = self.create_partitions(conversion)

        # Get relevant impressions per epoch per partition
        for partition in partitions:
            (x, y) = partition.epochs_window
            for epoch in range(x, y + 1):
                if epoch in self.impressions:
                    # Linear search TODO: Optimize?
                    # Maybe sort impressions by key and do log search or sth?
                    for impression in self.impressions[epoch]:

                        # Make sure impression is within the attribution_window and matches the conversion
                        if impression.belongs_in_attribution_window(
                            conversion.attribution_window
                        ) and impression.matches(
                            conversion.destination, conversion.filter
                        ):
                            if epoch not in partition.impressions_per_epoch:
                                partition.impressions_per_epoch[epoch] = []
                            partition.impressions_per_epoch[epoch].append(impression)

        # Initialize attribution logic
        if self.config.bias_detection_knob:
            attribution_function = LastTouchWithCount(
                sensitivity_metric=self.config.sensitivity_metric,
                attribution_cap=conversion.aggregatable_cap_value,
                kappa=self.config.bias_detection_knob,
            )

        else:
            attribution_function = LastTouchWithCount(
                sensitivity_metric=self.config.sensitivity_metric,
                attribution_cap=conversion.aggregatable_cap_value,
            )

        # Create the unbiased report per partition
        for partition in partitions:
            # partition.unbiased_report = partition.create_report(
            #     conversion.filter,
            #     conversion.key,
            #     bias_counting_strategy=self.config.bias_detection_knob,
            # )

            partition.unbiased_report = attribution_function.create_report(
                partition=partition,
                filter=conversion.filter,
                key_piece=conversion.key,
            )

        # IPA doesn't do on-device budget accounting
        if self.config.baseline != IPA:

            # Compute global sensitivity
            global_sensitivity = attribution_function.compute_global_sensitivity()

            # Budget accounting
            for partition in partitions:

                # Initialize filters for the origin
                origin_filters = maybe_initialize_filters(
                    self.filters_per_origin,
                    conversion.destination,
                    partition.epochs_window,
                    self.initial_budget,
                )

                # Compute the required budget and the epochs to pay depending on the baseline
                if self.config.baseline == COOKIEMONSTER_BASE:
                    # Epochs in this partition pay worst case budget
                    epochs_to_pay = partition.epochs_window
                    budget_required = conversion.epsilon

                elif self.config.baseline == COOKIEMONSTER:
                    # TODO: use attribution_function methods here.
                    if partition.epochs_window_size() == 1:
                        # Partition covers only one epoch. The epoch in this partition pays budget based on its individual sensitivity
                        # Assuming Laplace here
                        epochs_to_pay = partition.epochs_window
                        noise_scale = global_sensitivity / conversion.epsilon
                        budget_required = (
                            partition.compute_sensitivity(
                                self.config.sensitivity_metric
                            )
                            / noise_scale
                        )
                    else:
                        # Partition is union of at least two epochs.
                        budget_required = conversion.epsilon

                        epochs_to_pay = []
                        (x, y) = partition.epochs_window
                        for epoch in range(x, y + 1):
                            if not origin_filters.can_run(epoch, budget_required):
                                # Delete epoch from partition so that it will be ignored upon report creation, won't be added to epochs_to_pay either
                                if epoch in partition.impressions_per_epoch:
                                    del partition.impressions_per_epoch[epoch]

                            # Epochs empty of impressions are not paying any budget
                            if epoch in partition.impressions_per_epoch:
                                epochs_to_pay.append(epoch)

                            # epochs_to_pay are epochs that contain relevant impressions in partition.impressions_per_epoch AND can pay the required budget
                            # Report will be created based on the remaining impressions_per_epoch of the partition
                            # If no epoch could pay or no epoch contained relevant impressions, the report will be empty

                else:
                    raise ValueError(f"Unsupported baseline: {self.config.baseline}")

                filter_result = origin_filters.pay_all_or_nothing(
                    epochs_to_pay, budget_required
                )
                if not filter_result.succeeded():
                    # If epochs couldn't pay the required budget, erase the partition's impressions_per_epoch so that an empty report will be created
                    partition.impressions_per_epoch.clear()

        # Create the possibly biased report per partition
        for partition in partitions:
            partition.report = attribution_function.create_report(
                partition=partition,
                filter=conversion.filter,
                key_piece=conversion.key,
            )

        # TODO(Pierre): count the number of reports with cleared impressions, use that as upper bound
        # Does this really augment the individual sensitivity actually? Only for epochs with impressions. Or with relevant impressions?
        # TODO: seems that we already have many legit empty epochs in the synthetic dataset.

        # Other ideas:
        # Or cleared impressions that would have been relevant otherwise? No way to tell... Hence the prior.
        # If we use budgets, then we can combine on attribution value and reduce the individual sensitivity?

        # Aggregate partition reports to create a final report
        # final_report = Report()
        # for partition in partitions:
        #     final_report += partition.report

        # TODO: maybe neater to have attribution_function.sum_reports(partitions)
        final_report = sum(
            [partition.report for partition in partitions],
            start=attribution_function.create_empty_report(),
        )

        # Keep an unbiased version of the final report for experiments
        # unbiased_final_report = Report()
        # for partition in partitions:
        #     unbiased_final_report += partition.unbiased_report
        unbiased_final_report = sum(
            [partition.unbiased_report for partition in partitions],
            start=attribution_function.create_empty_report(),
        )

        conversion_result = ConversionResult(unbiased_final_report, final_report)
        return conversion_result

    def create_partitions(self, conversion: Conversion) -> List[Partition]:
        match conversion.partitioning_logic:
            case "":
                # No partitioning - take union of all epochs
                return [
                    Partition(
                        conversion.epochs_window,
                        conversion.attribution_logic,
                        conversion.aggregatable_value,
                    )
                ]
            case "uniform":
                # One epoch per partition, value distributed uniformly
                (x, y) = conversion.epochs_window
                per_partition_value = float(conversion.aggregatable_value) / (y - x + 1)
                return [
                    Partition((i, i), conversion.attribution_logic, per_partition_value)
                    for i in range(x, y + 1)
                ]
            case _:
                raise ValueError(
                    f"Unsupported partitioning logic: {conversion.partitioning_logic}"
                )
