import random
from typing import Any, Dict, List, Optional, Tuple, Union

from omegaconf import OmegaConf

from cookiemonster.attribution import (LastTouch,
                                       LastTouchWithAlteredReportCount,
                                       LastTouchWithEmptyEpochCount)
from cookiemonster.budget_accountant import BudgetAccountant
from cookiemonster.events import Conversion, Impression
from cookiemonster.report import HistogramReport, Report, ScalarReport
from cookiemonster.utils import (COOKIEMONSTER, COOKIEMONSTER_BASE, IPA,
                                 compute_global_sensitivity,
                                 maybe_initialize_filters)


class ConversionResult:
    def __init__(self, unbiased_final_report: Report, final_report: Report) -> None:
        self.unbiased_final_report = unbiased_final_report
        self.final_report = final_report


class Partition:
    """
    A simple container for impressions in a set of epochs on a single device.
    Also stores partially computed reports.
    Impressions might be modified in-place depending on budget consumption,
    this does not affect the true on-device impression.
    """

    def __init__(
        self,
        epochs_window: Tuple[int, int],
        attribution_logic: str,
        value: Optional[float],
    ) -> None:
        self.epochs_window = epochs_window
        self.attribution_logic = attribution_logic
        self.impressions_per_epoch: Dict[int, List[Impression]] = {}
        self.value = value
        self.report = None
        self.unbiased_report = None

    def epochs_window_size(self) -> int:
        return self.epochs_window[1] - self.epochs_window[0] + 1

    def get_epochs(self, reverse=False) -> List[int]:
        """Returns a list of epochs in the window. Not just epochs with events!"""
        if reverse:
            return list(range(self.epochs_window[1], self.epochs_window[0] - 1, -1))

        return list(range(self.epochs_window[0], self.epochs_window[1] + 1))

    def __str__(self):
        return str(self.__dict__)


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
            return self.process_conversion_event(event)

        else:
            raise ValueError(f"Unsupported event Type: {type(event)}")

    def process_conversion_event(self, conversion: Conversion) -> ConversionResult:
        """Searches for impressions to attribute within the attribution window that match
        the keys_to_match. Then creates a report using the attribution logic.
        Also stores an unbiased version of the report, to compute metrics.
        """

        # Initialize attribution logic. 
        # If conversion value is public, we can use it as attribution_cap,
        # which will be used for the global sensitivity.
        # Otherwise you would need to use attribution_cap_value
        if self.config.bias_detection_knob:
            
            # attribution_function = LastTouchWithEmptyEpochCount(
            #     sensitivity_metric=self.config.sensitivity_metric,
            #     attribution_cap=conversion.aggregatable_value,
            #     kappa=self.config.bias_detection_knob,
            # )
            
            attribution_function = LastTouchWithAlteredReportCount(
                sensitivity_metric=self.config.sensitivity_metric,
                attribution_cap=conversion.aggregatable_value,
                kappa=self.config.bias_detection_knob,
            )

        else:
            attribution_function = LastTouch(
                sensitivity_metric=self.config.sensitivity_metric,
                attribution_cap=conversion.aggregatable_value,
            )

        # Create partitioning
        partitions = self.create_partitions(conversion)

        # Treat different partitions as if they were different queries
        for partition in partitions:

            # Get relevant impressions per epoch
            for epoch in partition.get_epochs():
                # Create an empty impression set for all the epochs on the device
                # Useful to tell that they have budget (hearbeat)
                partition.impressions_per_epoch[epoch] = []

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
                            partition.impressions_per_epoch[epoch].append(impression)

            # Create unbiased report
            partition.unbiased_report = attribution_function.create_report(
                partition=partition,
                filter=conversion.filter,
                key_piece=conversion.key,
            )

            # Modify partition in place and pay on-device budget
            if self.config.baseline == COOKIEMONSTER_BASE:
                # Use global sensitivity for all epochs in partition
                # TODO: actually we could drop OOB epochs here too instead of aborting?
                # Initialize filters for the origin
                # TODO: the state of this function is odd. Maybe initilize when we try to pay?
                origin_filters = maybe_initialize_filters(
                    self.filters_per_origin,
                    conversion.destination,
                    partition.epochs_window,
                    self.initial_budget,
                )

                epochs_to_pay = partition.epochs_window

                # TODO: double check that we can use global sensitivity with the actual conversion value here.
                # - Unlike IPA, we can't wait for the whole batch to take the max
                # - But still ok with parallel composition
                budget_required = attribution_function.compute_global_sensitivity() / conversion.noise_scale
                # budget_required = conversion.epsilon

                filter_result = origin_filters.pay_all_or_nothing(
                    epochs_to_pay, budget_required
                )
                if not filter_result.succeeded():
                    # If any epoch in the partition couldn't pay the required budget,
                    # erase the partition's impressions_per_epoch so that an empty report will be created
                    partition.impressions_per_epoch.clear()

            elif self.config.baseline == COOKIEMONSTER:
                # The noise scale is fixed for the whole query
                noise_scale = conversion.noise_scale

                origin_filters = maybe_initialize_filters(
                    self.filters_per_origin,
                    conversion.destination,
                    partition.epochs_window,
                    self.initial_budget,
                )

                for epoch in partition.get_epochs():
                    individual_sensitivity = (
                        attribution_function.compute_individual_sensitivity(
                            partition, epoch
                        )
                    )

                    budget_required = individual_sensitivity / noise_scale
                    filter_result = origin_filters.pay_all_or_nothing(
                        [epoch], budget_required
                    )
                    if not filter_result.succeeded():
                        # Empty the impressions from `epoch`
                        del partition.impressions_per_epoch[epoch]

            else:
                # IPA doesn't do on-device budget accounting
                assert (
                    self.config.baseline == IPA
                ), f"Unsupported baseline: {self.config.baseline}"

            # Create the possibly biased report
            partition.report = attribution_function.create_report(
                partition=partition,
                filter=conversion.filter,
                key_piece=conversion.key,
            )

        # Aggregate partition reports to create a final report
        final_report = attribution_function.sum_reports(
            partition.report for partition in partitions
        )

        # Keep an unbiased version of the final report for experiments
        unbiased_final_report = attribution_function.sum_reports(
            partition.unbiased_report for partition in partitions
        )

        return ConversionResult(unbiased_final_report, final_report)

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
