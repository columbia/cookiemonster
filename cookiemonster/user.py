from omegaconf import OmegaConf
from typing import Dict, List, Any, Union

from cookiemonster.event_logger import EventLogger
from cookiemonster.report import Partition, Report
from cookiemonster.events import Impression, Conversion
from cookiemonster.budget_accountant import BudgetAccountant

from cookiemonster.utils import maybe_initialize_filters, compute_global_sensitivity

from cookiemonster.utils import (
    IPA,
    USER_EPOCH_ARA,
    COOKIEMONSTER,
    MONOEPOCH,
    MULTIEPOCH,
    BUDGET,
)


class ConversionResult:
    def __init__(self, unbiased_final_report: Report, final_report: Report) -> None:
        self.unbiased_final_report = unbiased_final_report
        self.final_report = final_report


class User:
    logger = EventLogger()

    def __init__(self, id: Any, config: OmegaConf) -> None:
        self.id = id
        self.config = config.user
        self.logging_keys = config.logs.logging_keys
        self.filters_per_origin: Dict[str, BudgetAccountant] = {}
        self.impressions: Dict[int, List[Impression]] = {}
        self.conversions: List[Conversion] = []

    def process_event(
        self, event: Union[Impression, Conversion]
    ) -> Union[None, ConversionResult]:
        if isinstance(event, Impression):
            if event.epoch not in self.impressions:
                self.impressions[event.epoch] = []
            self.impressions[event.epoch].append(event)

        elif isinstance(event, Conversion):
            self.conversions.append(event)
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

        # Create a report per partition
        for partition in partitions:
            partition.create_report(conversion.filter, conversion.key)

        # IPA doesn't do on-device budget accounting
        if self.config.baseline != IPA:

            # Compute global sensitivity
            global_sensitivity = compute_global_sensitivity(
                self.config.sensitivity_metric, conversion.aggregatable_cap_value
            )

            # Budget accounting
            for partition in partitions:

                origin_filters = maybe_initialize_filters(
                    self.filters_per_origin,
                    conversion.destination,
                    partition.epochs_window,
                    float(self.config.initial_budget),
                )

                if self.config.baseline == USER_EPOCH_ARA:
                    # Epochs in this partition pay worst case budget
                    filter_result = origin_filters.pay_all_or_nothing(
                        partition.epochs_window, conversion.epsilon
                    )

                elif self.config.baseline == COOKIEMONSTER:
                    if partition.epochs_window_size() == 1:
                        # Partition covers only one epoch. The epoch in this partition pays budget based on its individual sensitivity (Assuming Laplace)
                        noise_scale = global_sensitivity / conversion.epsilon
                        p_individual_epsilon = (
                            partition.compute_sensitivity(
                                self.config.sensitivity_metric
                            )
                            / noise_scale
                        )
                        filter_result = origin_filters.pay_all_or_nothing(
                            partition.epochs_window, p_individual_epsilon
                        )

                    else:
                        # Partition is union of at least two epochs.
                        if self.config.optimization == MONOEPOCH:
                            # Optimization 1 is for partitions that cover one epoch only so it is ineffective here
                            filter_result = origin_filters.pay_all_or_nothing(
                                partition.epochs_window, conversion.epsilon
                            )

                        elif self.config.optimization == MULTIEPOCH:
                            active_epochs = []
                            (x, y) = partition.epochs_window
                            for epoch in range(x, y + 1):
                                # Epochs empty of impressions are not paying any budget
                                if epoch in partition.impressions_per_epoch:
                                    active_epochs.append(epoch)

                            filter_result = origin_filters.pay_all_or_nothing(
                                active_epochs, conversion.epsilon
                            )

                        else:
                            raise ValueError(
                                f"Unsupported optimization: {self.config.optimization}"
                            )

                else:
                    raise ValueError(f"Unsupported baseline: {self.config.baseline}")

                if not filter_result.succeeded():
                    partition.null_report()

                if BUDGET in self.logging_keys:
                    User.logger.log(
                        BUDGET,
                        conversion.id,
                        conversion.destination,
                        self.id,
                        conversion.epochs_window,
                        filter_result.budget_consumed,
                        filter_result.status,
                    )

        # Aggregate partition reports to create a final report
        final_report = Report()
        for partition in partitions:
            final_report += partition.report

        # Keep an unbiased version of the final report for experiments
        unbiased_final_report = Report()
        for partition in partitions:
            unbiased_final_report += partition.unbiased_report

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


def get_log_events_across_users() -> Dict[str, Dict[str, Any]]:
    return User.logger
