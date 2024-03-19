from typing import Dict, List, Any, Union, Tuple
from omegaconf import OmegaConf
from systemx.budget import BasicBudget

from systemx.report import Partition, Report
from systemx.events import Impression, Conversion
from systemx.budget_accountant import BudgetAccountant
from systemx.utils import (
    attribution_window_to_list,
    kInsufficientBudgetError,
    kOk,
    kNulledReport,
    IPA,
    USER_EPOCH_ARA,
    SYSTEMX,
    MONOEPOCH,
    MULTIEPOCH,
)


class User:
    # static filters shared across users
    global_filters_per_origin: Dict[str, BudgetAccountant] = {}  # For IPA
    logs: Dict[str, Dict[str, Any]] = {}

    def __init__(self, id: Any, config: OmegaConf) -> None:
        self.id = id
        self.config = config
        self.filters_per_origin: Dict[str, BudgetAccountant] = {}
        self.impressions: Dict[int, List[Impression]] = {}
        self.conversions: List[Conversion] = []

    def process_event(
        self, event: Union[Impression, Conversion]
    ) -> Union[Report, None]:
        if isinstance(event, Impression):
            if event.epoch not in self.impressions:
                self.impressions[event.epoch] = []
            self.impressions[event.epoch].append(event)

        elif isinstance(event, Conversion):
            self.conversions.append(event)
            report = self.create_report(event)
            return report

        else:
            raise ValueError(f"Unsupported event Type: {type(event)}")

    def create_report(self, conversion: Conversion) -> Union[Report, str]:
        """Searches for impressions to attribute within the attribution window that match
        the keys_to_match. Then creates a report using the attribution logic."""

        # Create partitioning
        partitions = self.create_partitions(conversion)

        # Get relevant impressions per epoch per partition
        for partition in partitions:
            (x, y) = partition.attribution_window
            for epoch in range(x, y + 1):
                if epoch in self.impressions:
                    # Linear search TODO: Optimize?
                    # Maybe sort impressions by key and do log search or sth?
                    for impression in self.impressions[epoch]:
                        if impression.matches(
                            conversion.destination, conversion.filter
                        ):
                            if epoch not in partition.impressions_per_epoch:
                                partition.impressions_per_epoch[epoch] = []
                            partition.impressions_per_epoch[epoch].append(impression)

        # Create a report per partition
        for partition in partitions:
            partition.create_report(conversion.key)

        # Compute global sensitivity
        match self.config.sensitivity_metric:
            case "L1":
                global_sensitivity = conversion.aggregatable_cap_value
            case _:
                raise ValueError(
                    f"Unsupported sensitivity metric: {self.config.sensitivity_metric}"
                )
        assert global_sensitivity is not None

        filters_per_origin = (
            User.global_filters_per_origin
            if self.config.baseline == IPA
            else self.filters_per_origin
        )

        if conversion.destination not in User.logs:
            User.logs[conversion.destination] = {
                "conversion_timestamp": conversion.timestamp,
                "total_budget_consumed": 0,
                "user_id": self.id,
                "attribution_window": conversion.attribution_window,
                "status": kOk,
            }
        destination_logs = User.logs[conversion.destination]

        # Budget accounting
        for partition in partitions:

            maybe_initialize_filters(
                filters_per_origin,
                conversion.destination,
                partition.attribution_window,
                self.config,
            )

            if self.config.baseline == IPA:
                # Central DP. Advertiser consumes worst-case budget from all the requested epochs in his global filter
                if not pay_all_or_nothing(
                    filters_per_origin,
                    partition.attribution_window,
                    conversion.destination,
                    conversion.epsilon,
                    destination_logs,
                ):
                    # Report is rejected at this point, returns error
                    destination_logs["status"] = kInsufficientBudgetError
                    return kInsufficientBudgetError

            elif self.config.baseline == USER_EPOCH_ARA:
                # Epochs in this partition pay worst case budget
                if not pay_all_or_nothing(
                    filters_per_origin,
                    partition.attribution_window,
                    conversion.destination,
                    conversion.epsilon,
                    destination_logs,
                ):
                    destination_logs["status"] = kNulledReport
                    partition.null_report()

            elif self.config.baseline == SYSTEMX:
                if partition.attribution_window_size() == 1:
                    # Partition covers only one epoch. The epoch in this partition pays budget based on its individual sensitivity
                    # Assuming Laplace
                    noise_scale = global_sensitivity / conversion.epsilon
                    p_individual_epsilon = (
                        partition.compute_sensitivity(self.config.sensitivity_metric)
                        / noise_scale
                    )

                    if not pay_all_or_nothing(
                        filters_per_origin,
                        partition.attribution_window,
                        conversion.destination,
                        p_individual_epsilon,
                        destination_logs,
                    ):
                        destination_logs["status"] = kNulledReport
                        partition.null_report()
                else:
                    # Partition is union of at least two epochs.
                    if self.config.optimization == MONOEPOCH:
                        # Optimization 1 is for partitions that cover one epoch only so it is ineffective here
                        if not pay_all_or_nothing(
                            filters_per_origin,
                            partition.attribution_window,
                            conversion.destination,
                            conversion.epsilon,
                            destination_logs,
                        ):
                            destination_logs["status"] = kNulledReport
                            partition.null_report()

                    elif self.config.optimization == MULTIEPOCH:
                        active_epochs = []
                        (x, y) = partition.attribution_window
                        for epoch in range(x, y + 1):
                            # Epochs empty of impressions are not paying any budget
                            if epoch in partition.impressions_per_epoch:
                                active_epochs.append(epoch)

                        if not pay_all_or_nothing(
                            filters_per_origin,
                            active_epochs,
                            conversion.destination,
                            conversion.epsilon,
                            destination_logs,
                        ):
                            destination_logs["status"] = kNulledReport
                            partition.null_report()
                    else:
                        raise ValueError(
                            f"Unsupported optimization: {self.config.optimization}"
                        )
            else:
                raise ValueError(f"Unsupported baseline: {self.config.baseline}")

        # Aggregate partition reports to create a final report
        final_report = Report()
        for partition in partitions:
            final_report += partition.report

        return final_report

    def create_partitions(self, conversion: Conversion) -> List[Partition]:
        match conversion.partitioning_logic:
            case "":
                # No partitioning - take union of all epochs
                return [
                    Partition(
                        conversion.attribution_window,
                        conversion.attribution_logic,
                        conversion.aggregatable_value,
                    )
                ]
            case "uniform":
                # One epoch per partition, value distributed uniformly
                (x, y) = conversion.attribution_window
                per_partition_value = float(conversion.aggregatable_value) / (y - x + 1)
                return [
                    Partition((i, i), conversion.attribution_logic, per_partition_value)
                    for i in range(x, y + 1)
                ]
            case _:
                raise ValueError(
                    f"Unsupported partitioning logic: {conversion.partitioning_logic}"
                )

def pay_all_or_nothing(
    filters_per_origin,
    attribution_epochs: Union[Tuple[int, int], List[int]],
    destination: str,
    epsilon: float,
    logs: Dict[str, Any],
) -> bool:
    if isinstance(attribution_epochs, tuple):
        attribution_epochs = attribution_window_to_list(attribution_epochs)

    destination_filter = filters_per_origin[destination]

    # Check if all epochs have enough remaining budget
    for epoch in attribution_epochs:
        # Check if epoch has enough budget
        if not destination_filter.can_run(epoch, BasicBudget(epsilon)):
            return False

    # Consume budget from all epochs
    for epoch in attribution_epochs:
        destination_filter.consume_block_budget(epoch, BasicBudget(epsilon))
        logs["total_budget_consumed"] += epsilon

    return True


def maybe_initialize_filters(
    filters_per_origin, destination: str, attribution_epochs: Tuple[int, int], config
):
    attribution_epochs = attribution_window_to_list(attribution_epochs)
    if destination not in filters_per_origin:
        filters_per_origin[destination] = BudgetAccountant()
    destination_filter = filters_per_origin[destination]

    for epoch in attribution_epochs:
        # Maybe initialize epoch
        if destination_filter.get_block_budget(epoch) is None:
            destination_filter.add_new_block_budget(epoch, float(config.initial_budget))

def get_logs_across_users() -> Dict[str, Dict[str, Any]]:
    return User.logs