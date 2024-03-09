from typing import Dict, List, Any, Union, Tuple
from omegaconf import OmegaConf
from events import Impression, Conversion
from report import Partition, Report
from utils import attribution_window_to_list
from budgets import BudgetAccountant


class User:
    def __init__(self, id: Any, config: OmegaConf) -> None:
        self.id = id
        self.config = config

        self.filters_per_origin: Dict[str, BudgetAccountant] = {}
        self.impressions: Dict[int, List[Impression]] = {}
        self.conversions: List[Conversion] = {}

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

    def create_report(self, conversion: Conversion) -> Report:
        """Searches for impressions to attribute within the attribution window that match
        the keys_to_match. Then creates a report using the attribution logic."""

        # Create partitioning
        partitions = create_partitions(conversion)

        # Get relevant impressions per epoch per partition
        for partition in partitions:
            (x, y) = partition.attribution_window
            for epoch in range(x, y + 1):
                if epoch in self.impressions:
                    # Linear search TODO: Optimize?
                    # Maybe sort impressions by key and do log search or sth?
                    for impression in self.impressions[epoch]:
                        if impression.matches(
                            conversion.destination, conversion.keys_to_match
                        ):
                            if epoch not in partition.impressions_per_epoch:
                                partition.impressions_per_epoch[epoch] = []
                            partition.impressions_per_epoch[epoch].append(impression)

        # Create a report per partition
        for partition in partitions:
            partition.create_report()

        # Compute global sensitivity
        match self.config.sensitivity_metric:
            case "L1":
                global_sensitivity = conversion.aggregatable_cap_value
            case _:
                raise ValueError(
                    f"Unsupported sensitivity metric: {self.config.sensitivity_metric}"
                )
        assert global_sensitivity is not None

        # Budget accounting
        for partition in partitions:
            if self.config.optimization == "0":
                # No optimizations. Epochs in this partition pay worst case budget
                if not self.pay_all_or_nothing(
                    partition.attribution_window,
                    conversion,
                    destination,
                    conversion.epsilon,
                ):
                    partition.null_report()
                continue

            if partition.attribution_window_size() == 1:
                # Partition covers only one epoch. The epoch in this partition pays budget based on its individual sensitivity
                # Assuming Laplace
                noise_scale = global_sensitivity / conversion.epsilon
                p_individual_epsilon = (
                    partition.compute_sensitivity(self.config.sensitivity_metric)
                    / noise_scale
                )

                if not self.pay_all_or_nothing(
                    partition.attribution_window,
                    conversion.destination,
                    p_individual_epsilon,
                ):
                    partition.null_report()
            else:
                # Partition is union of at least two epochs.
                if self.config.optimization == "1":
                    # Optimization 1 is for partitions that cover one epoch only so it is ineffective here
                    if not self.pay_all_or_nothing(
                        partition.attribution_window,
                        conversion.destination,
                        conversion.epsilon,
                    ):
                        partition.null_report()

                elif self.config.optimization == "2":
                    active_epochs = []
                    (x, y) = partition.attribution_window
                    for epoch in range(x, y + 1):
                        # Epochs empty of impressions are not paying any budget
                        if epoch in partition.impressions_per_epoch:
                            active_epochs.append(epoch)

                    if not self.pay_all_or_nothing(
                        active_epochs, conversion.destination, conversion.epsilon
                    ):
                        partition.null_report()
                else:
                    raise ValueError(
                        f"Unsupported optimization: {self.config.optimization}"
                    )

        # Aggregate partition reports to create a final report
        final_report = Report()
        for partition in partitions:
            final_report += partition.report

        return final_report

    def get_partitions(self, conversion: Conversion) -> List[Partition]:
        match conversion.partitioning_logic:
            case "":
                # No partitioning - take union of all epochs
                return [
                    Partition(
                        conversion.attribution_window,
                        conversion.attribution_logic,
                        conversion.value,
                    )
                ]
            case "uniform":
                # One epoch per partition, value distributed uniformly
                per_partition_value = float(conversion.value) / len(partitions)
                (x, y) = attribution_window
                return [
                    Partition((i, i), conversion.attribution_logic, per_partition_value)
                    for i in range(x, y + 1)
                ]
            case _:
                raise ValueError(
                    f"Unsupported partitioning logic: {conversion.partitioning_logic}"
                )

    def pay_all_or_nothing(
        self,
        attribution_epochs: Union[Tuple[int, int], List[int]],
        destination: str,
        epsilon: float,
    ) -> bool:
        if isinstance(attribution_epochs, tuple):
            attribution_epochs = attribution_window_to_list(attribution_epochs)

        if destination not in self.filters_per_origin:
            self.filters_per_origin[destination] = BudgetAccountant(self.config)

        destination_filter = self.filters_per_origin[destination]

        # Check if all epochs have enough remaining budget
        for epoch in attribution_epochs:
            # Maybe initialize epoch
            if destination_filter.get_block_budget(epoch) is None:
                destination_filter.add_new_block_budget(epoch)

            # Check if epoch has enough budget
            if not destination_filter.can_run(epoch):
                return False

        # Consume budget from all epochs
        for epoch in attribution_epochs:
            destination_filter.consume_block_budget(epoch, epsilon)

        return True
