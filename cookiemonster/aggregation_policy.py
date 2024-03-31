from abc import ABC, abstractmethod
from enum import Enum
from omegaconf import DictConfig

from cookiemonster.query_batch import QueryBatch


class AggregationPolicyType(str, Enum):
    COUNT_POLICY = "count_conversion_policy"
    EPOCH_POLICY = "epoch_policy"


class AggregationPolicy(ABC):
    def __init__(self, config: DictConfig) -> None:
        self.min_interval = config.get("min_interval")
        if self.min_interval:
            self.max_interval = config.max_interval
        else:
            self.max_interval = config.interval

    @classmethod
    def create(cls, config: DictConfig) -> "AggregationPolicy":
        if config.type == AggregationPolicyType.COUNT_POLICY:
            return CountConversionPolicy(config)
        elif config.type == AggregationPolicyType.EPOCH_POLICY:
            return EpochPolicy(config)
        else:
            raise NotImplementedError(
                "The requested batch policy type has not been implemented"
            )

    @abstractmethod
    def should_calculate_summary_reports(
        self, query_batch: QueryBatch, *, tail: bool = False
    ) -> bool:
        pass


class CountConversionPolicy(AggregationPolicy):
    """
    CountConversionPolicy - an aggregation policy that will calculate the summary reports after the specified
    number of iterations has elapsed.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)

    def should_calculate_summary_reports(
        self, query_batch: QueryBatch, *, tail: bool = False
    ) -> bool:
        if query_batch.global_epsilon == -1:
            return False
        
        if tail and self.min_interval:
            return query_batch.size() >= self.min_interval
        else:
            return query_batch.size() == self.max_interval


class EpochPolicy(AggregationPolicy):
    """
    EpochPolicy - an aggregation policy that will calculate summary reports after the specified
    number of epochs has elapsed. Assumed that events will come in ascending order of epochs.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)

    def should_calculate_summary_reports(
        self, query_batch: QueryBatch, *, tail: bool = False
    ) -> bool:
        if query_batch.global_epsilon == -1:
            return False
        
        epoch_count = query_batch.epochs_window[1] - query_batch.epochs_window[0]
        if tail and self.min_interval:
            return epoch_count >= self.min_interval
        else:
            return epoch_count == self.max_interval
