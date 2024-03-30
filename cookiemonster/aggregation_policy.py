from abc import ABC, abstractmethod
from enum import Enum
from omegaconf import DictConfig

from cookiemonster.events import Impression, Conversion

class AggregationPolicyType(str, Enum):
    COUNT_POLICY = "count_conversion_policy"
    EPOCH_POLICY = "epoch_policy"

class AggregationPolicy(ABC):
    @classmethod
    def create(cls, config: DictConfig) -> "AggregationPolicy":
        if config.type == AggregationPolicyType.COUNT_POLICY:
            return CountConversionPolicy(config)
        elif config.type == AggregationPolicyType.EPOCH_POLICY:
            return EpochPolicy(config)
        else:
            raise NotImplementedError("The requested batch policy type has not been implemented")
        
    @abstractmethod
    def should_calculate_summary_reports(self, event: Impression | Conversion) -> bool:
        pass

class CountConversionPolicy(AggregationPolicy):
    """
    CountConversionPolicy - an aggregation policy that will calculate the summary reports after the specified
    number of iterations has elapsed.
    """
    def __init__(self, config: DictConfig) -> None:
        self.counts = config.interval
        self.current = 0
    
    def should_calculate_summary_reports(self, event: Impression | Conversion) -> bool:
        if isinstance(event, Conversion):
            self.current += 1
            return self.current % self.counts == 0
        else:
            return False
    
class EpochPolicy(AggregationPolicy):
    """
    EpochPolicy - an aggregation policy that will calculate summary reports after the specified
    number of epochs has elapsed. Assumed that events will come in ascending order of epochs.
    """
    def __init__(self, config: DictConfig) -> None:
        self.epochs = config.interval
        self.current = 0

    def should_calculate_summary_reports(self, event: Impression | Conversion) -> bool:
        if isinstance(event, Impression):
            old = self.current
            self.current = event.epoch
            return self.current - old >= self.epochs
        else:
            return False
