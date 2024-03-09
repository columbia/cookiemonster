from typing import List, Tuple


class Event:
    def __init__(self):
        pass


class Impression(Event):
    def __init__(self, epoch: int, destination: str, keys: Dict[str, Any]):
        self.epoch = epoch
        self.destination = destination
        self.keys = keys

    def matches(self, destination: str, keys_to_match: Dict[str, Any]):
        # Condition: destinations and keys must match
        if self.destination == destination and self.keys == keys_to_match:
            return True
        return False


class Conversion(Event):
    def __init__(
        self,
        destination: str,
        attribution_window: Tuple[int, int],
        attribution_logic: str,
        partitioning_logic: str,
        aggregatable_value: float,
        aggregatable_cap_value: float,
        keys_to_match: Dict[str, Any],
        metadata: Dict[str, Any],
        epsilon: float,
    ):
        self.destination = destination
        self.attribution_window = attribution_window
        self.attribution_logic = attribution_logic
        self.partitioning_logic = partitioning_logic
        self.aggregatable_value = aggregatable_value
        self.aggregatable_cap_value = aggregatable_cap_value
        self.keys_to_match = keys_to_match
        self.metadata = metadata
        self.epsilon = epsilon
