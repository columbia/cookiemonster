from typing import Dict, Any, Tuple, Optional


class Event:
    def __init__(self):
        pass


class Impression(Event):
    def __init__(self, epoch: int, destination: str, filter: str, key: str):
        self.epoch = epoch
        self.destination = destination
        self.filter = filter
        self.key = key

    def matches(self, destination: str, filter: str):
        # Condition: destinations and keys must match
        if self.destination == destination and self.filter == filter:
            return True
        return False

    def __str__(self):
        return f"|Impression| Epoch: {self.epoch}, Destination: {self.destination}"


class Conversion(Event):
    def __init__(
        self,
        destination: str,
        attribution_window: Tuple[int, int],
        attribution_logic: str,
        partitioning_logic: str,
        aggregatable_value: float,
        aggregatable_cap_value: float,
        filter: str,
        key: str,
        epsilon: float,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.destination = destination
        self.attribution_window = attribution_window
        self.attribution_logic = attribution_logic
        self.partitioning_logic = partitioning_logic
        self.aggregatable_value = aggregatable_value
        self.aggregatable_cap_value = aggregatable_cap_value
        self.filter = filter
        self.key = key
        self.metadata = metadata
        self.epsilon = epsilon

    def __str__(self):
        return f"|Conversion| Attribution-Window: {self.attribution_window}, Destination: {self.destination}, Value: {self.aggregatable_value}"
