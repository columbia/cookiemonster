from typing import Any, Dict, Optional, Tuple


class Event:
    def __init__(self):
        pass


class Impression(Event):
    def __init__(
        self,
        timestamp: int,
        epoch: int,
        destination: str,
        filter: str,
        key: str,
        user_id: Any,
    ):
        self.timestamp = timestamp
        self.epoch = epoch
        self.destination = destination
        self.filter = filter
        self.key = key
        self.user_id = user_id

    def matches(self, destination: str, filter: str):
        # Condition: destinations and keys must match
        if self.destination == destination and self.filter == filter:
            return True
        return False

    def belongs_in_attribution_window(self, attribution_window):
        return (
            self.timestamp >= attribution_window[0]
            and self.timestamp <= attribution_window[1]
        )

    def __str__(self):
        return f"|Impression| Epoch: {self.epoch}, Timestamp: {self.timestamp}, User: {self.user_id}, Filter: {self.filter}"


class Conversion(Event):
    def __init__(
        self,
        timestamp: int,
        id: int,
        epoch: int,
        destination: str,
        attribution_window: Tuple[int, int],
        epochs_window: Tuple[int, int],
        attribution_logic: str,
        partitioning_logic: str,
        aggregatable_value: float,
        aggregatable_cap_value: float,
        filter: str,
        key: str,
        epsilon: float,
        noise_scale: float,
        user_id: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.timestamp = timestamp
        self.id = id
        self.epoch = epoch
        self.destination = destination
        self.attribution_window = attribution_window
        self.epochs_window = epochs_window
        self.attribution_logic = attribution_logic
        self.partitioning_logic = partitioning_logic
        self.aggregatable_value = aggregatable_value
        self.aggregatable_cap_value = aggregatable_cap_value
        self.filter = filter
        self.key = key
        self.user_id = user_id
        self.metadata = metadata
        self.epsilon = epsilon
        self.noise_scale = noise_scale

    def __str__(self):
        return f"|Conversion| {self.id}, Attribution-Window in Timestamps: {self.attribution_window}, Epochs window: {self.epochs_window}, User: {self.user_id}, Filter: {self.filter}, Value: {self.aggregatable_value}"
