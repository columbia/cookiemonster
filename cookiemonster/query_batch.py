import math


class QueryBatch:
    def __init__(
        self, query_id: str, epsilon: float, sensitivity: int, biggest_id: int
    ) -> None:
        self.values = []
        self.unbiased_values = []
        self.epochs_window = (math.inf, 0)
        self.query_id = query_id
        self.global_epsilon = epsilon
        self.global_sensitivity = sensitivity
        self.biggest_id = biggest_id

    def size(self):
        return len(self.values)

    def update(self, value, unbiazed_value, epochs_window, biggest_id):
        self.values.append(value)
        self.unbiased_values.append(unbiazed_value)
        self.upate_epochs_window(epochs_window)
        self.biggest_id = max(biggest_id, self.biggest_id)

    def upate_epochs_window(self, epochs_window):
        (a, b) = epochs_window
        self.epochs_window = (
            min(a, self.epochs_window[0]),
            max(b, self.epochs_window[1]),
        )
