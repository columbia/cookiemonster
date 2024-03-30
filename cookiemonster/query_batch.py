import math


class QueryBatch:
    def __init__(self, query_id, epsilon, sensitivity) -> None:
        self.values = []
        self.unbiased_values = []
        self.epochs_window = (math.inf, 0)
        self.query_id = query_id
        self.global_epsilon = epsilon
        self.global_sensitivity = sensitivity

    def size(self):
        return len(self.values)

    def update(self, value, unbiazed_value, epochs_window):
        self.values.append(value)
        self.unbiased_values.append(unbiazed_value)
        self.upate_epochs_window(epochs_window)

    def upate_epochs_window(self, epochs_window):
        (a, b) = epochs_window
        self.epochs_window = (
            min(a, self.epochs_window[0]),
            max(b, self.epochs_window[1]),
        )
