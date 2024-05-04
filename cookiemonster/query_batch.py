from cookiemonster.utils import EpochsWindow


class QueryBatch:
    def __init__(
        self, query_id: str, epsilon: float, sensitivity: int, biggest_id: int
    ) -> None:
        self.values = []
        self.unbiased_values = []
        self.epochs_window = EpochsWindow()
        self.query_id = query_id
        self.global_epsilon = epsilon
        self.global_sensitivity = sensitivity
        self.noise_scale = sensitivity / epsilon
        self.biggest_id = biggest_id

    def size(self):
        return len(self.values)

    def update(self, value, unbiazed_value, epochs_window, biggest_id):
        self.values.append(value)
        self.unbiased_values.append(unbiazed_value)
        self.epochs_window.update(epochs_window)
        self.biggest_id = max(biggest_id, self.biggest_id)
