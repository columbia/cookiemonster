from cookiemonster.utils import EpochsWindow


class QueryBatch:
    def __init__(
        self, query_id: str, noise_scale:float, biggest_id: int
    ) -> None:
        self.values = []
        self.unbiased_values = []
        self.epochs_window = EpochsWindow()
        self.query_id = query_id
        # self.global_epsilon = epsilon
        # self.global_sensitivity = sensitivity
        # self.noise_scale = sensitivity / epsilon
        self.noise_scale = noise_scale
        self.biggest_id = biggest_id
        self.global_sensitivity = 0

    def size(self):
        return len(self.values)

    # def update(self, value, unbiazed_value, epochs_window, biggest_id):
    #     self.values.append(value)
    #     self.unbiased_values.append(unbiazed_value)
    #     self.epochs_window.update(epochs_window)
    #     self.biggest_id = max(biggest_id, self.biggest_id)

    def add_report(self, value, unbiased_value, global_sensitivity, epochs_window, biggest_id: int):
        self.values.append(value)
        self.unbiased_values.append(unbiased_value)
        self.epochs_window.update(epochs_window)
        self.biggest_id = max(biggest_id, self.biggest_id)

        # Constraint: 1 device-epoch sends at most 1 report per batch
        # Otherwise we need to take a sum
        self.global_sensitivity = max(self.global_sensitivity, global_sensitivity)

    def get_global_epsilon(self) -> float:
        return self.global_sensitivity / self.noise_scale