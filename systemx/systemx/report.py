from typing import Dict, Tuple, List, Optional

from systemx.events import Impression


class Report:
    def __init__(self):
        self.histogram: Dict[str, float] = {}

    def add(self, key: str, value: float):
        if key not in self.histogram:
            self.histogram[key] = 0
        self.histogram[key] += value

    def empty(
        self,
    ):
        return self.histogram == {}

    def __add__(self, other) -> "Report":
        report = Report()
        for key, value in self.histogram.items():
            report.add(key, value)
        for key, value in other.histogram.items():
            report.add(key, value)
        return report


class Partition:
    def __init__(
        self,
        epochs_window: Tuple[int, int],
        attribution_logic: str,
        value: Optional[float],
    ) -> None:
        self.epochs_window = epochs_window
        self.attribution_logic = attribution_logic
        self.impressions_per_epoch: Dict[int, List[Impression]] = {}
        self.value = value

    def epochs_window_size(self) -> int:
        return (
            self.epochs_window[1]
            - self.epochs_window[0]
            + 1
        )

    def compute_sensitivity(self, sensitivity_metric) -> float:
        match sensitivity_metric:
            case "L1":
                return sum(list(self.report.histogram.values()))

            case _:
                raise ValueError(
                    f"Unsupported sensitivity metric: {sensitivity_metric}"
                )

    def create_report(self, filter, key_piece: str) -> None:
        self.report = Report()

        match self.attribution_logic:
            case "last_touch":
                # Scan all impressions in epoch and keep the latest one
                epochs = sorted(list(self.impressions_per_epoch.keys()), reverse=True)
                for epoch in epochs:
                    impressions = self.impressions_per_epoch[epoch]
                    if impressions:
                        impression_key = impressions[-1].key
                        if impression_key == "nan":
                            impression_key = ""

                        # Sort impression keys and stringify them
                        bucket_key = impression_key + "_" + filter + "_" + key_piece
                        bucket_value = self.value

                        self.report.add(bucket_key, bucket_value)

                if self.report.empty():
                    bucket_key = "_" + filter + "_" + key_piece
                    bucket_value = 0
                    self.report.add(bucket_key, bucket_value)

            case _:
                raise ValueError(
                    f"Unsupported attribution logic: {self.attribution_logic}"
                )

    def null_report(self) -> None:
        self.report = Report()
