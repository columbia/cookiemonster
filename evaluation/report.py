from collections import namedtuple
from typing import Dict, Any, Union, Tuple, Optional

from events import Impression
from utils import attribution_window_to_list


class Report:
    def __init__(self):
        self.histogram: Dict[str, float] = {}

    def add(self, key: str, value: float):
        if key not in self.histogram:
            self.histogram[key] = 0
        self.histogram[key] += value

    def __add__(self, other) -> Report:
        report = Report()
        for key, value in self.histogram.items():
            report.add(key, value)
        for key, value in other.histogram.items():
            report.add(key, value)
        return report


class Partition:
    def __init__(
        self,
        attribution_window: Tuple[int, int],
        attribution_logic: str,
        value: Optional[float],
    ) -> None:
        self.attribution_window = attribution_window
        self.attribution_logic = attribution_logic
        self.impressions_per_epoch: Dict[int, List[Impression]] = {}
        self.value = value

    def attribution_window_size(self) -> int:
        return self.attribution_window[1] - self.attribution_window[0] + 1

    def compute_sensitivity(self, sensitivity_metric) -> float:
        match sensitivity_metric:
            case "L1":
                return sum(list(self.report.values()))

            case _:
                raise ValueError(
                    f"Unsupported sensitivity metric: {sensitivity_metric}"
                )

    def create_report(self) -> None:
        self.report = Report()

        match partition.attribution_logic:
            case "last_touch":
                # Scan all impressions in epoch and keep the latest one
                epochs = list(self.impressions_per_epoch.keys()).sort(reverse=True)
                for epoch in epochs:
                    impressions = self.impressions_per_epoch[epoch]
                    if impressions:
                        impression_keys = impressions[-1].keys

                        # Sort impression keys and stringify them
                        bucket_key = str(
                            {
                                key: impression_keys[key]
                                for key in sorted(impression_keys.keys())
                            }
                        )
                        bucket_value = self.value

                        self.report.add(bucket_key, bucket_value)
            case _:
                raise ValueError(
                    f"Unsupported attribution logic: {partition.attribution_logic}"
                )

    def null_report(self) -> None:
        self.report = Report()
