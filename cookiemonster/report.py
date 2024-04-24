import copy
from typing import Dict, List, Optional, Tuple

import numpy as np

from cookiemonster.events import Impression


class Report:
    def __init__(self):
        self.histogram: Dict[str, float] = {}

    def add(self, key: str, value: float):
        if key not in self.histogram:
            self.histogram[key] = 0
        self.histogram[key] += value

    def empty(self):
        return self.histogram == {}
    
    def get_query_value(self, impression_buckets):
        
        # Check that there is a single query (we don't support multi-queries for now)
        query_id = None
        for key in self.histogram.keys():
            _, filter, conversion_key = key.split("#")
            if query_id is None:
                query_id = f"{filter}#{conversion_key}"
            elif query_id != f"{filter}#{conversion_key}":
                raise ValueError("Multiple queries in the same report")
        
        # Collect the relevant buckets, pad with 0 if not present            
        values = []
        for impression_key in impression_buckets:
            key = f"{impression_key}#{query_id}"
            values.append(self.histogram.get(key, 0))
                            
        # Use a single float for scalar queries, np array otherwise
        value = values[0] if len(values) == 1 else np.array(values)
        return query_id, value
    
    
    def __add__(self, other) -> "Report":
        report = Report()
        for key, value in self.histogram.items():
            report.add(key, value)
        for key, value in other.histogram.items():
            report.add(key, value)
        return report
    
    def __str__(self):
        return str(self.histogram)


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
        self.report = None
        self.unbiased_report = None

    def epochs_window_size(self) -> int:
        return self.epochs_window[1] - self.epochs_window[0] + 1

    def compute_sensitivity(self, sensitivity_metric) -> float:
        assert self.unbiased_report is not None
        match sensitivity_metric:
            case "L1":
                return sum(list(self.unbiased_report.histogram.values()))

            case _:
                raise ValueError(
                    f"Unsupported sensitivity metric: {sensitivity_metric}"
                )

    def create_report(self, filter, key_piece: str, bias_counting_strategy = None) -> None:
        report = Report()

        match (self.attribution_logic, bias_counting_strategy):
            case ("last_touch", None):
                # Scan all impressions in epochs and keep the latest one
                epochs = sorted(list(self.impressions_per_epoch.keys()), reverse=True)
                for epoch in epochs:
                    impressions = self.impressions_per_epoch[epoch]
                    
                    if impressions:
                        impression_key = impressions[-1].key
                        if impression_key == "nan":
                            impression_key = ""

                        # Sort impression keys and stringify them
                        bucket_key = impression_key + "#" + filter + "#" + key_piece
                        bucket_value = self.value

                        report.add(bucket_key, bucket_value)
                        break

                if report.empty():
                    bucket_key = "#" + filter + "#" + key_piece
                    bucket_value = 0
                    report.add(bucket_key, bucket_value)
                    
            case ("last_touch", kappa):
                # Keep a default bucket to count epochs with no impressions (i.e. relevant events)
                assert isinstance(kappa, float) or isinstance(kappa, int)
                
                already_attributed = False
                                
                # Browse all the epochs, even those with no impressions
                (x, y) = self.epochs_window               
                for epoch in range(y, x-1, -1):
                    impressions = self.impressions_per_epoch.get(epoch, [])

                    if impressions and not already_attributed:
                        # For epochs with impressions but already_attributed, there is no bias
                        impression_key = impressions[-1].key
                        if impression_key == "nan":
                            impression_key = "main"

                        # Sort impression keys and stringify them
                        bucket_key = impression_key + "#" + filter + "#" + key_piece
                        bucket_value = self.value
                        report.add(bucket_key, bucket_value)
                        
                        already_attributed = True                            
                    else:
                        # This epoch has no impressions. 
                        # Maybe it is really the case, or maybe it got zeroed-out by a filter
                        default_bucket_prefix = "empty"
                        bucket_key = default_bucket_prefix + "#" + filter + "#" + key_piece
                        bucket_value = kappa
                        report.add(bucket_key, bucket_value)
                        
            case _:
                raise ValueError(
                    f"Unsupported attribution logic: {self.attribution_logic}"
                )
        return report

    def null_report(self) -> None:
        # Set default value 0 to all histogram bins
        for query_id in self.report.histogram.keys():
            self.report.histogram[query_id] = 0

    def __str__(self):
        return str(self.__dict__)