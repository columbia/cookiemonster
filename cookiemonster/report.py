from typing import Dict, List, Optional, Tuple

import numpy as np

from cookiemonster.events import Impression


class Report:
    """Stores the result of an attribution report.
    Also stores metadata about which query created the report.
    """

    def get_query_id(self):
        raise NotImplementedError

    def get_value(self):
        raise NotImplementedError

    def __add__(self, other) -> "Report":
        raise NotImplementedError


class ScalarReport(Report):
    def __init__(self):
        # Implemented as a histogram for historical reasons
        self.histogram: Dict[str, float] = {}

    def get_query_id(self):

        keys = list(self.histogram.keys())
        if len(keys) > 1:
            raise ValueError("Scalar report with multiple queries")
        if len(keys) == 0:
            raise ValueError(
                "Empty scalar report. Should not happen thanks to `create_report`"
            )

        key = key[0]

        # TODO: use proper tuples/objects instead of this string encoding
        _, filter, conversion_key = key.split("#")
        query_id = f"{filter}#{conversion_key}"
        return query_id

    def get_value(self):
        query_id = self.get_query_id()
        impression_key = ""
        key = f"{impression_key}#{query_id}"
        return self.histogram.get(key, 0)

    def __add__(self, other):
        report = Report()
        for key, value in self.histogram.items():
            report.add(key, value)
        for key, value in other.histogram.items():
            report.add(key, value)
        return report

    def add(self, key: str, value: float):
        if key not in self.histogram:
            self.histogram[key] = 0
        self.histogram[key] += value

    def empty(self):
        return self.histogram == {}

    def __str__(self):
        return str(self.histogram)


class HistogramReport(Report):
    def __init__(self, impression_buckets: List[str]):
        self.histogram: Dict[str, float] = {}
        self.impression_buckets = sorted(impression_buckets)

    def get_query_id(self):
        # Check that there is a single query (we don't support multi-queries for now)
        query_id = None
        for key in self.histogram.keys():
            _, filter, conversion_key = key.split("#")
            if query_id is None:
                query_id = f"{filter}#{conversion_key}"
            elif query_id != f"{filter}#{conversion_key}":
                raise ValueError("Multiple queries in the same report")

        return query_id

    def get_value(self):
        query_id = self.get_query_id()

        # Collect the relevant buckets, pad with 0 if not present
        values = []
        for impression_key in self.impression_buckets:
            key = f"{impression_key}#{query_id}"
            values.append(self.histogram.get(key, 0))

        return np.array(values)

    def __add__(self, other):
        # TODO: optimize this, sounds inefficient
        report = Report()
        for key, value in self.histogram.items():
            report.add(key, value)
        for key, value in other.histogram.items():
            report.add(key, value)
        return report

    def add(self, key: str, value: float):
        if key not in self.histogram:
            self.histogram[key] = 0
        self.histogram[key] += value

    def empty(self):
        return self.histogram == {}

    def __str__(self):
        return str(self.histogram)


class VectorReport(Report):
    def __init__(self, vector_size: int):
        self.vector = np.zeros(vector_size)

    # TODO: implement the rest for the Meta use case with linear models


# TODO: make this as dumb as possible. Shouldn't need value or attribution logic
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

    def create_report(
        self, filter, key_piece: str, bias_counting_strategy=None
    ) -> None:
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
                for epoch in range(y, x - 1, -1):
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
                        bucket_key = (
                            default_bucket_prefix + "#" + filter + "#" + key_piece
                        )
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
