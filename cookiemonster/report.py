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

        key = keys[0]

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
        report = ScalarReport()
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
        report = HistogramReport(impression_buckets=self.impression_buckets)
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
