import math
import os
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf
from typing import Dict, Any, Union, Tuple
from datetime import datetime

from systemx.events import Impression, Conversion


class Dataset:
    def __init__(self, config: OmegaConf) -> None:
        """A sequence of Events"""
        self.config = config
        impressions_filename = os.path.join(os.path.dirname(__file__), "..", self.config.impressions_path)
        conversions_filename = os.path.join(os.path.dirname(__file__), "..", self.config.conversions_path)
        self.impressions_data = pd.read_csv(impressions_filename)
        self.conversions_data = pd.read_csv(conversions_filename)

    @classmethod
    def create(cls, config: OmegaConf):
        match config.name:
            case "criteo":
                return Criteo(config)
            case "synthetic":
                return Synthetic(config)
            case _:
                raise ValueError(f"Unsupported config name: {config.name}")


class Synthetic(Dataset):
    def __init__(self, config: OmegaConf) -> None:
        super().__init__(config)
        self.workload_size = config.workload_size
        assert isinstance(self.workload_size, int)

        self.queries = list(range(self.workload_size))
        self.impressions_data.query("product_id in @self.queries", inplace=True)
        self.conversions_data.query("product_id in @self.queries", inplace=True)

    def event_reader(self):
        impression_timestamp = conversion_timestamp = 0

        impression = None
        conversion = None

        impressions_reader = self.impressions_data.iterrows()
        conversions_reader = self.conversions_data.iterrows()

        def read_impression():
            try:
                _, row = next(impressions_reader)

                impression_timestamp = row["timestamp"]
                impression_date = datetime.fromtimestamp(impression_timestamp)
                impression_day = (
                    7 * (impression_date.isocalendar().week - 1)
                ) + impression_date.isocalendar().weekday
                impression_epoch = impression_day // self.config.num_days_per_epoch

                impression = Impression(
                    timestamp=impression_timestamp,
                    epoch=impression_epoch,
                    destination=row["advertiser_id"],
                    filter=row["filter"],
                    key=str(row["key"]),
                )
                impression_user_id = row["user_id"]
                return impression, impression_timestamp, impression_user_id

            except StopIteration:
                return None, math.inf, None

        def read_conversion():
            try:
                _, row = next(conversions_reader)
                conversion_timestamp = row["timestamp"]

                num_seconds_attribution_window = (
                    self.config.num_days_attribution_window * 24 * 60 * 60
                )
                earliest_attribution_timestamp = max(
                    conversion_timestamp - num_seconds_attribution_window, 0
                )

                attribution_window = (
                    earliest_attribution_timestamp,
                    conversion_timestamp,
                )

                earliest_attribution_date = datetime.fromtimestamp(
                    earliest_attribution_timestamp
                )
                earliest_attribution_day = (
                    7 * (earliest_attribution_date.isocalendar().week - 1)
                ) + earliest_attribution_date.isocalendar().weekday

                conversion_date = datetime.fromtimestamp(conversion_timestamp)
                conversion_day = (
                    7 * (conversion_date.isocalendar().week - 1)
                ) + conversion_date.isocalendar().weekday

                conversion_epoch = conversion_day // self.config.num_days_per_epoch
                epochs_window = (
                    earliest_attribution_day // self.config.num_days_per_epoch,
                    conversion_epoch,
                )

                conversion = Conversion(
                    timestamp=conversion_timestamp,
                    epoch=conversion_epoch,
                    destination=row["advertiser_id"],
                    attribution_window=attribution_window,
                    epochs_window=epochs_window,
                    attribution_logic="last_touch",
                    partitioning_logic="",
                    aggregatable_value=row["amount"],
                    aggregatable_cap_value=row["aggregatable_cap_value"],
                    filter=row["filter"],
                    key=str(row["key"]),
                    epsilon=row["epsilon"],
                )

                conversion_user_id = row["user_id"]
                return conversion, conversion_timestamp, conversion_user_id

            except StopIteration:
                return None, math.inf, None

        while True:
            if not impression and impression_timestamp != math.inf:
                impression, impression_timestamp, impression_user_id = read_impression()
            if not conversion and conversion_timestamp != math.inf:
                conversion, conversion_timestamp, conversion_user_id = read_conversion()

            # Feed the event with the earliest timestamp
            if impression_timestamp == math.inf and conversion_timestamp == math.inf:
                break
            elif impression_timestamp <= conversion_timestamp:
                yield (impression_user_id, impression)
                impression = None
            else:
                yield (conversion_user_id, conversion)
                conversion = None

        yield None


class Criteo(Dataset):
    def __init__(self, config: OmegaConf) -> None:
        super().__init__(config)

    def event_reader(self):
        impression_timestamp = conversion_timestamp = 0

        impression = None
        conversion = None

        impressions_reader = self.impressions_data.iterrows()
        conversions_reader = self.conversions_data.iterrows()

        def read_impression():
            try:
                _, row = next(impressions_reader)

                impression_timestamp = row["click_timestamp"]
                impression_date = datetime.fromtimestamp(impression_timestamp)
                impression_day = (
                    7 * (impression_date.isocalendar().week - 1)
                ) + impression_date.isocalendar().weekday
                impression_epoch = impression_day // self.config.num_days_per_epoch

                impression = Impression(
                    timestamp=impression_timestamp,
                    epoch=impression_epoch,
                    destination=row["partner_id"],
                    filter=row["filter"],
                    key=str(row["key"]),
                )
                impression_user_id = row["user_id"]
                return impression, impression_timestamp, impression_user_id

            except StopIteration:
                return None, math.inf, None

        def read_conversion():
            try:
                _, row = next(conversions_reader)
                # conversion_day = row["conversion_day"]
                conversion_timestamp = row["conversion_timestamp"]

                num_seconds_attribution_window = (
                    self.config.num_days_attribution_window * 24 * 60 * 60
                )
                earliest_attribution_timestamp = max(
                    conversion_timestamp - num_seconds_attribution_window, 0
                )

                attribution_window = (
                    earliest_attribution_timestamp,
                    conversion_timestamp,
                )

                earliest_attribution_date = datetime.fromtimestamp(
                    earliest_attribution_timestamp
                )
                earliest_attribution_day = (
                    7 * (earliest_attribution_date.isocalendar().week - 1)
                ) + earliest_attribution_date.isocalendar().weekday

                conversion_date = datetime.fromtimestamp(conversion_timestamp)
                conversion_day = (
                    7 * (conversion_date.isocalendar().week - 1)
                ) + conversion_date.isocalendar().weekday

                conversion_epoch = conversion_day // self.config.num_days_per_epoch
                epochs_window = (
                    earliest_attribution_day // self.config.num_days_per_epoch,
                    conversion_epoch,
                )

                conversion = Conversion(
                    timestamp=conversion_timestamp,
                    epoch=conversion_epoch,
                    destination=row["partner_id"],
                    attribution_window=attribution_window,
                    epochs_window=epochs_window,
                    attribution_logic="last_touch",
                    partitioning_logic="",
                    aggregatable_value=row["count"],
                    aggregatable_cap_value=row["aggregatable_cap_value"],
                    filter=row["filter"],
                    key=str(row["key"]),
                    epsilon=row["epsilon"],
                )

                conversion_user_id = row["user_id"]
                return conversion, conversion_timestamp, conversion_user_id

            except StopIteration:
                return None, math.inf, None

        while True:
            if not impression and impression_timestamp != math.inf:
                impression, impression_timestamp, impression_user_id = read_impression()
            if not conversion and conversion_timestamp != math.inf:
                conversion, conversion_timestamp, conversion_user_id = read_conversion()

            # Feed the event with the earliest timestamp
            if impression_timestamp == math.inf and conversion_timestamp == math.inf:
                break
            elif impression_timestamp <= conversion_timestamp:
                yield (impression_user_id, impression)
                impression = None
            else:
                yield (conversion_user_id, conversion)
                conversion = None

        yield None
