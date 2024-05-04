import math
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Generator, Iterable, SupportsFloat, SupportsIndex, TypeAlias

import pandas as pd
from omegaconf import OmegaConf

from cookiemonster.events import Conversion, Event, Impression

_SupportsFloatOrIndex: TypeAlias = SupportsFloat | SupportsIndex


class Dataset(ABC):
    def __init__(self, config: OmegaConf) -> None:
        """A sequence of Events"""
        self.config = config
        self.impressions_path = os.path.join(
            os.path.dirname(__file__), "..", self.config.impressions_path
        )
        self.conversions_path = os.path.join(
            os.path.dirname(__file__), "..", self.config.conversions_path
        )
        self.impressions_data = None
        self.conversions_data = None

        self.conversions_counter = 0
        self.workload_size = config.workload_size
        assert isinstance(self.workload_size, int)

    @classmethod
    def create(cls, config: OmegaConf):
        match config.name:
            case "criteo":
                return Criteo(config)
            case "microbenchmark":
                return Microbenchmark(config)
            case "patcg":
                return Patcg(config)
            case _:
                raise ValueError(f"Unsupported dataset name: {config.name}")

    @abstractmethod
    def read_impression(self) -> tuple[Event | None, int | None, str | None]:
        pass

    @abstractmethod
    def read_conversion(self) -> tuple[Event | None, int | None, str | None]:
        pass

    def event_reader(self) -> Generator[tuple[str, Event] | None, None, None]:
        assert self.impressions_data is not None
        assert self.conversions_data is not None

        impression_timestamp = conversion_timestamp = 0

        impression = None
        conversion = None

        self.impressions_reader = self.impressions_data.iterrows()
        self.conversions_reader = self.conversions_data.iterrows()

        while True:
            if not impression and impression_timestamp != math.inf:
                impression, impression_timestamp, impression_user_id = (
                    self.read_impression()
                )
            if not conversion and conversion_timestamp != math.inf:
                conversion, conversion_timestamp, conversion_user_id = (
                    self.read_conversion()
                )

            # Feed the event with the earliest timestamp
            if impression_timestamp == math.inf and conversion_timestamp == math.inf:
                break
            elif impression_timestamp <= conversion_timestamp:
                yield (impression_user_id, impression)
                impression = None
            else:
                yield (conversion_user_id, conversion)
                conversion = None


class Microbenchmark(Dataset):
    def __init__(self, config: OmegaConf) -> None:
        super().__init__(config)
        self.impressions_data = pd.read_csv(self.impressions_path)
        self.conversions_data = pd.read_csv(self.conversions_path)

        # TODO: are we ever doing anything with this? Seems that we just run all the conversions
        # self.queries = list(range(self.workload_size))
        # self.conversions_data.query("product_id in @self.queries", inplace=True)

    def read_impression(self) -> tuple[Event | None, int | None, str | None]:
        try:
            _, row = next(self.impressions_reader)

            impression_timestamp = row["timestamp"]
            impression_date = datetime.fromtimestamp(impression_timestamp)
            impression_day = (
                7 * (impression_date.isocalendar().week - 1)
            ) + impression_date.isocalendar().weekday
            impression_epoch = math.ceil(
                impression_day / self.config.num_days_per_epoch
            )
            impression_user_id = row["user_id"]

            filter = "" if math.isnan(row["filter"]) else row["filter"]
            impression = Impression(
                timestamp=impression_timestamp,
                epoch=impression_epoch,
                destination=row["advertiser_id"],
                filter=filter,
                key=str(row["key"]),
                user_id=impression_user_id,
            )
            return impression, impression_timestamp, impression_user_id

        except StopIteration:
            return None, math.inf, None

    def read_conversion(self) -> tuple[Event | None, int | None, str | None]:
        try:
            _, row = next(self.conversions_reader)
            conversion_timestamp = row["timestamp"]
            self.conversions_counter += 1

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

            conversion_epoch = math.ceil(
                conversion_day / self.config.num_days_per_epoch
            )
            epochs_window = (
                math.ceil(earliest_attribution_day / self.config.num_days_per_epoch),
                conversion_epoch,
            )

            conversion_user_id = row["user_id"]
            filter = "" if math.isnan(row["filter"]) else row["filter"]
            conversion = Conversion(
                timestamp=conversion_timestamp,
                id=self.conversions_counter,
                epoch=conversion_epoch,
                destination=row["advertiser_id"],
                attribution_window=attribution_window,
                epochs_window=epochs_window,
                attribution_logic="last_touch",
                partitioning_logic="",
                aggregatable_value=row["amount"],
                aggregatable_cap_value=row["aggregatable_cap_value"],
                filter=filter,
                key=str(row["key"]),
                epsilon=row["epsilon"],
                noise_scale=row["aggregatable_cap_value"] / row["epsilon"],
                user_id=conversion_user_id,
            )

            return conversion, conversion_timestamp, conversion_user_id

        except StopIteration:
            return None, math.inf, None


class Criteo(Dataset):
    def __init__(self, config: OmegaConf) -> None:
        super().__init__(config)
        self.impressions_data = pd.read_csv(self.impressions_path)
        self.conversions_data = pd.read_csv(self.conversions_path)
        self.queries = list(range(self.workload_size))
        self.conversions_data.query("key in @self.queries", inplace=True)

    def read_impression(self) -> tuple[Event | None, int | None, str | None]:
        try:
            _, row = next(self.impressions_reader)

            impression_timestamp = row["click_timestamp"]
            impression_date = datetime.fromtimestamp(impression_timestamp)
            impression_day = (
                7 * (impression_date.isocalendar().week - 1)
            ) + impression_date.isocalendar().weekday
            impression_epoch = math.ceil(
                impression_day / self.config.num_days_per_epoch
            )
            impression_user_id = row["user_id"]
            filter = (
                ""
                if isinstance(row["filter"], _SupportsFloatOrIndex)
                and math.isnan(row["filter"])
                else row["filter"]
            )
            impression = Impression(
                timestamp=impression_timestamp,
                epoch=impression_epoch,
                destination=row["partner_id"],
                filter=filter,
                key=str(row["key"]),
                user_id=impression_user_id,
            )
            return impression, impression_timestamp, impression_user_id

        except StopIteration:
            return None, math.inf, None

    def read_conversion(self) -> tuple[Event | None, int | None, str | None]:
        try:
            _, row = next(self.conversions_reader)
            conversion_timestamp = row["conversion_timestamp"]
            self.conversions_counter += 1

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

            conversion_epoch = math.ceil(
                conversion_day / self.config.num_days_per_epoch
            )
            epochs_window = (
                math.ceil(earliest_attribution_day / self.config.num_days_per_epoch),
                conversion_epoch,
            )

            conversion_user_id = row["user_id"]
            filter = (
                ""
                if isinstance(row["filter"], _SupportsFloatOrIndex)
                and math.isnan(row["filter"])
                else row["filter"]
            )
            conversion = Conversion(
                timestamp=conversion_timestamp,
                id=self.conversions_counter,
                epoch=conversion_epoch,
                destination=row["partner_id"],
                attribution_window=attribution_window,
                epochs_window=epochs_window,
                attribution_logic="last_touch",
                partitioning_logic="",
                aggregatable_value=row["count"],
                aggregatable_cap_value=row["aggregatable_cap_value"],
                filter=filter,
                key=str(row["key"]),
                epsilon=row["epsilon"],
                noise_scale=row["aggregatable_cap_value"] / row["epsilon"],
                user_id=conversion_user_id,
            )

            return conversion, conversion_timestamp, conversion_user_id

        except StopIteration:
            return None, math.inf, None


class Patcg(Dataset):
    def __init__(self, config: OmegaConf) -> None:
        super().__init__(config)
        self.impressions_data = pd.read_csv(self.impressions_path)
        # self.queries = list(range(self.workload_size))

    def iter_conversions_data(self):
        # for chunk in pd.read_csv(self.conversions_path, chunksize=None):
        # chunk.query("key in @self.queries", inplace=True)
        chunk = pd.read_csv(self.conversions_path)
        yield chunk

    def event_reader(self) -> Generator[tuple[str, Event] | None, None, None]:
        assert self.impressions_data is not None
        self.conversions_data = self.iter_conversions_data()

        impression_timestamp = conversion_timestamp = 0

        impression = None
        conversion = None

        self.impressions_reader = self.impressions_data.iterrows()
        self.conversions_reader = next(self.conversions_data).iterrows()
        # self.conversions_reader = self.conversions_data.iterrows()

        while True:
            if not impression and impression_timestamp != math.inf:
                impression, impression_timestamp, impression_user_id = (
                    self.read_impression()
                )
            if not conversion and conversion_timestamp != math.inf:
                conversion, conversion_timestamp, conversion_user_id = (
                    self.read_conversion()
                )

            # Feed the event with the earliest timestamp
            if impression_timestamp == math.inf and conversion_timestamp == math.inf:
                break
            elif impression_timestamp <= conversion_timestamp:
                yield (impression_user_id, impression)
                impression = None
            else:
                yield (conversion_user_id, conversion)
                conversion = None

    def read_impression(self) -> tuple[Event | None, int | None, str | None]:
        try:
            _, row = next(self.impressions_reader)

            impression_timestamp = row["exp_timestamp"]
            impression_date = datetime.fromtimestamp(impression_timestamp)
            impression_day = (
                7 * (impression_date.isocalendar().week - 1)
            ) + impression_date.isocalendar().weekday
            impression_epoch = math.ceil(
                impression_day / self.config.num_days_per_epoch
            )
            impression_user_id = row["device_id"]
            advertiser_id = 1

            filter = "" if math.isnan(row["filter"]) else str(int(row["filter"]))
            impression = Impression(
                timestamp=impression_timestamp,
                epoch=impression_epoch,
                destination=advertiser_id,
                filter=filter,
                key=str(row["key"]),
                user_id=impression_user_id,
            )
            return impression, impression_timestamp, impression_user_id

        except StopIteration:
            return None, math.inf, None

    def read_conversion(self) -> tuple[Event | None, int | None, str | None]:
        try:
            _, row = next(self.conversions_reader)
            conversion_timestamp = row["conv_timestamp"]
            self.conversions_counter += 1

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

            conversion_epoch = math.ceil(
                conversion_day / self.config.num_days_per_epoch
            )
            epochs_window = (
                math.ceil(earliest_attribution_day / self.config.num_days_per_epoch),
                conversion_epoch,
            )

            conversion_user_id = row["device_id"]
            advertiser_id = 1
            filter = "" if math.isnan(row["filter"]) else str(int(row["filter"]))
            key = "" if "key" not in row else str(row["key"])
            conversion = Conversion(
                timestamp=conversion_timestamp,
                id=self.conversions_counter,
                epoch=conversion_epoch,
                destination=advertiser_id,
                attribution_window=attribution_window,
                epochs_window=epochs_window,
                attribution_logic="last_touch",
                partitioning_logic="",
                aggregatable_value=row["conv_amount"],
                aggregatable_cap_value=15,
                filter=filter,
                key=key,
                epsilon=row["epsilon"],
                noise_scale=15 / row["epsilon"],
                user_id=conversion_user_id,
            )

            return conversion, conversion_timestamp, conversion_user_id

        except StopIteration:
            try:
                self.conversions_reader = next(self.conversions_data).iterrows()
                return self.read_conversion()
            except StopIteration:
                return None, math.inf, None
