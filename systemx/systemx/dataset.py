import math
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf
from typing import Dict, Any, Union, Tuple

from systemx.events import Impression, Conversion


class Dataset:
    def __init__(self, config: OmegaConf) -> None:
        """A sequence of Events"""
        self.config = config
        self.impressions_data = pd.read_csv(self.config.impressions_path)
        self.conversions_data = pd.read_csv(self.config.conversions_path)

    @classmethod
    def create(cls, config: OmegaConf):
        match config.name:
            case "criteo":
                return Criteo(config)
            case _:
                raise ValueError(f"Unsupported config name: {config.name}")


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
                impression_epoch = row["click_day"] // self.config.num_days_per_epoch
                impression = Impression(
                    epoch=impression_epoch,
                    destination=row["partner_id"],
                    filter=row["filter"],
                    key=str(row["key"]),
                )
                impression_timestamp = row["click_timestamp"]
                impression_user_id = row["user_id"]
                return impression, impression_timestamp, impression_user_id

            except StopIteration:
                return None, math.inf, None

        def read_conversion():
            try:
                _, row = next(conversions_reader)
                conversion_epoch = (
                    row["conversion_day"] // self.config.num_days_per_epoch
                )
                conversion_timestamp = row["conversion_timestamp"]
                num_epochs_attribution_window = (
                    self.config.num_days_attribution_window - 1
                ) // self.config.num_days_per_epoch

                conversion = Conversion(
                    timestamp=conversion_timestamp,
                    destination=row["partner_id"],
                    attribution_window=(
                        max(conversion_epoch - num_epochs_attribution_window, 0),
                        conversion_epoch,
                    ),
                    attribution_logic="last_touch",
                    partitioning_logic="",
                    aggregatable_value=row["count"],
                    aggregatable_cap_value=row["aggregatable_cap_value"],
                    filter=row["filter"],
                    key=str(row["key"]),
                    epsilon=0.01,
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
