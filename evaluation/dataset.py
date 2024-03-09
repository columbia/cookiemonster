import pandas as pd
from loguru import logger
from omegaconf import OmegaConf
from typing import Dict, Any, Union
from events import Impression, Conversion


class Dataset:
    def __init__(self, config: OmegaConf) -> None:
        """A sequence of Events"""
        self.config = config
        self.data = pd.read_csv(self.config.dataset.path)

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

    def read_next_event(self) -> (str, Union[Impression, Conversion]):
        for _, row in self.data.iterrows():
            click_day = row["click_day"]
            conversion_day = row["conversion_day"]
            user_id = row["user_id"]
            partner_id = row["partner_id"]
            product_id = row["product_id"]
            product_price = row["product_price"]
            SalesAmountInEuro = row["SalesAmountInEuro"]

            if conversion_day < 0:
                event = Impression(
                    epoch=click_day,
                    destination=partner_id,
                    keys={"product_id": product_id},
                )
            else:
                event = Conversion(
                    destination=partner_id,
                    attribution_window=(max(conversion_day - 30, 0), conversion_day),
                    attribution_logic="last_touch",
                    partitioning_logic="",
                    aggregatable_value=SalesAmountInEuro,
                    aggregatable_cap_value=SalesAmountInEuro,  # TODO: what is the cap?
                    keys_to_match={"product_id": product_id},
                    metadata={"product_price": product_price},
                    epsilon=0.01,
                )

            yield (user_id, event)
