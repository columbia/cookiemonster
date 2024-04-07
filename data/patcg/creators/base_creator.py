import os
import logging
import pandas as pd
from omegaconf import DictConfig
from abc import ABC, abstractmethod
import dask.dataframe as dd


logging.basicConfig(level=logging.INFO)


class BaseCreator(ABC):

    impressions_file = os.path.join(
        os.path.dirname(__file__),
        "..",
        "publishers",
        "renamed_filtered_impressions.csv",
    )
    conversions_file = os.path.join(
        os.path.dirname(__file__), "..", "advertisers", "renamed_conversions.csv"
    )

    def __init__(
        self, config: DictConfig, impressions_path: str, conversions_path: str
    ):
        self.config = config
        self.df: pd.DataFrame | None = None
        self.impressions_path = os.path.join(
            os.path.dirname(__file__), "..", impressions_path
        )
        self.conversions_path = os.path.join(
            os.path.dirname(__file__), "..", conversions_path
        )
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def _read_dataframe(self, path) -> pd.DataFrame:
        df = pd.read_csv(path)
        return df

    @abstractmethod
    def specialize_impressions(self, impressions: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def specialize_conversions(self, conversions: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def create_impressions(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def create_conversions(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def create_datasets(self) -> None:
        self.logger.info("reading in PATCG impressions...")
        impressions = self._read_dataframe(BaseCreator.impressions_file)

        self.logger.info("specializing impressions...")
        impressions = self.specialize_impressions(impressions)

        self.logger.info("creating the impressions...")
        impressions = self.create_impressions(impressions)

        self.logger.info(f"writing impressions to {self.impressions_path}")
        impressions.to_csv(self.impressions_path, header=True, index=False)
        self.logger.info(f"dataset written to {self.impressions_path}")

        del impressions

        self.logger.info("reading in PATCG conversions...")
        conversions = self._read_dataframe(BaseCreator.conversions_file)

        self.logger.info("specializing conversions...")
        conversions = self.specialize_conversions(conversions)

        self.logger.info("creating the conversions...")
        conversions = self.create_conversions(conversions)
