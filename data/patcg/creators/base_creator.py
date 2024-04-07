import os
import logging
from abc import ABC, abstractmethod

from omegaconf import DictConfig

logging.basicConfig(level=logging.INFO)

# if os.getenv("USE_PANDAS", "false").lower() == "true":
import pandas as pd
# else:
    # import modin.pandas as pd

# os.environ["MODIN_ENGINE"] = "ray"


class BaseCreator(ABC):

    impressions_file = os.path.join(
        os.path.dirname(__file__), "..", "publishers", "converter_impressions.csv"
    )
    conversions_file = os.path.join(
        os.path.dirname(__file__), "..", "advertisers", "conversions.csv"
    )

    def __init__(
        self, config: DictConfig, impressions_filename: str, conversions_filename: str
    ):
        self.config = config
        self.df: pd.DataFrame | None = None
        self.impressions_filename = os.path.join(
            os.path.dirname(__file__), "..", impressions_filename
        )
        self.conversions_filename = os.path.join(
            os.path.dirname(__file__), "..", conversions_filename
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
        
        self.logger.info("writing impressions")
        impressions.to_csv(BaseCreator.impressions_file, header=True, index=False)
        self.logger.info(f"dataset written to {self.impressions_filename}")

        del impressions

        self.logger.info("reading in PATCG conversions...")
        conversions = self._read_dataframe(BaseCreator.conversions_file)
        
        self.logger.info("specializing conversions...")
        conversions = self.specialize_conversions(conversions)

        self.logger.info("creating the conversions...")
        conversions = self.create_conversions(conversions)

        self.logger.info("writing conversions")
        conversions.to_csv(BaseCreator.conversions_file, header=True, index=False)
        self.logger.info(f"dataset written to {self.conversions_filename}")
