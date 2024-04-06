from abc import ABC, abstractmethod
import logging
import numpy as np
import os

from omegaconf import DictConfig

if os.getenv("USE_PANDAS", "false").lower() == "true":
    import pandas as pd
else:
    import modin.pandas as pd

os.environ["MODIN_ENGINE"] = "ray"


class BaseCreator(ABC):

    impressions_file = os.path.join(os.path.dirname(__file__), "..", "publishers", "converter_impressions.csv")
    conversions_file = os.path.join(os.path.dirname(__file__), "..", "advertisers", "conversions.csv")

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
    def specialize_df(self, impressions_df: pd.DataFrame, conversions_df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def create_impressions(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def create_conversions(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def create_datasets(self) -> None:
        self.logger.info("reading in PATCG dataset...")
        self.impressions_df = self._read_dataframe(BaseCreator.impressions_file)
        self.conversions_df = self._read_dataframe(BaseCreator.conversions_file)

        self.logger.info("specializing the dataset...")
        self.impressions_df, self.conversions_df = self.specialize_df(self.impressions_df, self.conversions_df)

        self.logger.info("creating the impressions...")
        impressions = self.create_impressions(self.df)

        self.logger.info("creating the conversions...")
        conversions = self.create_conversions(self.df)

        self.logger.info("writing the datasets out to the file paths specified")
        df_and_fp = [
            (impressions, self.impressions_filename),
            (conversions, self.conversions_filename),
        ]

        for df, filepath in df_and_fp:
            if os.path.exists(filepath):
                os.remove(filepath)

            df.to_csv(filepath, header=True, index=False)
            self.logger.info(f"dataset written to {filepath}")
