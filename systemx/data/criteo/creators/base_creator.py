from abc import ABC, abstractmethod
import logging
import numpy as np
import os

if os.getenv("LOCAL", "false").lower() == "true":
    import pandas as pd
else:
    import modin.pandas as pd

os.environ["MODIN_ENGINE"] = "ray"

class BaseCreator(ABC):
    
    data_file = os.path.join(os.path.dirname(__file__), os.getenv("CRITEO_DATA_FILE_PATH", "../Criteo_Conversion_Search/CriteoSearchData"))

    def __init__(self, impressions_filename: str, conversions_filename: str):
        self.df: pd.DataFrame | None = None
        self.impressions: pd.DataFrame | None = None
        self.conversions: pd.DataFame | None = None
        self.impressions_filename = os.path.join(os.path.dirname(__file__), '..', impressions_filename)
        self.conversions_filename = os.path.join(os.path.dirname(__file__), '..', conversions_filename)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s"))
        self.logger.addHandler(stream_handler)

    def _read_dataframe(self) -> pd.DataFrame:
        dtype = {
            "Sale": np.int32,
            "SalesAmountInEuro": np.float64,
            "Time_delay_for_conversion": np.int32,
            "click_timestamp": np.int32,
            "nb_clicks_1week": pd.Int64Dtype(),
            "product_price": np.float64,
            "product_age_group": str,
            "device_type": str,
            "audience_id": str,
            "product_gender": str,
            "product_brand": str,
            "product_category1": str,
            "product_category2": str,
            "product_category3": str,
            "product_category4": str,
            "product_category5": str,
            "product_category6": str,
            "product_category7": str,
            "product_country": str,
            "product_id": str,
            "product_title": str,
            "partner_id": str,
            "user_id": str,
        }
        na_values = {
            "click_timestamp": "0",
            "nb_clicks_1week": "-1",
            "product_price": "-1",
            "product_age_group": "-1",
            "device_type": "-1",
            "audience_id": "-1",
            "product_gender": "-1",
            "product_brand": "-1",
            "product_category1": "-1",
            "product_category2": "-1",
            "product_category3": "-1",
            "product_category4": "-1",
            "product_category5": "-1",
            "product_category6": "-1",
            "product_category7": "-1",
            "product_country": "-1",
            "product_id": "-1",
            "product_title": "-1",
            "partner_id": "-1",
            "user_id": "-1",
        }
        columns_to_drop = [
            "product_category1",
            "product_category2",
            "product_category3",
            "product_category4",
            "product_category5",
            "product_category6",
            "product_category7",
            "nb_clicks_1week",
            "device_type",
            "product_title",
            "product_brand",
            "product_gender",
            "audience_id",
            "product_age_group",
            "product_country",
        ]

        df = pd.read_csv(
            BaseCreator.data_file,
            names=dtype.keys(),
            dtype=dtype,
            na_values=na_values,
            header=None,
            sep="\t",
        )
        df = df.drop(columns=columns_to_drop)
        return df
    
    @abstractmethod
    def specialize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def create_impressions(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def create_conversions(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def create_datasets(self) -> None:
        self.logger.info("reading in criteo dataset...")
        self.df = self._read_dataframe()

        self.logger.info("specializing the dataset...")
        self.df = self.specialize_df(self.df)

        self.logger.info("creating the impressions...")
        self.impressions = self.create_impressions(self.df)

        self.logger.info("creating the conversions...")
        self.conversions = self.create_conversions(self.df)

        self.logger.info("writing the datasets out to the file paths specified")
        df_and_fp = [
            (self.impressions, self.impressions_filename),
            (self.conversions, self.conversions_filename)
        ]
        
        for (df, filepath) in df_and_fp:
            if os.path.exists(filepath):
                os.remove(filepath)

            df.to_csv(filepath, header=True, index=False)
            self.logger.info(f"dataset written to {filepath}")
