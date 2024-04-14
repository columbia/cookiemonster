from abc import ABC, abstractmethod
import pandas as pd
import logging
import numpy as np
from omegaconf import DictConfig
import os


class BaseCreator(ABC):

    data_file = os.path.join(
        os.path.dirname(__file__),
        os.getenv(
            "CRITEO_DATA_FILE_PATH", "../Criteo_Conversion_Search/CriteoSearchData"
        ),
    )

    def __init__(
        self,
        config: DictConfig,
        impressions_filename: str,
        conversions_filename: str,
        augmented_impressions_filename: str | None = None,
    ):
        self.config = config
        self.df: pd.DataFrame | None = None
        self.impressions_filename = os.path.join(
            os.path.dirname(__file__), "..", impressions_filename
        )
        self.conversions_filename = os.path.join(
            os.path.dirname(__file__), "..", conversions_filename
        )
        if augmented_impressions_filename:
            self.augmented_impressions_filename = os.path.join(
                os.path.dirname(__file__), "..", augmented_impressions_filename
            )
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(filename)s:%(lineno)s -- %(message)s"
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        file_handler = logging.FileHandler("criteo_dataset_creator.log", mode="w")
        file_handler.setFormatter(formatter)

        self.logger.addHandler(stream_handler)
        self.logger.addHandler(file_handler)

        self.advertiser_column_name = "partner_id"
        self.product_column_name = "product_id"
        self.user_column_name = "user_id"
        self.conversion_columns_to_drop = [
            "SalesAmountInEuro",
            "product_price",
            "nb_clicks_1week",
            "Time_delay_for_conversion",
            "Sale",
            "click_timestamp",
        ]
        self.impression_columns_to_use = [
            "click_timestamp",
            "user_id",
            "partner_id",
            "filter",
        ]

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

        df = pd.read_csv(
            BaseCreator.data_file,
            names=dtype.keys(),
            dtype=dtype,
            na_values=na_values,
            header=None,
            sep="\t",
        )
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

    @abstractmethod
    def augment_impressions(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def create_datasets(self) -> None:
        self.logger.info("reading in criteo dataset...")
        self.df = self._read_dataframe()

        self.logger.info("specializing the dataset...")
        self.df = self.specialize_df(self.df)

        self.logger.info("creating the impressions...")
        idf = self.create_impressions(self.df)
        impressions = idf[[*self.impression_columns_to_use, "key"]]
        impressions = impressions.sort_values(by=["click_timestamp"])

        self.logger.info("creating the conversions...")
        cdf = self.create_conversions(self.df)
        conversions = cdf.drop(columns=self.conversion_columns_to_drop)

        self.logger.info("writing the datasets out to the file paths specified")
        df_and_fp = [
            (impressions, self.impressions_filename),
            (conversions, self.conversions_filename),
        ]

        for d, filepath in df_and_fp:
            if os.path.exists(filepath):
                os.remove(filepath)

            d.to_csv(filepath, header=True, index=False)
            self.logger.info(f"dataset written to {filepath}")

        if self.augmented_impressions_filename:
            self.logger.info("augmenting impressions...")
            aidf = self.augment_impressions(cdf)
            if not aidf.empty:
                augmented_impressions = aidf[[*self.impression_columns_to_use, "key"]]
                augmented_impressions = pd.concat([impressions, augmented_impressions])
                augmented_impressions = augmented_impressions.sort_values(
                    by=["click_timestamp"]
                )

                fp = self.augmented_impressions_filename
                if os.path.exists(fp):
                    os.remove(fp)

                augmented_impressions.to_csv(fp, header=True, index=False)
                self.logger.info(f"dataset written to {fp}")
