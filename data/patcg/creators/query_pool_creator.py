import os
import math
from omegaconf import DictConfig
from data.patcg.creators.base_creator import BaseCreator, pd


class QueryPoolDatasetCreator(BaseCreator):

    def __init__(self, config: DictConfig) -> None:
        super().__init__(
            config,
            "v375_patcg_impressions.csv",
            "v375_patcg_conversions.csv",
        )
        self.dimension_domain_size = 10
        self.user_column_name = "device_id"

    def create_impressions(self, impressions: pd.DataFrame) -> pd.DataFrame:
        impressions = impressions[["exp_timestamp", "device_id"]]
        impressions = impressions.sort_values(by=["exp_timestamp"])
        impressions["filter"] = ""
        impressions["key"] = ""
        return impressions

    def _process_batches(self, conversions: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Enforcing cap on user contribution...")
        batch_size = self.config.max_batch_size

        seen_users = set()
        row_count = 0

        def __mark_include(row: pd.Series):
            nonlocal seen_users
            nonlocal row_count
            if row_count == batch_size:
                row_count = 0
                seen_users = set()

            user = row[self.user_column_name]
            if user in seen_users:
                return False
            else:
                seen_users.add(user)
                row_count += 1
                return True

        queries = []
        for dimension_value in range(self.dimension_domain_size):
            self.logger.info(f"Processing query {dimension_value}...")
            query_result = conversions.loc[conversions["product_id"] == dimension_value]
            query_result = query_result.sort_values(by=["conv_timestamp"])

            query_result = query_result.assign(
                include=query_result.apply(
                    lambda row: __mark_include(row),
                    axis=1,
                ),
            )
            query_result = query_result.loc[query_result.include]
            query_result = query_result.drop(columns=["include"])
            seen_users = set()
            row_count = 0

            # now split the query_result into its batches
            query_result = query_result.reset_index(drop=True)
            query_result_length = query_result.shape[0]
            nbatches = query_result_length // batch_size
            remaining = query_result_length % batch_size
            total_batches = nbatches + 1 if remaining > 0 else nbatches

            i = 0
            self.logger.info(f"Processing batches...")
            while i < nbatches:
                start = i * batch_size
                end = (i + 1) * batch_size
                unique_query_key = dimension_value * total_batches + i
                query_result.loc[start:end, "query_key"] = str(unique_query_key)
                print(unique_query_key)
                query_result.loc[start:end, "epsilon"] = self._set_epsilon()
                i += 1

            i = i * batch_size
            if (
                i < query_result_length
                and query_result_length - i >= self.config.min_batch_size
            ):
                unique_query_key = dimension_value * total_batches + nbatches
                print(unique_query_key)
                query_result.loc[i:, "query_key"] = str(unique_query_key)
                query_result.loc[i:, "epsilon"] = self._set_epsilon()
            
            queries.append(query_result)

        conversions = pd.concat(queries, ignore_index=True)
        return conversions

    def _clip_contribution_value(self, conversions: pd.DataFrame) -> pd.DataFrame:
        conversions["conv_amount"] = conversions["conv_amount"].clip(
            upper=self.config.cap_value
        )
        return conversions

    def _set_epsilon(self) -> pd.DataFrame:
        [a, b] = self.config.accuracy
        expected_result = 1500 #1000 * 5
        epsilon = self.config.cap_value * math.log(1 / b) / (a * expected_result)
        return epsilon

    def create_conversions(self, conversions: pd.DataFrame) -> pd.DataFrame:

        # count    7.472919e+07
        # mean     1.213794e+01
        # std      1.593148e+01
        # min      2.000000e-02
        # 25%      3.740000e+00
        # 50%      7.350000e+00
        # 75%      1.445000e+01
        # max      1.882690e+03
        # Name: conv_amount, dtype: float64

        conversions.rename(columns={"conv_attribute_2": "product_id"}, inplace=True)
        conversions = conversions[
            ["conv_timestamp", "device_id", "product_id", "conv_amount"]
        ]

        conversions = self._process_batches(conversions)
        conversions = conversions.sort_values(by=["conv_timestamp"])
        conversions = self._clip_contribution_value(conversions)
        conversions["filter"] = ""
        conversions = conversions.drop(columns=["product_id"])
        return conversions
