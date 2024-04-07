import math
from omegaconf import DictConfig
from data.patcg.creators.base_creator import BaseCreator, pd


class QueryPoolDatasetCreator(BaseCreator):

    def __init__(self, config: DictConfig) -> None:
        super().__init__(
            config,
            "patcg_impressions.csv",
            "patcg_conversions.csv",
        )
        self.dimension_domain_size = 10
        self.user_column_name = "device_id"

    def specialize_impressions(self, impressions: pd.DataFrame) -> pd.DataFrame:
        impressions.rename(columns={"exp_attribute_2": "ad_creative"}, inplace=True)
        impressions = impressions[["exp_timestamp", "device_id", "ad_creative"]]
        return impressions

    def create_impressions(self, impressions: pd.DataFrame) -> pd.DataFrame:
        impressions = impressions.sort_values(by=["exp_timestamp"])
        impressions["filter"] = impressions["ad_creative"].astype(str)
        impressions["key"] = ""
        return impressions
    
    def specialize_conversions(self, conversions: pd.DataFrame) -> pd.DataFrame:
        conversions = conversions[["conv_timestamp", "device_id", "conv_amount"]]
        conversions = conversions.sort_values(by=["conv_timestamp"])
        return conversions

    def _enforce_user_contribution_cap(self, conversions: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Enforcing cap on user contribution...")
        seen_users = set()
        row_count = 0

        def __mark_include(row: pd.Series):
            nonlocal seen_users
            nonlocal row_count
            if row_count == self.config.scheduled_batch_size:
                row_count = 0
                seen_users = set()

            user = row["device_id"]
            if user in seen_users:
                return False
            else:
                seen_users.add(user)
                row_count += 1
                return True

        conversions = conversions.assign(
            include=conversions.apply(
                lambda row: __mark_include(row),
                axis=1,
            ),
        )

        conversions = conversions.loc[conversions.include]
        conversions = conversions.drop(columns=["include"])
        conversions = conversions.reset_index()
        print(conversions.shape)
        return conversions

    def _tag_batch(self, conversions: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Tagging batches with batch id...")

        num_conversions = len(conversions)
        megabatches_per_query = num_conversions // self.config.scheduled_batch_size
        print("megabatches", megabatches_per_query)
        remaining = num_conversions % self.config.scheduled_batch_size
        print("minibatch size", remaining)

        for i in range(megabatches_per_query):
            batch_start = i * self.config.scheduled_batch_size
            batch_end = (i + 1) * self.config.scheduled_batch_size
            print("   ", batch_start, batch_end)

            conversions.loc[batch_start:batch_end, "batch_id"] = f"_{i}"

        if remaining > 0:
            batch_start = megabatches_per_query * self.config.scheduled_batch_size
            batch_end = (
                megabatches_per_query * self.config.scheduled_batch_size
            ) + remaining
            print("  rem   ", batch_start, batch_end)
            conversions.loc[batch_start:batch_end, "batch_id"] = (
                f"_{megabatches_per_query}"
            )
        
        conversions["key"] = ""
        return conversions

    def _clip_contribution_value(self, conversions: pd.DataFrame) -> pd.DataFrame:
        conversions["conv_amount"] = conversions["conv_amount"].clip(
            upper=self.config.cap_value
        )
        return conversions

    def _set_epsilon(self, conversions: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Setting epsilons...")

        def set_epsilon_given_accuracy(n):
            [a, b] = self.config.accuracy
            return self.config.cap_value * math.log(1 / b) / (n * a)  # 0.069

        batch_sizes = (
            conversions.groupby("batch_id").size().reset_index(name="batch_size")
        )
        batch_sizes = batch_sizes.assign(
            nonempty_batch_size=batch_sizes.apply(
                lambda row: (self.config.attribution_rate * row.batch_size) / 10, axis=1
            )
        )
        batch_sizes = batch_sizes.assign(
            epsilon=batch_sizes.apply(
                lambda row: set_epsilon_given_accuracy(row.nonempty_batch_size), axis=1
            )
        )
        conversions = batch_sizes.merge(conversions, how="inner", on="batch_id")
        conversions = conversions[
            ["conv_timestamp", "device_id", "batch_id", "epsilon", "conv_amount"]
        ]
        return conversions

    def _create_record_per_query(self, conversions: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("creating a conversion record per query...")

        num_conversions = len(conversions)
        # Repeat conversions once per query and tag the different query types with their query type id
        conversions = pd.concat(
            [conversions] * self.config.dimension_domain_size, ignore_index=True
        )

        for i in range(self.config.dimension_domain_size):
            query_start = i * num_conversions
            query_end = (i + 1) * num_conversions
            print(query_start, query_end)
            conversions.loc[query_start:query_end, "filter"] = str(i)
        return conversions

    def _get_unique_key(self, conversions: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Setting key...")
        conversions["batch_id"] = conversions['filter'].astype(str) + conversions['batch_id'].astype(str)
        conversions.rename(columns={"batch_id": "query_key"}, inplace=True)
        return conversions

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

        # We have a total of 75M conversions and an attribution rate of ~1%.
        # Around 750K conversions will be attributed. Assuming each attribution will count towards
        # only one out of the 10 queries, we expect around 75K attributed conversions per query assuming they are uniform.
        # So say we want a batch size of non-empty reports of around 20K we should schedule each query around 4 times
        # to have a total of 40 queries.
        # I should keep an actual batch size of 1.875M reports to approximately reach the desired batch size of 20k per query.

        conversions = self._enforce_user_contribution_cap(conversions)
        conversions = self._tag_batch(conversions)
        conversions = self._clip_contribution_value(conversions)
        conversions = self._set_epsilon(conversions)
        conversions = self._create_record_per_query(conversions)
        conversions = self._get_unique_key(conversions)
        conversions = conversions.sort_values(by=["conv_timestamp"])
        return conversions
