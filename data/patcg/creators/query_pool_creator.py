import os
import math
from omegaconf import DictConfig
from data.patcg.creators.base_creator import BaseCreator, pd


class QueryPoolDatasetCreator(BaseCreator):

    def __init__(self, config: DictConfig) -> None:
        super().__init__(
            config,
            "patcg_impressions.csv",
            "patcg_conversions",
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
        conversions = conversions.reset_index(drop=True)
        print(conversions.shape)
        return conversions

    def _clip_contribution_value(self, conversions: pd.DataFrame) -> pd.DataFrame:
        conversions["conv_amount"] = conversions["conv_amount"].clip(
            upper=self.config.cap_value
        )
        return conversions

    def _set_epsilon(self, batch_size) -> pd.DataFrame:
        # self.logger.info("Setting epsilons...")

        def set_epsilon_given_accuracy(n):
            [a, b] = self.config.accuracy
            return self.config.cap_value * math.log(1 / b) / (n * a)  # 0.069
        
        nonempty_batch_size = self.config.attribution_rate * batch_size / 10
        return set_epsilon_given_accuracy(nonempty_batch_size)
        

    def _write_batch(
        self, batch_size, nbatches, remaining, query_key, total_batches, conversions: pd.DataFrame
    ) -> pd.DataFrame:
        self.logger.info(f"Create batches for query {query_key}...")

        conversions["filter"] = query_key

        for i in range(nbatches):
            bs = i * batch_size
            be = (i + 1) * batch_size - 1

            print("     Query:", query_key, "Batch:", bs, be)

            unique_query_key = query_key * total_batches + i
            query_path = os.path.join(
                self.conversions_path, f"{unique_query_key}.csv"
            )
            # conversions.loc[bs:be, "query_key"] = unique_query_key
            conversions.loc[bs:be, "epsilon"] = self._set_epsilon(batch_size)
            conversions.iloc[bs:be].to_csv(query_path, header=True, index=False)

        if remaining > 0:
            bs = nbatches * batch_size
            be = nbatches * batch_size + remaining - 1
            print("     Query:", query_key, "Minibatch", bs, be)

            unique_query_key = query_key * total_batches + nbatches
            query_path = os.path.join(
                self.conversions_path, f"{unique_query_key}.csv"
            )
            # conversions.loc[bs:be, "query_key"] = f"{query_key}_{nbatches}"
            conversions.loc[bs:be, "epsilon"] = self._set_epsilon(remaining)
            conversions.iloc[bs:be].to_csv(query_path, header=True, index=False)


    def _write_queries(self, conversions: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"Creating queries...")

        num_conversions = len(conversions)
        batch_size = self.config.scheduled_batch_size
        nbatches = num_conversions // batch_size
        remaining = num_conversions % batch_size
        total_batches =  nbatches + 1 if remaining > 0 else nbatches

        print("megabatches", nbatches)
        print("minibatch size", remaining)

        for i in range(self.config.dimension_domain_size):
            self._write_batch(batch_size, nbatches, remaining,i,  total_batches, conversions)

            
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

                   
        if not os.path.exists(self.conversions_path):
            os.makedirs(self.conversions_path)
                   
        conversions = self._enforce_user_contribution_cap(conversions)
        conversions = self._clip_contribution_value(conversions)
        conversions = self._write_queries(conversions)
        return conversions
