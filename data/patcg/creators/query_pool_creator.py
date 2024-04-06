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


    def specialize_df(self, impressions_df: pd.DataFrame, conversions_df: pd.DataFrame) -> pd.DataFrame:
        impressions_df.rename(columns={'exp_attribute_6': 'ad_creative'}, inplace=True)
        conversions_df.rename(columns={'conv_amount': 'amount'}, inplace=True)
        impressions_df = impressions_df[["timestamp", "device_id", "ad_creative"]]
        conversions_df = conversions_df[["timestamp", "device_id", "amount"]]
        return impressions_df, conversions_df


    def create_impressions(self, impressions: pd.DataFrame) -> pd.DataFrame:
        impressions = impressions.sort_values(by=["timestamp"])
        impressions["filter"] = impressions["ad_creative"]
        impressions["key"] = ""
        return impressions


    def _create_record_per_query(self, conversions: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("creating a conversion record per query...")
        num_conversions = len(conversions)
        megabatches = num_conversions // self.config.scheduled_batch_size 
        remaining = num_conversions % self.config.scheduled_batch_size

        # Repeat conversions self.dimension_domain_size times each time setting the filter to a different value
        all_conversions = pd.concat([conversions] * self.dimension_domain_size, ignore_index=True)

        for i in range(self.dimension_domain_size):
            query_start = i * num_conversions
            query_end = (i+1) * num_conversions
            all_conversions.loc[query_start : query_end, "filter"] = i

            for j in range(megabatches):
                batch_start = query_start + (j * self.config.scheduled_batch_size)
                batch_end = query_start + ((j+1) * self.config.scheduled_batch_size)
                all_conversions.loc[batch_start : batch_end, "query_key"] = f"ad{i}_{j}"

            if remaining > 0:
                batch_start = query_start + (megabatches * self.config.scheduled_batch_size)
                batch_end = query_start + (megabatches * self.config.scheduled_batch_size) + remaining
                all_conversions.loc[batch_start : batch_end, "query_key"] = f"ad{i}_{megabatches}"
        return all_conversions


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
        
        # self.config.scheduled_batch_size = 1875000

        # Set epsilons
        [a, b] = self.config.accuracy
        cap_value = self.config.cap_value
        n = 20000

        def set_epsilon_given_accuracy(a, b, s, n):
            return s * math.log(1 / b) / (n * a)    # 0.069

        conversions["epsilon"] = set_epsilon_given_accuracy(a, b, cap_value, n)
        conversions['amount'] = conversions['amount'].clip(upper=cap_value)
        conversions["key"] = ""
        conversions = self._create_record_per_query(conversions)

        # conversions.groupby("query_key").size().reset_index(name="counts")
        # conversions.groupby(["query_key", "device_id"]).size().reset_index(name="counts")
        # TODO: limit user contribution

        # TODO: re enable filter bins in the code
        # TODO: Choose a uniform attribute instead?
        conversions = conversions.sort_values(by=["timestamp"])
        return conversions
