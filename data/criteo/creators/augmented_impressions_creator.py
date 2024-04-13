from omegaconf import DictConfig
from data.criteo.creators.query_pool_creator import QueryPoolDatasetCreator, pd


class AugmentedImpressionsDatasetCreator(QueryPoolDatasetCreator):

    def __init__(self, config: DictConfig) -> None:
        super().__init__(
            config,
            "criteo_augmented_query_pool_impressions.csv",
            "criteo_augmented_query_pool_conversions.csv",
        )

    def create_impressions(self, df: pd.DataFrame) -> pd.DataFrame:
        impressions = df[self.impression_columns_to_use]
        impressions = impressions.sort_values(by=["click_timestamp"])
        impressions["key"] = ""

        """
        TODO: Randomly select u% of users who have attributed conversions.
        For each user, take the impressions attributed to their conversions
        and create x% more impressions, each with a y click_timestamp,
        where x is a config variable we pass in, and y is randomly sampled from
        [conversion_timestamp - attribution_window, conversion_timestamp].
        """

        return impressions
