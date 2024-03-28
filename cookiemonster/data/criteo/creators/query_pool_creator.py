from dataclasses import dataclass
from datetime import datetime

from cookiemonster.data.criteo.creators.base_creator import BaseCreator, pd
from cookiemonster.data.criteo.creators.epsilon_calculator import get_epsilon_from_accuracy_for_counts

@dataclass
class ScalarQuery:
    partner_id: str
    dimension_name: str
    dimension_value: str 
    
class QueryPoolDatasetCreator(BaseCreator):

    MIN_CONVERSIONS_REQUIRED = 20_000
    PURCHASE_COUNT_CAP_VALUE = 5

    def __init__(self) -> None:
        super().__init__(
            "criteo_query_pool_impressions.csv", "criteo_query_pool_conversions.csv"
        )
        self.query_pools: dict[ScalarQuery, int] = {}
        self.dimensions = [
            'product_category1', 'product_category2', 'product_category3', 
            'product_category4', 'product_category5', 'product_category6', 'product_category7',
            'product_age_group', 'device_type', 'audience_id', 'product_gender', 'product_brand',
            'product_country',
        ]

    def _run_basic_specialization(self, df: pd.DataFrame) -> pd.DataFrame:
        # create some other columns from existing data for easier reading
        df = df.assign(
            click_datetime=df["click_timestamp"].apply(
                lambda x: datetime.fromtimestamp(x)
            )
        )
        df = df.assign(
            click_day=df["click_datetime"].apply(
                lambda x: (7 * (x.isocalendar().week - 1)) + x.isocalendar().weekday
            )
        )
        min_click_day = df["click_day"].min()
        df["click_day"] -= min_click_day

        df = df.assign(
            conversion_timestamp=df["Time_delay_for_conversion"] + df["click_timestamp"]
        )
        df = df.assign(
            conversion_datetime=df["conversion_timestamp"].apply(
                lambda x: datetime.fromtimestamp(x)
            )
        )
        df = df.assign(
            conversion_day=df["conversion_datetime"].apply(
                lambda x: (7 * (x.isocalendar().week - 1)) + x.isocalendar().weekday
            )
        )
        df["conversion_day"] -= min_click_day

        filter = "-"
        df["filter"] = filter
        return df
    
    def _augment_df_with_synthetic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        # Step 1. Add on the max number of added dimensions we are going to need

        max_N = 0
        for each advertiser:
            M = count of conversions for that advertiser
            N = M // QueryPoolDatasetCreator.MIN_CONVERSIONS_REQUIRED
            if max_N < N:
                max_N = N
        
        for i in range(1, max_N + 1):
            conversions[f"bucket{i}"] = -1

        # Step 2. Populate the appropriate dimensions with synthetic values
        
        for each advertiser:
            M = count of conversions
            N = M // QueryPoolDatasetCreator.MIN_CONVERSIONS_REQUIRED
            for i in range(N, 0, -1):
                generate i group values that we can assign out
                uniformly distribute the group values across all M conversions; if not conversion, set it to -1.
                replace the current placeholder column with the new dimension series in the df.
        """
        return df
    
    def _populate_query_pools(self, df: pd.DataFrame) -> None:
        """
        for each advertiser:
            M = count of conversions
            for each dimension in self.dimensions:
                dimension_counts = `select count(1), dimension from df where partner_id=advertiser group by dimension`
                for count, value in dimension_counts:
                    if count >= QueryPoolDatasetCreator.MIN_CONVERSIONS_REQUIRED:
                        scalar_query = ScalarQuery(
                            partner_id=advertiser,
                            dimension_name=dimension,
                            dimension_value=value,
                        )
                        self.query_counts[scalar_query] = count
        """

    def specialize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=['partner_id', 'user_id', "product_id"])
        df = self._run_basic_specialization(df)
        df = self._augment_df_with_synthetic_features(df)
        self._populate_query_pools(df)
        return df
    
    def create_impressions(self, df: pd.DataFrame) -> pd.DataFrame:
        impressions = df[
            ["click_timestamp", "click_day", "user_id", "partner_id", "filter"]
        ]
        impressions = impressions.sort_values(by=["click_timestamp"])
        impressions["key"] = "-"
        return impressions
    
    def _create_record_per_query(self, conversions: pd.DataFrame) -> pd.DataFrame:
        """
        new_conversions = pd.DataFrame()

        for each conversion in conversions:
            heap = heapq()
            for each dimension in self.dimensions:
                scalar_query = scalar_query = ScalarQuery(
                    partner_id=conversion.partner_id,
                    dimension_name=dimension,
                    dimension_value=conversion[dimension],
                )
                count = self.query_counts(scalar_query)
                if count:
                    heap.push((count, scalar_query))
            
            while heap:
                (count, scalar_query) = heap
                new_conversion = copy(conversion)
                new_conversion[query_dimension] = scalar_query.dimension
                new_conversion["query_count"] = count
                new_conversions.add(new_conversion)
        
        return new_conversions
        """

    def create_conversions(self, df: pd.DataFrame) -> pd.DataFrame:
        conversions = df.loc[df.Sale == 1]
        conversions = self._create_record_per_query(df.loc[df.Sale == 1])
        
        # Compute counts
        conversions = conversions.assign(
            count=(
                conversions["SalesAmountInEuro"] // conversions["product_price"]
            ).apply(lambda c: min(c, QueryPoolDatasetCreator.PURCHASE_COUNT_CAP_VALUE))
        )
        conversions = conversions.drop(columns=["product_price", "SalesAmountInEuro"])

        # Get epsilons from accuracy
        conversions["epsilon"] = conversions["query_count"].apply(
            lambda c: get_epsilon_from_accuracy_for_counts(c, QueryPoolDatasetCreator.PURCHASE_COUNT_CAP_VALUE)
        )
        conversions = conversions.drop(columns=["query_count"])

        conversions["aggregatable_cap_value"] = QueryPoolDatasetCreator.PURCHASE_COUNT_CAP_VALUE
        conversions["key"] = "purchaseCount"

        return conversions
