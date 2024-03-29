from datetime import datetime

from cookiemonster.data.criteo.creators.base_creator import BaseCreator, pd
from cookiemonster.data.criteo.creators.epsilon_calculator import get_epsilon_from_accuracy_for_counts

QueryKey = tuple[str, str, str] # (partner_id, dimension_value, dimension_name)

class QueryPoolDatasetCreator(BaseCreator):

    MIN_CONVERSIONS_REQUIRED = 20_000
    PURCHASE_COUNT_CAP_VALUE = 5

    def __init__(self) -> None:
        super().__init__(
            "criteo_query_pool_impressions.csv", "criteo_query_pool_conversions.csv"
        )
        self.query_pool: dict[QueryKey, int] = {}
        self.dimension_names = [
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

        df["filter"] = "-"
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
            curr = f"bucket{i}"
            conversions[curr] = -1
            self.dimension_names.append(curr)

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
        conversions = df.loc[(df.Sale == 1)]
        for dimension_name in self.dimension_names:
            conversions = conversions.assign(dimension_name=dimension_name)
            counts = conversions.groupby(['partner_id', dimension_name, 'dimension_name']).Sale.count()
            counts = counts[counts >= QueryPoolDatasetCreator.MIN_CONVERSIONS_REQUIRED]
            self.query_pool.update(counts.to_dict())

        self.logger.info(f"Generated the following query pool:")
        keys = [x for x in self.query_pool.keys()]
        keys.sort()
        for key in keys:
            count = self.query_pool[key]
            (partner_id, dimension, dimension_name) = key
            print(f"{count} many products purchased for partner_id ({partner_id}), {dimension_name} ({dimension})")


    def specialize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=['partner_id', 'user_id', "product_id"])
        df = self._augment_df_with_synthetic_features(df)
        self._populate_query_pools(df)
        df = self._run_basic_specialization(df)
        return df
    

    def create_impressions(self, df: pd.DataFrame) -> pd.DataFrame:
        impressions = df[
            ["click_timestamp", "click_day", "user_id", "partner_id", "filter"]
        ]
        impressions = impressions.sort_values(by=["click_timestamp"])
        impressions["key"] = "-"
        return impressions
    

    def _create_record_per_query(self, conversions: pd.DataFrame) -> pd.DataFrame:
        new_conversions = pd.DataFrame()
        used_dimension_names = set(map(lambda x: x[2], self.query_pool.keys()))
        for dimension in used_dimension_names:
            conversions = conversions.assign(
                query_key=conversions.apply(
                    lambda conversion: (conversion['partner_id'], conversion[dimension], dimension),
                    axis=1
                )
            )
            conversions = conversions.assign(
                included=conversions.query_key.isin(self.query_pool.keys())
            )
            conversions_to_use = conversions.loc[conversions.included]
            
            conversions_to_use = conversions_to_use.assign(
                query_count=conversions_to_use.apply(
                    lambda conversion: self.query_pool[(conversion['partner_id'], conversion[dimension], dimension)],
                    axis=1
                )
            )
            new_conversions = pd.concat([new_conversions, conversions_to_use])

        new_conversions = new_conversions.drop(columns=['included'])
        return new_conversions

    def create_conversions(self, df: pd.DataFrame) -> pd.DataFrame:
        conversions = self._create_record_per_query(df.loc[df.Sale == 1])
        
        # Compute counts
        conversions = conversions.assign(
            count=(
                conversions["SalesAmountInEuro"] // conversions["product_price"]
            ).apply(lambda c: min(c, QueryPoolDatasetCreator.PURCHASE_COUNT_CAP_VALUE))
        )

        # Get epsilons from accuracy
        conversions = conversions.assign(
            epsilon=conversions["query_count"].apply(
                lambda c: get_epsilon_from_accuracy_for_counts(c, QueryPoolDatasetCreator.PURCHASE_COUNT_CAP_VALUE)
            )
        )

        conversions = conversions.drop(columns=[
            "query_key",
            "query_count",
            "Time_delay_for_conversion",
            "nb_clicks_1week",
            "product_title",
            "product_price",
            "SalesAmountInEuro",
        ])

        conversions["aggregatable_cap_value"] = QueryPoolDatasetCreator.PURCHASE_COUNT_CAP_VALUE
        conversions["key"] = "purchaseCount"

        return conversions
