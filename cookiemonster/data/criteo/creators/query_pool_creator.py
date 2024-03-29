from datetime import datetime
from uuid import uuid4

from cookiemonster.data.criteo.creators.base_creator import BaseCreator, pd
from cookiemonster.data.criteo.creators.epsilon_calculator import get_epsilon_from_accuracy_for_counts

QueryKey = tuple[str, str, str] # (partner_id, dimension_value, dimension_name)

class QueryPoolDatasetCreator(BaseCreator):

    MIN_CONVERSIONS_REQUIRED = 20_000
    PURCHASE_COUNT_CAP_VALUE = 5

    def __init__(self) -> None:
        super().__init__(
            "criteo_query_pool_impressions.csv", "criteo_query_pool_conversions_uncorrupted.csv"
        )
        self.query_pool: dict[QueryKey, int] = {}
        self.dimension_names = [
            'product_category1', 'product_category2', 'product_category3', 
            'product_category4', 'product_category5', 'product_category6', 'product_category7',
            'product_age_group', 'device_type', 'audience_id', 'product_gender', 'product_brand',
            'product_country',
        ]


    def _run_basic_specialization(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("running basic df specialization...")
        # create some other columns from existing data for easier reading
        df = df.assign(
            click_datetime=df["click_timestamp"].apply(
                lambda x: datetime.fromtimestamp(x)
            ),
            conversion_timestamp=df["Time_delay_for_conversion"] + df["click_timestamp"]
        )

        df = df.assign(
            click_day=df["click_datetime"].apply(
                lambda x: (7 * (x.isocalendar().week - 1)) + x.isocalendar().weekday
            ),
            conversion_datetime=df["conversion_timestamp"].apply(
                lambda x: datetime.fromtimestamp(x)
            )
        )

        min_click_day = df["click_day"].min()
        df["click_day"] -= min_click_day

        df = df.assign(
            conversion_day=df["conversion_datetime"].apply(
                lambda x: (7 * (x.isocalendar().week - 1)) + x.isocalendar().weekday
            )
        )
        df["conversion_day"] -= min_click_day
        df["filter"] = "-"
        return df


    def _augment_df_with_synthetic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("adding synthetic categorical features to the dataset...")

        N = (df.loc[(df.Sale == 1)]
             .groupby(['partner_id']).Sale.count()
             .map(lambda m: m // QueryPoolDatasetCreator.MIN_CONVERSIONS_REQUIRED)
             .max()
        )
        self.logger.info(f"will add {N} synthetic dimensions to the dataset")

        
        impressions = df.loc[(df.Sale != 1)]
        conversions = df.loc[(df.Sale == 1)]
        M = conversions.shape[0]
        synthetics = pd.DataFrame()
        for i in range(1, N + 1):
            curr = f"synthetic_category{i}"
            self.dimension_names.append(curr)
            impressions = impressions.assign(**{curr: -1})

            size = M // i
            remainder = M % i
            chunks = []
            types = []
            for _ in range(1, i + 1):
                type = str(uuid4()).upper().replace('-', '')
                types.append(type)
                chunk = [type]*size
                chunks += chunk

            type_length = len(types)
            for i in range(remainder):
                chunks.append(types[i % type_length])

            synthetics[curr] = chunks

        
        # randomly shuffle the records so we can round robin the synthetic values
        conversions = conversions.sample(frac=1).reset_index(drop=True)
        conversions = pd.concat([conversions, synthetics], axis=1)

        # restore the original order of the df
        enhanced_df = pd.concat([impressions, conversions]).sort_values(by=['click_timestamp'])
        return enhanced_df
    

    def _populate_query_pools(self, df: pd.DataFrame) -> None:
        self.logger.info("populating the query pools...")
        conversions = df.loc[(df.Sale == 1)]
        for dimension_name in self.dimension_names:
            conversions = conversions.assign(dimension_name=dimension_name)
            counts = conversions.groupby(['partner_id', dimension_name, 'dimension_name']).Sale.count()
            counts = counts[counts >= QueryPoolDatasetCreator.MIN_CONVERSIONS_REQUIRED]
            if not counts.empty:
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
        self.logger.info("creating a conversion record per query...")
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
        
        conversions = conversions.assign(
            count=(
                conversions["SalesAmountInEuro"] // conversions["product_price"]
            ).apply(lambda c: min(c, QueryPoolDatasetCreator.PURCHASE_COUNT_CAP_VALUE)),
            epsilon=conversions["query_count"].apply(
                lambda c: get_epsilon_from_accuracy_for_counts(c, QueryPoolDatasetCreator.PURCHASE_COUNT_CAP_VALUE)
            ),
            key="purchaseCount",
            aggregatable_cap_value=QueryPoolDatasetCreator.PURCHASE_COUNT_CAP_VALUE,
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

        return conversions
