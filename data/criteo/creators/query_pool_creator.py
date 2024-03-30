from datetime import datetime
from uuid import uuid4

from omegaconf import DictConfig

from data.criteo.creators.base_creator import BaseCreator, pd
from data.criteo.creators.epsilon_calculator import get_epsilon_from_accuracy_for_counts

QueryKey = tuple[str, str, str] # (advertiser_value, dimension_value, dimension_name)

class QueryPoolDatasetCreator(BaseCreator):

    def __init__(self, config: DictConfig) -> None:
        super().__init__(
            config,
            "criteo_query_pool_impressions.csv",
            "criteo_query_pool_conversions.csv",
        )
        self.query_pool: dict[QueryKey, int] = {} # query -> number of conversions
        self.dimension_names = [
            "product_category1",
            "product_category2",
            "product_category3",
            "product_category4",
            "product_category5",
            "product_category6",
            "product_category7",
            "product_age_group",
            "device_type",
            "audience_id",
            "product_gender",
            "product_brand",
            "product_country",
        ]
        self.advertiser_column_name = 'partner_id'
        self.product_column_name = 'product_id'
        self.user_column_name = 'user_id'

        self.conversion_columns_to_drop = [
            "SalesAmountInEuro", "product_price",
            "nb_clicks_1week", "Time_delay_for_conversion",
            "Sale", "click_timestamp", "click_day", "click_datetime",
        ]
        self.impression_columns_to_use = [
            "click_timestamp", "click_day", "user_id", "partner_id", "filter"
        ]
        self.min_conversions_required_for_dp = config.min_conversions_required_for_dp


    def _run_basic_specialization(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("running basic df specialization...")
        # create some other columns from existing data for easier reading
        df = df.assign(
            click_datetime=df["click_timestamp"].apply(
                lambda x: datetime.fromtimestamp(x)
            ),
            conversion_timestamp=df["Time_delay_for_conversion"]
            + df["click_timestamp"],
        )

        df = df.assign(
            click_day=df["click_datetime"].apply(
                lambda x: (7 * (x.isocalendar().week - 1)) + x.isocalendar().weekday
            ),
            conversion_datetime=df["conversion_timestamp"].apply(
                lambda x: datetime.fromtimestamp(x)
            ),
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
    
    def _augment_df_with_synthetic_product_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("determining synthetic categorical features to add to the dataset...")
        
        synthetics_map = {} # (advert, product_id, buckets) -> value
        max_min_upper_bound = 0
        for advertiser in df[self.advertiser_column_name].unique():
            advertiser_chunk = df.loc[df[self.advertiser_column_name] == advertiser]
            conversion_count = advertiser_chunk.loc[advertiser_chunk.Sale == 1].shape[0]
            unique_products = pd.Series(advertiser_chunk[self.product_column_name].unique())
            unique_products = unique_products.sample(frac=1)

            upper_bound_unique_products = unique_products.shape[0] - 1
            upper_bound_num_buckets = conversion_count // self.min_conversions_required_for_dp
            min_upper_bound = min(upper_bound_unique_products, upper_bound_num_buckets)

            if min_upper_bound > max_min_upper_bound:
                max_min_upper_bound = min_upper_bound

            for buckets in range(2, min_upper_bound + 1):
                values = []
                for _ in range(buckets):
                    synthetic_value = str(uuid4()).upper().replace('-', '')
                    values.append(synthetic_value)
                
                i = 0
                for product in unique_products:
                    synthetics_map[(advertiser, product, buckets)] = values[i % buckets]
                    i += 1

        def lookup_synthetic_value(row: pd.Series, bucket: int):
            key = (row[self.advertiser_column_name], row[self.product_column_name], bucket)
            return synthetics_map.get(key, pd.NA)

        self.logger.info(f"adding {max_min_upper_bound-1} synthetic dimensions to the dataset...")

        to_add = {}
        for curr_bucket in range(2, max_min_upper_bound + 1):
            to_add[f"synthetic_category{curr_bucket}"] = df.apply(
                lambda row: lookup_synthetic_value(row, curr_bucket),
                axis=1
            )

        return df.assign(**to_add)


    def _populate_query_pools(self, df: pd.DataFrame) -> None:
        self.logger.info("populating the query pools...")
        conversions = df.loc[(df.Sale == 1)]
        for dimension_name in self.dimension_names:
            conversions = conversions.assign(dimension_name=dimension_name)
            counts = conversions.groupby([self.advertiser_column_name, dimension_name, 'dimension_name']).Sale.count()
            counts = counts[counts >= self.min_conversions_required_for_dp]
            if not counts.empty:
                self.query_pool.update(counts.to_dict())

        keys = [x for x in self.query_pool.keys()]
        keys.sort()
        log_lines = []
        for key in keys:
            count = self.query_pool[key]
            (partner_id, dimension, dimension_name) = key
            log_lines.append(f"{count} total conversion records from partner_id ({partner_id}), {dimension_name} ({dimension})")

        query_pool_contents = str.join('\n\t', log_lines)
        self.logger.info(f"Generated the following query pool:\n\t{query_pool_contents}\n")


    def specialize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=[self.advertiser_column_name, self.user_column_name, self.product_column_name])
        
        # TODO: [PM] should we include the outlier when really running this?
        df = df[df.partner_id != 'E3DDEB04F8AFF944B11943BB57D2F620']

        self._augment_df_with_synthetic_product_features(df)
        
        self._populate_query_pools(df)
        df = self._run_basic_specialization(df)
        return df

    def create_impressions(self, df: pd.DataFrame) -> pd.DataFrame:
        impressions = df[self.impression_columns_to_use]
        impressions = impressions.sort_values(by=["click_timestamp"])
        impressions["key"] = "-"
        return impressions
    

    def _get_used_dimension_names(self) -> set:
        return set(map(lambda x: x[2], self.query_pool.keys()))
    

    def _create_record_per_query(self, conversions: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("creating a conversion record per query...")
        conversion_chunks = []
        for dimension in self._get_used_dimension_names():
            conversions = conversions.assign(
                query_key=conversions.apply(
                    lambda conversion: (conversion[self.advertiser_column_name], conversion[dimension], dimension),
                    axis=1
                )
            )
            conversions = conversions.assign(
                included=conversions.query_key.isin(self.query_pool.keys())
            )
            conversions_to_use = conversions.loc[conversions.included]


            conversions_to_use = conversions_to_use.assign(
                conversion_count=conversions_to_use.apply(
                    lambda conversion: self.query_pool[(conversion[self.advertiser_column_name], conversion[dimension], dimension)],
                    axis=1
                )
            )
            
            conversion_chunks.append(conversions_to_use)

        return pd.concat(conversion_chunks)
    
    @staticmethod
    def _compute_product_count(conversion):
        sell_price = conversion["SalesAmountInEuro"]
        offer_price = conversion["product_price"]
        if sell_price and offer_price:
            return sell_price // offer_price
        else:
            return 1

    def create_conversions(self, df: pd.DataFrame) -> pd.DataFrame:
        conversions = df.loc[df.Sale == 1]
        purchase_counts = conversions.apply(QueryPoolDatasetCreator._compute_product_count, axis=1)

        """
        TODO: [PM] what should we cap our purchase counts at?
        Aggregatable reports description:
        count    1.279493e+06
        mean     4.705447e+00
        std      1.581949e+02
        min      0.000000e+00
        25%      1.000000e+00
        50%      1.000000e+00
        75%      2.000000e+00
        max      8.661200e+04
        skew     352.22094940782813

        so, maybe 5 is reasonable? should we calculate this a different way generally?
        """
        max_purchase_counts = 5

        conversions = conversions.assign(count=purchase_counts)

        self.log_descriptions_of_reports(conversions)
        
        conversions = self._create_record_per_query(conversions)
        
        conversions = conversions.assign(
            epsilon=conversions["conversion_count"].apply(
                lambda conversion_count: get_epsilon_from_accuracy_for_counts(conversion_count, max_purchase_counts)
            ),
            key=conversions.apply(
                lambda conversion: f"{str.join('|', conversion.query_key)}|purchaseCount",
                axis=1
            ),
            aggregatable_cap_value=max_purchase_counts,
        )

        query_epsilons = str.join('', conversions.apply(
            lambda conversion: f"\t{conversion['query_key']}, epsilon: {conversion['epsilon']}\n",
            axis=1
        ).unique())
        self.logger.info(f"Query pool epsilons:\n{query_epsilons}")

        unused_dimension_names = set(self.dimension_names) - self._get_used_dimension_names()
        columns_we_created = ["query_key", "conversion_count"]

        to_drop = [
            *unused_dimension_names,
            *columns_we_created,
            *self.conversion_columns_to_drop,
        ]

        return conversions.drop(columns=to_drop)

    def log_descriptions_of_reports(self, conversions):
        counts = conversions["count"]
        self.logger.info(f"Aggregatable reports description:\n{counts.describe()}")
        self.logger.info(f"Aggregatable reports skew: {counts.skew()}")

        true_sums = []
        for dimension in self.dimension_names:
            true_sums_for_dimension = conversions.groupby([self.advertiser_column_name, dimension])['count'].sum()
            true_sums.append(true_sums_for_dimension)
        
        all_true_summary_reports = pd.concat(true_sums).reset_index(drop=True)
        self.logger.info(f"True summary reports description:\n{all_true_summary_reports.describe()}")
        self.logger.info(f"True summary reports skew: {all_true_summary_reports.skew()}")
