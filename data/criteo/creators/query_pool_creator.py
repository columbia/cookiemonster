import math
import numpy as np
from omegaconf import DictConfig
from uuid import uuid4


from data.criteo.creators.base_creator import BaseCreator, pd
from cookiemonster.epsilon_calculator import (
    get_epsilon_for_high_probability_relative_error_wrt_avg_prior,
)


class QueryPoolDatasetCreator(BaseCreator):

    def __init__(
        self,
        config: DictConfig,
    ) -> None:
        super().__init__(
            config,
            impressions_filename="criteo_query_pool_impressions.csv",
            conversions_filename="criteo_query_pool_conversions.csv",
            augmented_impressions_filename="criteo_query_pool_augmented_impressions.csv",
            augmented_conversions_filename="criteo_query_pool_augmented_conversions.csv",
        )
        self.used_dimension_names = set()

        self.dimension_names = [
            "product_category3",
        ]

        self.enforce_one_user_contribution_per_query = (
            config.enforce_one_user_contribution_per_query
        )
        self.max_batch_size = config.max_batch_size
        self.min_batch_size = config.min_batch_size
        self.advertiser_filter = config.get("advertiser_filter", [])
        self.advertiser_exclusions = config.get("advertiser_exclusions", [])

        # See notebooks/criteo_dataset_analysis.ipynb
        # 4) for an explanation for these numbers
        self.max_purchase_counts = 5
        self.expected_average_purchase_counts = 2

    def specialize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(
            subset=[
                self.advertiser_column_name,
                self.user_column_name,
                self.product_column_name,
            ]
        )
        if self.advertiser_filter:
            df = df[df[self.advertiser_column_name].isin(self.advertiser_filter)]

        if self.advertiser_exclusions:
            df = df[~df[self.advertiser_column_name].isin(self.advertiser_exclusions)]

        df = df.assign(filter="")
        return df

    def create_impressions(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.assign(key="")
        return df

    def _create_queries(
        self,
        conversions: pd.DataFrame,
    ) -> pd.DataFrame:

        seen_users = set()
        row_count = 0

        def __mark_include(row: pd.Series):
            nonlocal seen_users
            nonlocal row_count
            if row_count == self.max_batch_size:
                row_count = 0
                seen_users = set()

            user = row[self.user_column_name]
            if user in seen_users:
                return False
            else:
                seen_users.add(user)
                row_count += 1
                return True

        advertisers = conversions[self.advertiser_column_name].unique()
        query_batches = {}
        for advertiser in advertisers:
            ad_conversions = conversions.loc[
                conversions[self.advertiser_column_name] == advertiser
            ]
            for dimension_name in self.dimension_names:
                dimension_values = ad_conversions[dimension_name].dropna().unique()
                for dimension_value in dimension_values:
                    query_result = ad_conversions.loc[
                        ad_conversions[dimension_name] == dimension_value
                    ]
                    query_result = query_result.sort_values(by=["conversion_timestamp"])

                    # we have our total query. now we need to break it up into batches

                    # no point in continuing if the total query isn't even a mini batch.
                    if query_result.shape[0] < self.min_batch_size:
                        continue

                    if self.enforce_one_user_contribution_per_query:
                        # need to iterate row by row taking unique users up until
                        # max conversion count for dp.
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

                    query_result["query_key"] = [
                        (advertiser, dimension_value, dimension_name)
                    ] * query_result.shape[0]

                    # now split the query_result into its batches
                    query_result = query_result.reset_index(drop=True)
                    query_result_length = query_result.shape[0]
                    num_big_reports = query_result_length // self.max_batch_size
                    i = 0
                    while i < num_big_reports:
                        start = i * self.max_batch_size
                        end = (i + 1) * self.max_batch_size
                        batch = query_result.iloc[start:end]
                        assert batch.shape[0] >= self.min_batch_size
                        assert batch.shape[0] <= self.max_batch_size
                        if advertiser not in query_batches:
                            query_batches[advertiser] = []
                        query_batches[advertiser].append(batch)
                        self.used_dimension_names.add(dimension_name)
                        i += 1

                    i = i * self.max_batch_size
                    if (
                        i < query_result_length
                        and query_result_length - i >= self.min_batch_size
                    ):
                        batch = query_result.iloc[i:]
                        assert batch.shape[0] >= self.min_batch_size
                        assert batch.shape[0] <= self.max_batch_size
                        if advertiser not in query_batches:
                            query_batches[advertiser] = []
                        query_batches[advertiser].append(batch)
                        self.used_dimension_names.add(dimension_name)

        final_batches = []
        for batches in query_batches.values():
            for i, batch in enumerate(batches):
                final_batch = batch.assign(
                    epsilon=get_epsilon_for_high_probability_relative_error_wrt_avg_prior(
                        sensitivity=self.max_purchase_counts,
                        batch_size=batch.shape[0],
                        expected_average_result=self.expected_average_purchase_counts,
                        relative_error=0.05,
                        failure_probability=0.01,
                    ),
                    aggregatable_cap_value=self.max_purchase_counts,
                    key=i,
                )
                final_batches.append(final_batch)

        return pd.concat(final_batches)

    def create_conversions(self, df: pd.DataFrame) -> pd.DataFrame:

        def __compute_product_count(conversion, cap: int) -> int:
            sell_price = conversion["SalesAmountInEuro"]
            offer_price = conversion["product_price"]
            if sell_price and offer_price:
                return min(cap, max(1, sell_price // offer_price))
            else:
                return 1

        conversions = df.loc[df.Sale == 1]

        conversions = conversions.assign(
            count=conversions.apply(
                lambda conversion: __compute_product_count(
                    conversion, self.max_purchase_counts
                ),
                axis=1,
            ),
            conversion_timestamp=conversions.apply(
                lambda conversion: max(0, conversion["Time_delay_for_conversion"])
                + conversion["click_timestamp"],
                axis=1,
            ),
        )
        conversions = self._create_queries(conversions)
        conversions = conversions.sort_values(by=["conversion_timestamp"])

        self.log_query_epsilons(conversions)

        unused_dimension_names = set(self.dimension_names) - self.used_dimension_names
        columns_we_created = ["query_key"]

        to_drop = [
            *unused_dimension_names,
            *columns_we_created,
        ]

        return conversions.drop(columns=to_drop)

    def log_query_epsilons(self, conversions):
        queries = (
            conversions[["key", "query_key", "epsilon"]]
            .apply(
                lambda conversion: (
                    conversion["key"],
                    *conversion["query_key"],
                    conversion["epsilon"],
                ),
                axis=1,
            )
            .unique()
        )

        queries = sorted(queries, key=lambda q: (q[1], q[0]))

        query_epsilons = []
        for query in queries:
            msg = f"\tquery {query[0]}: {self.advertiser_column_name} '{query[1]}' {query[3]} '{query[2]}', epsilon: {query[4]}\n"
            query_epsilons.append(msg)
        self.logger.info(f"Query pool epsilons:\n{''.join(query_epsilons)}")

        query_tuples = pd.DataFrame(
            [[*x] for x in queries],
            columns=[
                "key",
                "advertiser",
                "dimension_value",
                "dimension_name",
                "epsilon",
            ],
        )

        advertiser_grouping = query_tuples.groupby(["advertiser"])
        advertiser_query_count = pd.DataFrame(
            advertiser_grouping.size().items(), columns=["advertiser", "query_count"]
        ).sort_values(by=["query_count"], ascending=False)
        advertiser_epsilon_sum = pd.DataFrame(
            advertiser_grouping.epsilon.sum().items(),
            columns=["advertiser", "epsilon_sum"],
        ).sort_values(by=["epsilon_sum"], ascending=False)

        pd.set_option("display.max_rows", None)
        self.logger.info(f"Query count per advertiser:\n{advertiser_query_count}")
        self.logger.info(f"Sum of epsilons per advertiser:\n{advertiser_epsilon_sum}")
        pd.reset_option("display.max_rows")

    def augment_impressions(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.loc[df.Sale == 1]
        df = df.assign(
            conversion_timestamp=df.apply(
                lambda conversion: max(0, conversion["Time_delay_for_conversion"])
                + conversion["click_timestamp"],
                axis=1,
            ),
        )

        augment_rates = self.config.get("augment_rates")
        if not augment_rates:
            msg = "received request to augment dataset, but no augment rates. will not augment impressions"
            self.logger.warning(msg)
            return pd.DataFrame()

        impressions_augment_rate = augment_rates.get("impressions")
        if not impressions_augment_rate:
            msg = "received request to augment impressions, but no augment rate was specified. will not augment impressions"
            self.logger.warning(msg)
            return pd.DataFrame()

        attribution_window = 30  # days
        attribution_window_seconds = attribution_window * 60 * 60 * 24
        impressions_to_add = math.ceil(impressions_augment_rate * attribution_window)

        def get_click_timestamps(attribution_window_end: int) -> int:
            nonlocal impressions_to_add
            nonlocal attribution_window_seconds
            attribution_window_start = (
                attribution_window_end - attribution_window_seconds
            )
            return [
                np.random.randint(attribution_window_start, attribution_window_end)
                for _ in range(impressions_to_add)
            ]

        df = df.assign(
            click_timestamps=df["conversion_timestamp"].apply(
                lambda ct: get_click_timestamps(ct)
            )
        )
        df = df.explode("click_timestamps")
        df.drop(columns=["click_timestamp"], inplace=True)
        df.rename(columns={"click_timestamps": "click_timestamp"}, inplace=True)
        df = df.assign(key="")

        return df

    def _get_attribute_domains(self, df: pd.DataFrame) -> dict[str, list]:
        attribute_domains = {}
        for dimension_name in self.dimension_names:
            attribute_domains[dimension_name] = list(
                df[dimension_name].dropna().unique()
            )
        return attribute_domains

    def augment_conversions(self, df: pd.DataFrame) -> pd.DataFrame:
        augment_rates = self.config.get("augment_rates")
        if not augment_rates:
            msg = "received request to augment dataset, but no augment rates. will not augment conversions"
            self.logger.warning(msg)
            return pd.DataFrame()

        conversions_augment_rates = augment_rates.get("conversions", {})
        new_users_augment_rate = conversions_augment_rates.get("new_users")
        existing_users_augment_rate = conversions_augment_rates.get("existing_users")

        df = df.loc[df.Sale == 1]

        if new_users_augment_rate:
            # To see the impact on RMSRE
            return self._augment_conversions_with_new_users(new_users_augment_rate, df)
        elif existing_users_augment_rate:
            # To provide different points of contention
            return self._augment_conversions_with_existing_users(existing_users_augment_rate, df)
        else:
            msg = "received request to augment conversions, but no augment rates were specified. will not augment conversions"
            self.logger.warning(msg)
            return pd.DataFrame()

    def _augment_conversions_with_new_users(self, augment_rate: float, df: pd.DataFrame) -> pd.DataFrame:
        attribution_window = 30  # days
        attribution_window_seconds = attribution_window * 60 * 60 * 24

        # for each advertiser, add (conversions_augment_rate*100)% more conversions to bring their
        # attribution rate down. these conversions will be unattributed and scattered throughout the dataset
        chunks = []
        for destination, dest_df in df.groupby([self.advertiser_column_name]):
            attribute_domains = self._get_attribute_domains(dest_df)
            num_conversions = dest_df.shape[0]
            num_conversions_to_add = math.ceil(
                num_conversions * augment_rate
            )

            min_timestamp = dest_df.click_timestamp.min()
            max_timestamp = dest_df.click_timestamp.max() + attribution_window_seconds

            records = []
            for _ in range(num_conversions_to_add):
                user_id = str(uuid4()).replace("-", "").upper()
                conversion_timestamp = np.random.randint(
                    min_timestamp, max_timestamp + 1
                )
                count = np.random.randint(1, self.max_purchase_counts + 1)
                product_price = 1
                sales_amount_in_euro = count * product_price
                record = {
                    self.advertiser_column_name: destination[0],
                    self.user_column_name: user_id,
                    "Sale": 1,
                    "click_timestamp": 0,
                    "Time_delay_for_conversion": conversion_timestamp,
                    "SalesAmountInEuro": sales_amount_in_euro,
                    "product_price": product_price,
                    "nb_clicks_1week": np.nan,
                    "filter": "",
                }
                for dimension_name in self.dimension_names:
                    values = attribute_domains[dimension_name]
                    if len(values):
                        i = np.random.randint(0, len(values))
                        record[dimension_name] = values[i]
                    else:
                        record[dimension_name] = np.nan
                records.append(record)

            chunks.append(pd.DataFrame.from_records(records))

        augmented_conversions = pd.concat(chunks)
        columns_to_use = [
            self.advertiser_column_name,
            self.user_column_name,
            "Sale",
            "click_timestamp",
            "Time_delay_for_conversion",
            "SalesAmountInEuro",
            "product_price",
            "nb_clicks_1week",
            "filter",
            *self.dimension_names,
        ]

        original_records = self.df[columns_to_use]
        augmented_conversions = pd.concat(
            [original_records, augmented_conversions.astype(original_records.dtypes)]
        )

        return augmented_conversions
    
    def _augment_conversions_with_existing_users(self, existing_users: dict, df: pd.DataFrame) -> pd.DataFrame:
        augment_rate = existing_users.get("rate") # the proability that the user converts multiple times
        ntimes = existing_users.get("ntimes", 1) # the number of additional times a user will convert

        if not augment_rate:
            msg = "no augment rate specified for augmenting existing users conversions. will not augment conversions"
            self.logger.warning(msg)
            return df

        # for each advertiser, add (conversions_augment_rate*100)% more conversions to bring their
        # attribution rate down. these conversions will be unattributed and scattered throughout the dataset
        chunks = []
        for _, dest_df in df.groupby([self.advertiser_column_name]):
            records = []
            for _, conversion in dest_df.iterrows():
                start = conversion.click_timestamp
                end = conversion.click_timestamp + max(0, conversion.Time_delay_for_conversion)
                if start == end:
                    continue

                for _ in range(ntimes):    
                    if np.random.rand() < augment_rate:
                        conversion_timestamp = np.random.randint(start, end)
                        count = np.random.randint(1, self.max_purchase_counts + 1)
                        product_price = 1
                        sales_amount_in_euro = count * product_price
                        record = {
                            **conversion.to_dict(),
                            "click_timestamp": 0,
                            "Time_delay_for_conversion": conversion_timestamp,
                            "SalesAmountInEuro": sales_amount_in_euro,
                            "product_price": product_price,
                        }
                        records.append(record)
            chunks.append(pd.DataFrame.from_records(records))

        columns_to_use = [
            self.advertiser_column_name,
            self.user_column_name,
            "Sale",
            "click_timestamp",
            "Time_delay_for_conversion",
            "SalesAmountInEuro",
            "product_price",
            "nb_clicks_1week",
            "filter",
            *self.dimension_names,
        ]
        augmented_conversions = pd.concat(chunks)[columns_to_use]
        original_records = self.df[columns_to_use]
        augmented_conversions = pd.concat(
            [original_records, augmented_conversions.astype(original_records.dtypes)]
        )

        return augmented_conversions
