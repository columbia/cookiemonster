import numpy as np
from omegaconf import DictConfig


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
            "criteo_query_pool_impressions.csv",
            "criteo_query_pool_conversions.csv",
            "criteo_query_pool_augmented_impressions.csv"
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
        self.augment_dataset: bool = (
            config.get("augment_dataset", "false").lower() == "true"
        )
        self.advertiser_filter = config.get("advertiser_filter", [])
        self.advertiser_exclusions = config.get("advertiser_exclusions", [])

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

        if self.augment_dataset:
            df = self._augment_df_with_advertiser_bin_cover(df)

        df = df.assign(filter="")
        return df

    def create_impressions(self, df: pd.DataFrame) -> pd.DataFrame:
        df["key"] = ""
        return df

    def _create_queries(
        self,
        conversions: pd.DataFrame,
        max_purchase_counts: int,
        expected_average_purchase_counts: int,
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
                dimension_values = ad_conversions[dimension_name].unique()
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
                        sensitivity=max_purchase_counts,
                        batch_size=batch.shape[0],
                        expected_average_result=expected_average_purchase_counts,
                        relative_error=0.05,
                        failure_probability=0.01,
                    ),
                    aggregatable_cap_value=max_purchase_counts,
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

        # See notebooks/criteo_dataset_analysis.ipynb
        # 4) for an explanation for these numbers
        max_purchase_counts = 5
        expected_average_purchase_counts = 1.4

        conversions = conversions.assign(
            count=conversions.apply(
                lambda conversion: __compute_product_count(
                    conversion, max_purchase_counts
                ),
                axis=1,
            ),
            conversion_timestamp=conversions.apply(
                lambda conversion: max(0, conversion["Time_delay_for_conversion"])
                + conversion["click_timestamp"],
                axis=1,
            ),
        )
        conversions = self._create_queries(
            conversions, max_purchase_counts, expected_average_purchase_counts
        )
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

    def augument_impressions(self, df: pd.DataFrame) -> pd.DataFrame:
        augment_rate = self.config.augment_rate
        if not augment_rate:
            msg = "received request to augment dataset, but no augment rate. will not augment impressions"
            self.logger.warning(msg)
            return pd.DataFrame()
        
        attribution_window = 30 # days
        attribution_window_seconds = attribution_window * 60 * 60 * 24
        impressions_to_add = augment_rate * attribution_window

        impressions = []
        for row in df.iterrows():
            attribution_window_end = row["conversion_timestamp"]
            attribution_window_start = attribution_window_end - attribution_window_seconds
            for _ in range(impressions_to_add):
                impression = row.copy(deep=True)
                click_timestamp = np.random.randint(attribution_window_start, attribution_window_end)
                impression["click_timestamp"] = click_timestamp
                impressions.append(impression)

        return pd.DataFrame(impressions)
