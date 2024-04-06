import itertools
from omegaconf import DictConfig
import prtpy
from uuid import uuid4


from data.criteo.creators.base_creator import BaseCreator, pd
from data.criteo.creators.epsilon_calculator import get_epsilon_from_accuracy_for_counts


class QueryPoolDatasetCreator(BaseCreator):

    def __init__(self, config: DictConfig) -> None:
        super().__init__(
            config,
            "criteo_query_pool_impressions.csv",
            "criteo_query_pool_conversions.csv",
        )
        self.used_dimension_names = set()

        self.advertiser_column_name = "partner_id"
        self.product_column_name = "product_id"
        self.user_column_name = "user_id"

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

        self.conversion_columns_to_drop = [
            "SalesAmountInEuro",
            "product_price",
            "nb_clicks_1week",
            "Time_delay_for_conversion",
            "Sale",
            "click_timestamp",
        ]
        self.impression_columns_to_use = [
            "click_timestamp",
            "user_id",
            "partner_id",
            "filter",
        ]
        self.enforce_one_user_contribution_per_query = (
            config.enforce_one_user_contribution_per_query
        )
        self.max_conversions_required_for_dp = config.max_conversions_required_for_dp
        self.min_conversions_required_for_dp = config.min_conversions_required_for_dp
        self.estimated_conversion_rate = config.estimated_conversion_rate
        self.plot_query_pool: bool = (
            config.get("plot_query_pool", "false").lower() == "true"
        )
        self.augment_dataset: bool = (
            config.get("augment_dataset", "false").lower() == "true"
        )
        self.advertiser_filter = config.get("advertiser_filter", [])

    def _run_basic_specialization(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("running basic df specialization...")
        df = df.assign(filter="")
        return df

    # TODO: [PM] Bring up with the group. Perhaps we will want to bring back the other
    # grouping methods (from previous commits)
    def _augment_df_with_advertiser_bin_cover(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        We upsample the dataset to increase the number of queries to run. To accomplish this,
        for each advertiser, we create synthetic groupings of products that could represent
        another way of categorizing those products together.

        A query is accepted into an advertiser's query pool as long as the number of conversions
        driving that query is >= self.min_conversions_required_for_dp. Given a rough estimate of
        a typical conversion rate across products, self.estimated_conversion_rate, we back into
        a minimum number of impressions required for dp queries, min_impressions_required_for_dp.

        Using this, our upsampling problem can be phrased as: for each advertiser, group their
        products in such a way that maximizes the number of additional queries asked. In other words,
        group products into bins such that the sum of the impression counts of products within a bin
        is at least min_impressions_required_for_dp, and such that the number of bins is maximal.

        This is an example of the bin-covering problem where the bin size is min_impressions_required_for_dp.
        We use the prtpy implementation of Csirik-Frenk-Labbe-Zhang's 3/4 approximation.
        https://en.wikipedia.org/wiki/Bin_covering_problem#Three-classes_bin-filling_algorithm.
        """
        min_impressions_required_for_dp = (
            self.min_conversions_required_for_dp // self.estimated_conversion_rate
        )

        bin_assignments = {}
        for advertiser in df[self.advertiser_column_name].unique():
            advertiser_chunk = df.loc[df[self.advertiser_column_name] == advertiser]
            product_impression_counts = advertiser_chunk.groupby(
                [self.product_column_name]
            ).size()

            count_map = {}
            for product_impression_count in product_impression_counts.items():
                products = count_map.get(product_impression_count[1])
                if products:
                    products.append(product_impression_count[0])
                else:
                    count_map[product_impression_count[1]] = [
                        product_impression_count[0]
                    ]

            bins = prtpy.pack(
                algorithm=prtpy.covering.threequarters,
                binsize=min_impressions_required_for_dp,
                items=product_impression_counts,
            )
            if len(bins) >= 2:
                bin_names = []
                for bin in bins:
                    bin_name = str(uuid4()).upper().replace("-", "")
                    bin_names.append(bin_name)
                    for count in bin:
                        product = count_map[count].pop()
                        bin_assignments[(advertiser, product)] = bin_name

                # we've maxed out the number of bins we can create, so just
                # stick the unbinned products in the bins in a round robin
                # fashion.
                unbinned_products = itertools.chain(*count_map.values())

                num_bins = len(bin_names)
                for i, unbinned in enumerate(unbinned_products):
                    bin_assignments[(advertiser, unbinned)] = bin_names[i % num_bins]

        df = df.assign(
            synthetic_category=df.apply(
                lambda row: bin_assignments.get(
                    (row[self.advertiser_column_name], row[self.product_column_name]),
                    pd.NA,
                ),
                axis=1,
            )
        )
        self.dimension_names.append("synthetic_category")
        return df

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

        if self.augment_dataset:
            df = self._augment_df_with_advertiser_bin_cover(df)

        df = self._run_basic_specialization(df)
        return df

    def create_impressions(self, df: pd.DataFrame) -> pd.DataFrame:
        impressions = df[self.impression_columns_to_use]
        impressions = impressions.sort_values(by=["click_timestamp"])
        impressions["key"] = ""
        return impressions

    def _compute_product_count(self, conversion, cap: int) -> int:
        sell_price = conversion["SalesAmountInEuro"]
        offer_price = conversion["product_price"]
        if sell_price and offer_price:
            return min(cap, sell_price // offer_price)
        elif offer_price:
            return 0
        else:
            return 1

    def _create_queries(
        self, conversions: pd.DataFrame, max_purchase_counts: int
    ) -> pd.DataFrame:

        def __mark_include(row: pd.Series, seen_users: set, row_count: int):
            if row_count == self.max_conversions_required_for_dp:
                row_count = 0
                seen_users = set()

            user = row[self.user_column_name]
            if user in seen_users:
                return False
            else:
                seen_users.add(user)
                row_count += 1
                return True

        seen_users = set()
        row_count = 0
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

                    # we have our total query. need to iterate row by row taking unique users up until
                    # max conversion count for dp

                    if self.enforce_one_user_contribution_per_query:
                        query_result = query_result.assign(
                            include=query_result.apply(
                                lambda row: __mark_include(row, seen_users, row_count),
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
                    num_big_reports = (
                        query_result_length // self.max_conversions_required_for_dp
                    )
                    i = 0
                    while i < num_big_reports:
                        start = i * self.max_conversions_required_for_dp
                        end = (i + 1) * self.max_conversions_required_for_dp
                        batch = query_result.iloc[start : end]
                        assert batch.shape[0] >= self.min_conversions_required_for_dp
                        assert batch.shape[0] <= self.max_conversions_required_for_dp
                        advertiser_queries = query_batches.get(advertiser, [])
                        advertiser_queries.append(batch)
                        self.used_dimension_names.add(dimension_name)
                        i += 1

                    i = i * self.max_conversions_required_for_dp
                    if (
                        i < query_result_length
                        and query_result_length - i
                        >= self.min_conversions_required_for_dp
                    ):
                        batch = query_result.iloc[i:]
                        assert batch.shape[0] >= self.min_conversions_required_for_dp
                        assert batch.shape[0] <= self.max_conversions_required_for_dp
                        advertiser_queries = query_batches.get(advertiser, [])
                        advertiser_queries.append(batch)
                        self.used_dimension_names.add(dimension_name)

        final_batches = []
        for _, batches in query_batches:
            for i, batch in enumerate(batches):
                final_batch = batch.assign(
                    epsilon=get_epsilon_from_accuracy_for_counts(
                        batch.shape[0], max_purchase_counts
                    ),
                    aggregatable_cap_value=max_purchase_counts,
                    key=i,
                )
                final_batches.append(final_batch)

        return pd.concat(final_batches)

    def create_conversions(self, df: pd.DataFrame) -> pd.DataFrame:
        conversions = df.loc[df.Sale == 1]

        """
        purchase count description across all conversion events:
        count    1.279493e+06
        mean     4.705447e+00
        std      1.581949e+02
        min      0.000000e+00
        25%      1.000000e+00
        50%      1.000000e+00
        75%      2.000000e+00
        max      8.661200e+04
        skew     352.22094940782813

        So, 5 seems like a reasonable cap value.
        """
        max_purchase_counts = 5

        conversions = conversions.assign(
            count=conversions.apply(
                lambda conversion: self._compute_product_count(
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
        conversions = conversions.drop(columns=self.conversion_columns_to_drop)
        conversions = self._create_queries(conversions, max_purchase_counts)

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

        if self.plot_query_pool:
            import matplotlib.pyplot as plt

            ax = advertiser_query_count.plot(
                # x="advertiser", # the advertiser long names make it impossible to read
                y=["query_count"],
                kind="bar",
            )
            plt.tight_layout()
            fig = ax.get_figure()
            fig.savefig("./criteo_advertiser_query_count.png")

            ax = advertiser_epsilon_sum.plot(
                # x="advertiser", # the advertiser long names make it impossible to read
                y=["epsilon_sum"],
                kind="bar",
            )
            plt.tight_layout()
            fig = ax.get_figure()
            fig.savefig("./criteo_advertiser_epsilon_sum.png")

        pd.set_option("display.max_rows", None)
        self.logger.info(f"Query count per advertiser:\n{advertiser_query_count}")
        self.logger.info(f"Sum of epsilons per advertiser:\n{advertiser_epsilon_sum}")
        pd.reset_option("display.max_rows")
