from datetime import datetime
import itertools
from omegaconf import DictConfig
import prtpy
from uuid import uuid4


from data.criteo.creators.base_creator import BaseCreator, pd
from data.criteo.creators.epsilon_calculator import get_epsilon_from_accuracy_for_counts

QueryKey = tuple[str, str, str]  # (advertiser_value, dimension_value, dimension_name)


class QueryPoolDatasetCreator(BaseCreator):

    def __init__(self, config: DictConfig) -> None:
        super().__init__(
            config,
            "criteo_query_pool_impressions.csv",
            "criteo_query_pool_conversions.csv",
        )
        self.query_pool: dict[QueryKey, int] = {}  # query -> number of conversions

        self.advertiser_column_name = "partner_id"
        self.product_column_name = "product_id"
        self.user_column_name = "user_id"

        self.dimension_names = [
            self.advertiser_column_name,
            self.product_column_name,
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
            "click_day",
            "click_datetime",
        ]
        self.impression_columns_to_use = [
            "click_timestamp",
            "click_day",
            "user_id",
            "partner_id",
            "filter",
        ]
        self.min_conversions_required_for_dp = config.min_conversions_required_for_dp
        self.estimated_conversion_rate = config.estimated_conversion_rate

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

        # TODO: [PM] maybe using a dataframe for bin assignments rather than a map is faster
        # than the final, row-wise apply?
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
        return df

    def _populate_query_pools(self, df: pd.DataFrame) -> None:
        self.logger.info("populating the query pools...")
        conversions = df.loc[(df.Sale == 1)]
        for dimension_name in self.dimension_names:
            conversions = conversions.assign(dimension_name=dimension_name)
            counts = conversions.groupby(
                [self.advertiser_column_name, dimension_name, "dimension_name"]
            ).Sale.count()
            counts = counts[counts >= self.min_conversions_required_for_dp]
            if not counts.empty:
                self.query_pool.update(counts.to_dict())

        keys = [x for x in self.query_pool.keys()]
        keys.sort()
        log_lines = []
        for key in keys:
            count = self.query_pool[key]
            (partner_id, dimension, dimension_name) = key
            log_lines.append(
                f"{count} total conversion records from partner_id ({partner_id}), {dimension_name} ({dimension})"
            )

        query_pool_contents = str.join("\n\t", log_lines)
        self.logger.info(
            f"Generated the following query pool:\n\t{query_pool_contents}\n"
        )

    def specialize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(
            subset=[
                self.advertiser_column_name,
                self.user_column_name,
                self.product_column_name,
            ]
        )

        # TODO: [PM] should we include the outlier when really running this?
        # df = df[df.partner_id != "E3DDEB04F8AFF944B11943BB57D2F620"]

        df = self._augment_df_with_advertiser_bin_cover(df)

        self._populate_query_pools(df)
        df = self._run_basic_specialization(df)
        return df

    def create_impressions(self, df: pd.DataFrame) -> pd.DataFrame:
        impressions = df[self.impression_columns_to_use]
        impressions = impressions.sort_values(by=["click_timestamp"])
        impressions["key"] = "-"
        # TODO: [PM] drop random sample of impressions?
        return impressions

    def _get_used_dimension_names(self) -> set:
        return set(map(lambda x: x[2], self.query_pool.keys()))

    def _create_record_per_query(self, conversions: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("creating a conversion record per query...")
        conversion_chunks = []
        for dimension in self._get_used_dimension_names():
            conversions = conversions.assign(
                query_key=conversions.apply(
                    lambda conversion: (
                        conversion[self.advertiser_column_name],
                        conversion[dimension],
                        dimension,
                    ),
                    axis=1,
                )
            )
            conversions = conversions.assign(
                included=conversions.query_key.isin(self.query_pool.keys())
            )
            conversions_to_use = conversions.loc[conversions.included]

            conversions_to_use = conversions_to_use.assign(
                conversion_count=conversions_to_use.apply(
                    lambda conversion: self.query_pool[
                        (
                            conversion[self.advertiser_column_name],
                            conversion[dimension],
                            dimension,
                        )
                    ],
                    axis=1,
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
        purchase_counts = conversions.apply(
            QueryPoolDatasetCreator._compute_product_count, axis=1
        )

        """
        TODO: [PM] what should we cap our purchase counts at?
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

        so, maybe 5 is reasonable. should we calculate this a different way generally?
        """
        max_purchase_counts = 5

        conversions = conversions.assign(count=purchase_counts)

        self.log_description_of_conversions(conversions)

        conversions = self._create_record_per_query(conversions)

        conversions = conversions.assign(
            epsilon=conversions["conversion_count"].apply(
                lambda conversion_count: get_epsilon_from_accuracy_for_counts(
                    conversion_count, max_purchase_counts
                )
            ),
            key=conversions.apply(
                lambda conversion: f"{str.join('|', conversion.query_key)}|purchaseCount",
                axis=1,
            ),
            aggregatable_cap_value=max_purchase_counts,
        )

        query_epsilons = str.join(
            "",
            conversions.apply(
                lambda conversion: f"\t{conversion['query_key']}, epsilon: {conversion['epsilon']}\n",
                axis=1,
            ).unique(),
        )
        self.logger.info(f"Query pool epsilons:\n{query_epsilons}")

        unused_dimension_names = (
            set(self.dimension_names) - self._get_used_dimension_names()
        )
        columns_we_created = ["query_key", "conversion_count"]

        to_drop = [
            *unused_dimension_names,
            *columns_we_created,
            *self.conversion_columns_to_drop,
        ]

        return conversions.drop(columns=to_drop)

    def log_description_of_conversions(self, conversions):
        counts = conversions["count"]
        self.logger.info(
            f"purchase count description across all conversion events:\n{counts.describe()}"
        )
        self.logger.info(
            f"purchase count skew across all conversion events: {counts.skew()}"
        )

        true_sums = []
        for dimension in self.dimension_names:
            true_sums_for_dimension = conversions.groupby(
                [self.advertiser_column_name, dimension]
            )["count"].sum()
            true_sums.append(true_sums_for_dimension)

        all_true_summary_reports = pd.concat(true_sums).reset_index(drop=True)
        self.logger.info(
            f"True summary reports across all queries description (no synthetic features added):\n{all_true_summary_reports.describe()}"
        )
        self.logger.info(
            f"True summary reports across all queries skew (no synthetic features added): {all_true_summary_reports.skew()}"
        )
