from cookiemonster.data.criteo.creators.base_creator import BaseCreator, pd
from cookiemonster.data.criteo.creators.epsilon_calculator import get_epsilon_from_accuracy_for_counts

class ThreeAdversitersDatasetCreator(BaseCreator):
    """
    Rather than querying across all advertisers, we handpick three advertisers to focus on.
    The conversions here expect count queries.
    """
    
    def __init__(self) -> None:
        super().__init__("criteo_impressions_three_advertisers.csv", "criteo_conversions_three_advertisers.csv")


    def specialize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=["product_id", "partner_id", "user_id"])

        df["click_timestamp"] = df["click_timestamp"] - df["click_timestamp"].min()
        # df["click_datetime"] = df["click_timestamp"].apply(lambda x: datetime.fromtimestamp(x))
        # df["click_day"] = df["click_datetime"].apply(
        #     lambda x: (7 * (x.isocalendar().week - 1)) + x.isocalendar().weekday
        # )

        df.loc[(df["Sale"] == 1) & (df["Time_delay_for_conversion"] == -1), "Sale"] = 0

        df["conversion_timestamp"] = df["Time_delay_for_conversion"] + df["click_timestamp"]
        # df["conversion_datetime"] = df["conversion_timestamp"].apply(
        #     lambda x: datetime.fromtimestamp(x)
        # )
        # df["conversion_day"] = df["conversion_datetime"].apply(
        #     lambda x: (7 * (x.isocalendar().week - 1)) + x.isocalendar().weekday
        # )

        # Handpick data for 3 advertisers
        df = df.query(
            "partner_id=='9D9E93D1D461D7BAE47FB67EC0E01B62' or partner_id=='F122B91F6D102E4630817566839A4F1F' or partner_id=='9FF550C0B17A3C493378CB6E2DEEE6E4'"
        )

        # Hash their product ids into three buckets
        def hash_to_buckets(s):
            hash_value = hash(s)
            normalized_hash = (hash_value % 10000) / 10000
            if normalized_hash < 1 / 3:
                return 0
            elif normalized_hash < 2 / 3:
                return 1
            else:
                return 2

        df["product_id_group"] = df["product_id"].apply(hash_to_buckets)
        df["filter"] = "product_group_id=" + df["product_id_group"].astype(str)
        return df


    def create_impressions(self, df: pd.DataFrame) -> pd.DataFrame:
        # Get impressions
        impressions = df[
            [
                "click_timestamp",
                # "click_day",
                "user_id",
                "partner_id",
                "product_id_group",
                "filter",
            ]
        ]
        impressions = impressions.sort_values(by=["click_timestamp"])
        impressions["key"] = ""
        return impressions
    

    def create_conversions(self, df: pd.DataFrame) -> pd.DataFrame:
        # Get conversions
        conversions = pd.DataFrame(df.loc[df.Sale == 1])[
            [
                "conversion_timestamp",
                # "conversion_day",
                "user_id",
                "partner_id",
                "product_id_group",
                "filter",
                "SalesAmountInEuro",
                "product_price",
            ]
        ]
        conversions = conversions.sort_values(by=["conversion_timestamp"])

        # Process SalesAmountInEuro to not be smaller than product_price
        conversions.loc[
            conversions["SalesAmountInEuro"] < conversions["product_price"], "SalesAmountInEuro"
        ] = conversions["product_price"]

        # Compute counts
        conversions["count"] = conversions["SalesAmountInEuro"] // conversions["product_price"]
        conversions = conversions.drop(columns=["product_price", "SalesAmountInEuro"])

        # Cap Counts
        cap_value = 5
        conversions.loc[conversions["count"] > cap_value, "count"] = cap_value

        # Get epsilons from accuracy
        x = (
            conversions.groupby(["partner_id", "product_id_group"])
            .size()
            .reset_index(name="count")
        )
        x["epsilon"] = x["count"].apply(lambda c: get_epsilon_from_accuracy_for_counts(c, cap_value))
        x = x.drop(columns=["count"])
        conversions = conversions.merge(x, on=["partner_id", "product_id_group"], how="left")

        conversions["aggregatable_cap_value"] = cap_value
        conversions["key"] = "purchaseCount"
        return conversions
