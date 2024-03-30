from datetime import datetime

from omegaconf import DictConfig

from cookiemonster.data.criteo.creators.base_creator import BaseCreator, pd
from cookiemonster.data.criteo.creators.epsilon_calculator import (
    get_epsilon_from_accuracy_for_counts,
)


class PartnerCountDatasetCreator(BaseCreator):
    """
    Rather than looking at the product_id, we treat the partner_id as the "product" that the user bought.
    Additionally, the conversions expect count queries.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__(
            config,
            "criteo_partner_counts_impressions.csv",
            "criteo_partner_counts_conversions.csv",
        )

    def specialize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=["product_id", "partner_id"])

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

        df.drop(columns=[
            "product_category1",
            "product_category2",
            "product_category3",
            "product_category4",
            "product_category5",
            "product_category6",
            "product_category7",
            "nb_clicks_1week",
            "device_type",
            "product_title",
            "product_brand",
            "product_gender",
            "audience_id",
            "product_age_group",
            "product_country",
        ])

        return df

    def create_impressions(self, df: pd.DataFrame) -> pd.DataFrame:
        impressions = df[
            ["click_timestamp", "click_day", "user_id", "partner_id", "filter"]
        ]
        impressions = impressions.sort_values(by=["click_timestamp"])
        impressions["key"] = "-"
        return impressions

    def create_conversions(self, df: pd.DataFrame) -> pd.DataFrame:
        conversions = pd.DataFrame(df.loc[df.Sale == 1])[
            [
                "conversion_timestamp",
                "conversion_day",
                "user_id",
                "partner_id",
                "SalesAmountInEuro",
                "product_price",
                "filter",
            ]
        ]

        conversions = conversions.sort_values(by=["conversion_timestamp"])

        # Process SalesAmountInEuro to not be smaller than product_price
        conversions.loc[
            conversions["SalesAmountInEuro"] < conversions["product_price"],
            "SalesAmountInEuro",
        ] = conversions["product_price"]

        # Compute counts
        cap_value = 5
        conversions = conversions.assign(
            count=(
                conversions["SalesAmountInEuro"] // conversions["product_price"]
            ).apply(lambda c: min(c, cap_value))
        )
        conversions = conversions.drop(columns=["product_price", "SalesAmountInEuro"])

        # Get epsilons from accuracy
        x = conversions.groupby(["partner_id"]).size().reset_index(name="count")
        x["epsilon"] = x["count"].apply(
            lambda c: get_epsilon_from_accuracy_for_counts(c, cap_value)
        )
        x = x.drop(columns=["count"])
        conversions = conversions.merge(x, on=["partner_id"], how="left")

        conversions["aggregatable_cap_value"] = cap_value
        conversions["key"] = "purchaseCount"
        return conversions
