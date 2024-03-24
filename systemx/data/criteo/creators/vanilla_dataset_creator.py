from datetime import datetime

from systemx.data.criteo.creators.base_creator import BaseCreator, pd

class DatasetCreator(BaseCreator):
    def __init__(self) -> None:
        super().__init__("criteo_impressions.csv", "criteo_conversions.csv")

    def specialize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=["product_id", "partner_id"])

        # create some other columns from existing data for easier reading
        df = df.assign(click_datetime=df["click_timestamp"].apply(lambda x: datetime.fromtimestamp(x)))
        df = df.assign(click_day=df["click_datetime"].apply(
            lambda x: (7 * (x.isocalendar().week - 1)) + x.isocalendar().weekday
        ))
        min_click_day = df["click_day"].min()
        df["click_day"] -= min_click_day

        df = df.assign(conversion_timestamp= df["Time_delay_for_conversion"] + df["click_timestamp"])
        df = df.assign(conversion_datetime=df["conversion_timestamp"].apply(
            lambda x: datetime.fromtimestamp(x)
        ))
        df = df.assign(conversion_day=df["conversion_datetime"].apply(
            lambda x: (7 * (x.isocalendar().week - 1)) + x.isocalendar().weekday
        ))
        df["conversion_day"] -= min_click_day

        filter = "-"
        df["filter"] = filter
        return df

    def create_impressions(self, df: pd.DataFrame) -> pd.DataFrame:
        impressions = df[["click_timestamp", "click_day", "user_id", "partner_id", "filter"]]
        impressions = impressions.sort_values(by=["click_timestamp"])
        impressions["key"] = "purchaseValue"
        return impressions
    
    def create_conversions(self, df: pd.DataFrame) -> pd.DataFrame:
        conversions = pd.DataFrame(df.loc[df.Sale == 1])[
            [
                "conversion_timestamp",
                "conversion_day",
                "user_id",
                "partner_id",
                "SalesAmountInEuro",
                "filter",
            ]
        ]  # 'product_id', 'product_price',
        conversions["SalesAmountInEuro"] = conversions["SalesAmountInEuro"].round(decimals=0)
        conversions = conversions.sort_values(by=["conversion_timestamp"])
        # Conversion is related to one query only
        max_values = conversions.groupby(["partner_id"])["SalesAmountInEuro"].max()
        max_values = max_values.reset_index(name="aggregatable_cap_value")
        max_values["aggregatable_cap_value"] = max_values["aggregatable_cap_value"]
        conversions = conversions.merge(max_values, on=["partner_id"], how="left")
        conversions["key"] = "2020"
        return conversions

if __name__ == "__main__":
    DatasetCreator().create_datasets()
