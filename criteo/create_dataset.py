# import pandas as pd
import modin.pandas as pd
import numpy as np
from datetime import datetime
import os

os.environ["MODIN_ENGINE"] = "ray"

DATA_FILE = "Criteo_Conversion_Search/CriteoSearchData"
dtype = {
    "Sale": np.int32,
    "SalesAmountInEuro": np.float64,
    "Time_delay_for_conversion": np.int32,
    "click_timestamp": np.int32,
    "nb_clicks_1week": pd.Int64Dtype(),
    "product_price": np.float64,
    "product_age_group": str,
    "device_type": str,
    "audience_id": str,
    "product_gender": str,
    "product_brand": str,
    "product_category1": str,
    "product_category2": str,
    "product_category3": str,
    "product_category4": str,
    "product_category5": str,
    "product_category6": str,
    "product_category7": str,
    "product_country": str,
    "product_id": str,
    "product_title": str,
    "partner_id": str,
    "user_id": str,
}
na_values = {
    "click_timestamp": "0",
    "nb_clicks_1week": "-1",
    "product_price": "-1",
    "product_age_group": "-1",
    "device_type": "-1",
    "audience_id": "-1",
    "product_gender": "-1",
    "product_brand": "-1",
    "product_category1": "-1",
    "product_category2": "-1",
    "product_category3": "-1",
    "product_category4": "-1",
    "product_category5": "-1",
    "product_category6": "-1",
    "product_category7": "-1",
    "product_country": "-1",
    "product_id": "-1",
    "product_title": "-1",
    "partner_id": "-1",
    "user_id": "-1",
}
columns_to_drop = [
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
]


df = pd.read_csv(
    DATA_FILE,
    names=dtype.keys(),
    dtype=dtype,
    na_values=na_values,
    header=None,
    sep="\t",
)
df = df.drop(columns=columns_to_drop)
df = df.dropna(subset=["product_id", "partner_id"])

# create some other columns from existing data for easier reading
df["click_datetime"] = df["click_timestamp"].apply(lambda x: datetime.fromtimestamp(x))
df["click_day"] = df["click_datetime"].apply(
    lambda x: (7 * (x.isocalendar().week - 1)) + x.isocalendar().weekday
)
min_click_day = df["click_day"].min()
df["click_day"] -= min_click_day

df["conversion_timestamp"] = df["Time_delay_for_conversion"] + df["click_timestamp"]
df["conversion_datetime"] = df["conversion_timestamp"].apply(
    lambda x: datetime.fromtimestamp(x)
)
df["conversion_day"] = df["conversion_datetime"].apply(
    lambda x: (7 * (x.isocalendar().week - 1)) + x.isocalendar().weekday
)
df["conversion_day"] -= min_click_day

filter = "-"
df["filter"] = filter

impressions = df[["click_timestamp", "click_day", "user_id", "partner_id", "filter"]]
impressions = impressions.sort_values(by=["click_timestamp"])
impressions["key"] = "purchaseValue"

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

impressions.to_csv("criteo_impressions.csv", header=True, index=False)
conversions.to_csv("criteo_conversions.csv", header=True, index=False)
