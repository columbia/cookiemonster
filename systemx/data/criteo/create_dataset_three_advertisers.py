# import pandas as pd
import modin.pandas as pd
import numpy as np
from datetime import datetime
import math
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
df = df.dropna(subset=["product_id", "partner_id", "user_id"])

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

# Handpick data for 3 advertisers
df = df.query("partner_id=='9D9E93D1D461D7BAE47FB67EC0E01B62' or partner_id=='F122B91F6D102E4630817566839A4F1F' or partner_id=='9FF550C0B17A3C493378CB6E2DEEE6E4'")

# Hash their product ids into three buckets
def hash_to_buckets(s):
    hash_value = hash(s)
    normalized_hash = (hash_value % 10000) / 10000
    if normalized_hash < 1/3:
        return 0
    elif normalized_hash < 2/3:
        return 1
    else:
        return 2


df["product_id_group"] = df["product_id"].apply(hash_to_buckets)


df["filter"] = "product_group_id=" + df["product_id_group"].astype(str)

# Get impressions
impressions = df[["click_timestamp", "click_day", "user_id", "partner_id", "product_id_group", "filter"]]
impressions = impressions.sort_values(by=["click_timestamp"])
impressions["key"] = "purchaseCount"

# Get conversions
conversions = pd.DataFrame(df.loc[df.Sale == 1])[
    [
        "conversion_timestamp",
        "conversion_day",
        "user_id",
        "partner_id",
        "product_id_group",
        "filter"
    ]
]
conversions = conversions.sort_values(by=["conversion_timestamp"])

def get_epsilon_from_accuracy(n):
    s=1 
    a=0.05
    b=0.01
    return s * math.log(1/b) / (n * a)


# Get epsilons from accuracy
x = conversions.groupby(["partner_id", "product_id_group"]).size().reset_index(name="count")
x["epsilon"] = x["count"].apply(get_epsilon_from_accuracy)
conversions = conversions.merge(x, on=["partner_id", "product_id_group"], how="left")

conversions["aggregatable_cap_value"] = 1
conversions["key"] = "product_group_id=" + conversions["product_id_group"].astype(str)

impressions.to_csv("criteo_impressions_three_advertisers.csv", header=True, index=False)
conversions.to_csv("criteo_conversions_three_advertisers.csv", header=True, index=False)
