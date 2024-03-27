import uuid
import math
import typer

import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any
from omegaconf import OmegaConf

# from multiprocessing import Manager, Process
import numpy as np

app = typer.Typer()


def generate_uuid():
    return uuid.uuid4().hex.upper()


def right_skewed_probability(levels, p_start=2.0 / 3, p_end=1.0 / 3):
    prob = np.linspace(p_start, p_end, levels)
    prob /= prob.sum()
    return prob


def generate_column_uniform(levels, n):
    values = [val for val in range(levels)]
    return np.random.choice(values, size=n)


def generate_column_right_skewed(levels, n):
    values = [val for val in range(levels)]
    return np.random.choice(values, size=n, p=right_skewed_probability(levels))


def generate_log_normal_distribution(n, mean_lognormal=0.5, sigma_lognormal=1.0):
    return np.ceil(
        np.random.lognormal(mean=mean_lognormal, sigma=sigma_lognormal, size=n)
    ).astype(int)


def generate_poisson_distribution(n):
    distribution = np.ceil(np.random.poisson(lam=0.5, size=n)).astype(int)
    distribution_2 = [val + 1 for val in distribution]
    return distribution_2


def generate_random_dates(start_date, num_days, num_samples):
    # start_seconds = 0
    start_seconds = int((start_date - datetime.datetime(1969, 12, 31)).total_seconds())
    end_seconds = start_seconds + (num_days * 24 * 60 * 60)
    random_seconds = np.random.randint(
        start_seconds, end_seconds, size=num_samples, dtype=int
    )
    return random_seconds


def generate_publisher_user_profile(config):
    data = {}
    data["device_id"] = list(range(config.num_users))
    data["pub_profile_1"] = generate_column_uniform(2, config.num_users)
    data["pub_profile_2"] = generate_column_uniform(10, config.num_users)
    data["pub_profile_3"] = generate_column_uniform(1000, config.num_users)
    data["pub_profile_4"] = generate_column_uniform(10000, config.num_users)
    return pd.DataFrame(data)


def generate_ad_exposure_records(start_date, config, publisher_user_profile):
    # lognormal_distribution = generate_log_normal_distribution(config.num_users)
    # records_size = sum(lognormal_distribution)

    num_impressions_per_user = config.per_day_user_impressions_rate * config.num_days
    normal_distribution = np.absolute(
        np.ceil(
            np.random.normal(
                loc=num_impressions_per_user, scale=0, size=config.num_users
            )
        ).astype(int)
    )
    records_size = sum(normal_distribution)

    data = {}
    data["device_id"] = np.repeat(
        publisher_user_profile["device_id"], normal_distribution
    )
    data["exp_timestamp"] = generate_random_dates(
        start_date, config.num_days, records_size
    )
    data["exp_attribute_5"] = generate_column_right_skewed(2, records_size)
    data["exp_attribute_6"] = generate_column_right_skewed(10, records_size)
    data["exp_attribute_7"] = generate_column_right_skewed(1000, records_size)
    data["exp_attribute_8"] = generate_column_right_skewed(10000, records_size)
    return pd.DataFrame(data)


def generate_conversion_records(
    start_date, config, publisher_user_profile, ad_exposure_records
):

    # Build conversion profile for each user
    mask_1 = publisher_user_profile["pub_profile_1"] == 1
    mask_2 = publisher_user_profile["pub_profile_2"] > 0
    mask_3 = publisher_user_profile["pub_profile_3"] > 0
    mask_4 = publisher_user_profile["pub_profile_4"] > 0

    publisher_user_profile["means"] = 1.0

    publisher_user_profile.loc[mask_1, "means"] += (
        0.04 * publisher_user_profile.loc[mask_1, "pub_profile_1"] / 9
    )
    publisher_user_profile.loc[mask_2, "means"] += (
        0.04 * publisher_user_profile.loc[mask_2, "pub_profile_2"] / 999
    )
    publisher_user_profile.loc[mask_3, "means"] += (
        0.04 * publisher_user_profile.loc[mask_3, "pub_profile_3"] / 9999
    )
    publisher_user_profile.loc[mask_4, "means"] += (
        0.04 * publisher_user_profile.loc[mask_4, "pub_profile_4"] / 99999
    )

    mask_1 = ad_exposure_records["exp_attribute_5"] == 1
    mask_2 = ad_exposure_records["exp_attribute_6"] > 0
    mask_3 = ad_exposure_records["exp_attribute_7"] > 0
    mask_4 = ad_exposure_records["exp_attribute_8"] > 0

    ad_exposure_records["scaleup"] = 0.0
    ad_exposure_records.loc[mask_1, "scaleup"] += 0.04
    ad_exposure_records.loc[mask_2, "scaleup"] += (
        0.04 * ad_exposure_records.loc[mask_2, "exp_attribute_6"] / 9
    )
    ad_exposure_records.loc[mask_3, "scaleup"] += (
        0.04 * ad_exposure_records.loc[mask_3, "exp_attribute_7"] / 999
    )
    ad_exposure_records.loc[mask_4, "scaleup"] += (
        0.04 * ad_exposure_records.loc[mask_4, "exp_attribute_8"] / 9999
    )

    scaleup = (
        ad_exposure_records.groupby("device_id")["scaleup"]
        .sum()
        .reset_index(name="scaleup")
    )
    publisher_user_profile = publisher_user_profile.merge(
        scaleup, how="inner", on="device_id"
    )

    publisher_user_profile["means"] += publisher_user_profile["scaleup"]

    num_converted_users = int(config.user_conversions_rate * config.num_users)
    # For each converted user we generate <user_contributions_per_query> conversions for each scheduling cycle
    data = {}

    batch_size = num_converted_users * config.user_contributions_per_query
    records_size = batch_size * config.num_schedules
    data["conv_timestamp"] = np.sort(
        generate_random_dates(start_date, config.num_days, records_size)
    )

    batch = (
        np.ones(num_converted_users).astype(int) * config.user_contributions_per_query
    )

    device_ids = []
    conv_amounts = []

    for _ in range(config.num_schedules):
        # Randomly select users to convert
        converted_devices = publisher_user_profile.sample(
            n=num_converted_users, replace=False
        )
        distinct_devices_batch = np.repeat(converted_devices["device_id"].values, batch)
        distinct_devices_mean_values = np.repeat(
            converted_devices["means"].values, batch
        )
        # We ensure that a user doesn't appear more than <user_contributions_per_query> times within a batch
        device_ids.append(distinct_devices_batch)
        conv_amounts.append(
            np.round(
                np.random.lognormal(
                    mean=distinct_devices_mean_values,
                    sigma=0.2,
                    size=batch_size,
                ),
                1,
            )
        )

    data["device_id"] = np.hstack(device_ids)
    data["conv_amount"] = np.hstack(conv_amounts)

    # Cap value to bound user contribution
    data["conv_amount"] = np.clip(
        data["conv_amount"], a_min=None, a_max=config.cap_value
    )

    return pd.DataFrame(data)


def create_synthetic_dataset(config: Dict[str, Any]):
    """
    Dataset constraints:
    a) Each user must contribute at most with <user_contributions_per_query> reports per query batch
    b) We need Q disjoint queries (say Q different product ids or some other feature) for an advertiser
    c) we have one advertiser
    d) the total conversion reports for each query is K (this means we probably need many many users)
    e) each user produces impressions for the product ids with a frequency I
    f) say a total of D days

    Some possible values:
    Q = 100, K=20K, I= [value that yields no impressions at all, value that yields at least one impression per day], D = 90
    """

    def create_data_for_query(product_id, publisher_user_profile, results):

        impressions_start_date = datetime.datetime(2024, 1, 1)
        impressions = generate_ad_exposure_records(
            impressions_start_date, config, publisher_user_profile
        )

        # Give impressions a head start of 1 month so that conversions always have an available attribution window of 30 days
        conversions_start_date = datetime.datetime(2024, 1, 31)
        conversions = generate_conversion_records(
            conversions_start_date, config, publisher_user_profile, impressions
        )
        # Process impressions and conversions
        impressions = impressions[["device_id", "exp_timestamp"]]
        conversions = conversions[["device_id", "conv_timestamp", "conv_amount"]]

        impressions = impressions.rename(
            columns={"device_id": "user_id", "exp_timestamp": "timestamp"}
        )

        conversions = conversions.rename(
            columns={
                "device_id": "user_id",
                "conv_timestamp": "timestamp",
                "conv_amount": "amount",
            }
        )

        impressions["advertiser_id"] = advertiser_id
        impressions["product_id"] = product_id

        conversions["advertiser_id"] = advertiser_id
        conversions["product_id"] = product_id

        # Set keys
        impressions["key"] = ""
        conversions["key"] = "purchaseValue"

        # Set filters
        impressions["filter"] = "product_id=" + impressions["product_id"].astype(str)
        conversions["filter"] = "product_id=" + conversions["product_id"].astype(str)

        results[product_id] = {"impressions": impressions, "conversions": conversions}

    config = OmegaConf.create(config)
    advertiser_id = 1  # generate_uuid()

    total_synthetic_impressions = []
    total_synthetic_conversions = []

    # <user_contributions_per_query> conversions allowed per user for each batch
    config.num_users = math.ceil(
        config.scheduled_batch_size
        / (config.user_contributions_per_query * config.user_conversions_rate)
    )

    print("Number of users: ", config.num_users)
    publisher_user_profile = generate_publisher_user_profile(config)

    results = {}
    for product_id in range(config.num_disjoint_queries):
        print("Processing query number: ", product_id)
        create_data_for_query(product_id, publisher_user_profile, results)

    # processes = []
    # manager = Manager()
    # results = manager.dict()
    # for product_id in range(config.num_disjoint_queries):
    #     processes.append(
    #         Process(
    #             target=create_data_for_query,
    #             args=(product_id, publisher_user_profile, results),
    #         )
    #     )
    #     processes[product_id].start()

    # for process in processes:
    #     process.join()

    total_synthetic_impressions = []
    total_synthetic_conversions = []

    for product_id, data in results.items():
        total_synthetic_impressions.append(data["impressions"])
        total_synthetic_conversions.append(data["conversions"])

    total_synthetic_impressions_df = pd.concat(total_synthetic_impressions)
    total_synthetic_conversions_df = pd.concat(total_synthetic_conversions)

    [a, b] = config.accuracy
    s = config.cap_value
    n = config.scheduled_batch_size

    def set_epsilon_given_accuracy(a, b, s, n):
        return s * math.log(1 / b) / (n * a)

    epsilon_per_query = set_epsilon_given_accuracy(a, b, s, n)
    total_synthetic_conversions_df["epsilon"] = epsilon_per_query

    total_synthetic_conversions_df["aggregatable_cap_value"] = config.cap_value

    # Sort impressions and conversions
    total_synthetic_impressions_df = total_synthetic_impressions_df.sort_values(
        by=["timestamp"]
    )
    # total_synthetic_conversions_df = total_synthetic_conversions_df.sort_values(
    #     by=["timestamp"]
    # )

    total_synthetic_impressions_df.to_csv(
        f"synthetic_impressions_conv_rate_{config.user_conversions_rate}_impr_rate_{config.per_day_user_impressions_rate}.csv",
        header=True,
        index=False,
    )
    total_synthetic_conversions_df.to_csv(
        f"synthetic_conversions_conv_rate_{config.user_conversions_rate}_impr_rate_{config.per_day_user_impressions_rate}.csv",
        header=True,
        index=False,
    )


@app.command()
def create_dataset(omegaconf: str = "config.json"):
    omegaconf = OmegaConf.load(omegaconf)
    return create_synthetic_dataset(omegaconf)


if __name__ == "__main__":
    app()
