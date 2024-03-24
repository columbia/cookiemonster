import uuid
import math
import typer
import random
import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any
from omegaconf import OmegaConf
from multiprocessing import Manager, Process

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


def generate_random_date(start_date, num_days):
    # start_seconds = int((start_date - datetime.datetime(1970, 1, 1)).total_seconds())
    # end_seconds = int((end_date - datetime.datetime(1970, 1, 1)).total_seconds())
    start_seconds = 0
    end_seconds = start_seconds + (num_days * 24 * 60 * 60)
    random_seconds = random.randint(start_seconds, end_seconds)
    return datetime.datetime.utcfromtimestamp(random_seconds)


def generate_publisher_user_profile(config):
    id_attribute = "device_id"
    attribute_columns = [
        "pub_profile_1",
        "pub_profile_2",
        "pub_profile_3",
        "pub_profile_4",
        "pub_profile_5",
        "pub_profile_6",
        "pub_profile_7",
        "pub_profile_8",
    ]
    segment = "pub_segment"

    data = {}
    data[id_attribute] = [generate_uuid() for _ in range(config.num_users)]
    data[attribute_columns[0]] = generate_column_uniform(2, config.num_users)
    data[attribute_columns[1]] = generate_column_uniform(10, config.num_users)
    data[attribute_columns[2]] = generate_column_uniform(1000, config.num_users)
    data[attribute_columns[3]] = generate_column_uniform(10000, config.num_users)
    data[attribute_columns[4]] = generate_column_right_skewed(2, config.num_users)
    data[attribute_columns[5]] = generate_column_right_skewed(10, config.num_users)
    data[attribute_columns[6]] = generate_column_right_skewed(1000, config.num_users)
    data[attribute_columns[7]] = generate_column_right_skewed(10000, config.num_users)
    data[segment] = [val // 1000 + 1 for val in range(config.num_users)]

    return pd.DataFrame(data)


def generate_ad_exposure_records(config, publisher_user_profile):
    device_id = "device_id"
    exp_record_id = "exp_record_id"
    exp_timestamp = "exp_timestamp"
    exp_ad_interaction = "exp_ad_interaction"
    attribute_columns = [
        "exp_attribute_1",
        "exp_attribute_2",
        "exp_attribute_3",
        "exp_attribute_4",
        "exp_attribute_5",
        "exp_attribute_6",
        "exp_attribute_7",
        "exp_attribute_8",
    ]

    # lognormal_distribution = generate_log_normal_distribution(config.num_users)
    # records_size = sum(lognormal_distribution)

    num_impressions_per_user = config.per_day_impressions_rate * config.num_days
    normal_distribution = np.absolute(
        np.ceil(
            np.random.normal(
                loc=num_impressions_per_user, scale=0, size=config.num_users
            )
        ).astype(int)
    )
    records_size = sum(normal_distribution)

    start_date = datetime.datetime(2024, 1, 1)
    # end_date = end_date = datetime.datetime(2024, 1, 31)

    data = {}
    data[exp_record_id] = [generate_uuid() for _ in range(records_size)]
    data[device_id] = np.repeat(publisher_user_profile[device_id], normal_distribution)
    data[exp_timestamp] = [
        generate_random_date(start_date, config.num_days) for _ in range(records_size)
    ]
    data[exp_ad_interaction] = np.random.choice(
        ["view", "click"], size=records_size, p=[0.99, 0.01]
    )
    data[attribute_columns[0]] = generate_column_uniform(2, records_size)
    data[attribute_columns[1]] = generate_column_uniform(10, records_size)
    data[attribute_columns[1]] = generate_column_uniform(10000, records_size)
    data[attribute_columns[3]] = generate_column_uniform(10000, records_size)
    data[attribute_columns[4]] = generate_column_right_skewed(2, records_size)
    data[attribute_columns[5]] = generate_column_right_skewed(10, records_size)
    data[attribute_columns[6]] = generate_column_right_skewed(1000, records_size)
    data[attribute_columns[7]] = generate_column_right_skewed(10000, records_size)
    return pd.DataFrame(data)


def get_converted_users(config, publisher_user_profile, ad_exposure_records):
    id_attribute = "device_id"
    probabilities = [0.01] * config.num_users

    # Set conversion probabilities for each user
    for i in range(config.num_users):
        scaleup = 0
        probability = 0.01
        if publisher_user_profile.loc[i]["pub_profile_1"] == 1:
            scaleup += probability * 0.02
        if publisher_user_profile.loc[i]["pub_profile_2"] > 0:
            scaleup += (
                probability * 0.02 * publisher_user_profile.loc[i]["pub_profile_2"] / 9
            )
        if publisher_user_profile.loc[i]["pub_profile_3"] > 0:
            scaleup += (
                probability
                * 0.02
                * publisher_user_profile.loc[i]["pub_profile_3"]
                / 999
            )
        if publisher_user_profile.loc[i]["pub_profile_4"] > 0:
            scaleup += (
                probability
                * 0.02
                * publisher_user_profile.loc[i]["pub_profile_4"]
                / 9999
            )
        probabilities[i] += scaleup

    for _, record in ad_exposure_records.iterrows():
        id = record[id_attribute]
        probability = 0.01
        scaleup = 0
        if record["exp_attribute_5"] == 1:
            scaleup += probability * 0.02
        if record["exp_attribute_6"] > 0:
            scaleup += probability * 0.02 * record["exp_attribute_6"] / 9
        if record["exp_attribute_7"] > 0:
            scaleup += probability * 0.02 * record["exp_attribute_7"] / 999
        if record["exp_attribute_8"] > 0:
            scaleup += probability * 0.02 * record["exp_attribute_8"] / 9999
        device_index = publisher_user_profile.loc[
            publisher_user_profile[id_attribute] == id
        ].index[0]
        probabilities[device_index] += scaleup

    publisher_user_profile["probability"] = probabilities
    converted_users_count = config.num_users * config.conversion_rate_per_query
    converted_users = publisher_user_profile[id_attribute].sample(
        n=converted_users_count, weights=publisher_user_profile["probability"]
    )
    return converted_users.tolist()


def generate_conversion_records(config, publisher_user_profile, ad_exposure_records):

    converted_users = get_converted_users(
        config, publisher_user_profile, ad_exposure_records
    )
    device_id = "device_id"
    conv_record_id = "conv_record_id"
    conv_timestamp = "conv_timestamp"
    attribute_columns = [
        "conv_attribute_1",
        "conv_attribute_2",
        "conv_attribute_3",
        "conv_attribute_4",
        "conv_attribute_5",
        "conv_attribute_6",
        "conv_attribute_7",
        "conv_attribute_8",
    ]
    conv_amount = "conv_amount"
    converted_user_count = len(converted_users)
    amount_means = [1.0] * converted_user_count

    for i in range(converted_user_count):
        scaleup = 0
        mean = 1.0
        device_index = publisher_user_profile.loc[
            publisher_user_profile[device_id] == converted_users[i]
        ].index[0]
        if publisher_user_profile.loc[device_index]["pub_profile_1"] == 1:
            scaleup += mean * 0.04
        if publisher_user_profile.loc[device_index]["pub_profile_2"] > 0:
            scaleup += (
                mean
                * 0.04
                * publisher_user_profile.loc[device_index]["pub_profile_2"]
                / 9
            )
        if publisher_user_profile.loc[device_index]["pub_profile_3"] > 0:
            scaleup += (
                mean
                * 0.04
                * publisher_user_profile.loc[device_index]["pub_profile_3"]
                / 999
            )
        if publisher_user_profile.loc[device_index]["pub_profile_4"] > 0:
            scaleup += (
                mean
                * 0.04
                * publisher_user_profile.loc[device_index]["pub_profile_4"]
                / 9999
            )
        amount_means[i] += scaleup

    for _, record in ad_exposure_records.iterrows():
        id = record[device_id]

        if id not in converted_users:
            continue

        mean = 1.0
        scaleup = 0
        if record["exp_attribute_5"] == 1:
            scaleup += mean * 0.04
        if record["exp_attribute_6"] > 0:
            scaleup += mean * 0.04 * record["exp_attribute_6"] / 9
        if record["exp_attribute_7"] > 0:
            scaleup += mean * 0.04 * record["exp_attribute_7"] / 999
        if record["exp_attribute_8"] > 0:
            scaleup += mean * 0.04 * record["exp_attribute_8"] / 9999

        amount_means[converted_users.index(id)] += scaleup

    poisson_distribution = [
        1
    ] * converted_user_count  # generate_poisson_distribution(converted_user_count)
    # One conversion per user at most for each query so that we bound user contribution for the IPA baseline
    records_size = sum(poisson_distribution)
    start_date = datetime.datetime(2024, 1, 1)
    # end_date = end_date = datetime.datetime(2024, 1, 31)

    data = {}
    data[conv_record_id] = [generate_uuid() for _ in range(records_size)]
    data[device_id] = np.repeat(converted_users, poisson_distribution)
    mean_values = np.repeat(amount_means, poisson_distribution)
    data[conv_timestamp] = [
        generate_random_date(start_date, config.num_days) for _ in range(records_size)
    ]

    data[attribute_columns[0]] = generate_column_uniform(2, records_size)
    data[attribute_columns[1]] = generate_column_uniform(10, records_size)
    data[attribute_columns[2]] = generate_column_uniform(1000, records_size)
    data[attribute_columns[3]] = generate_column_uniform(10000, records_size)
    data[attribute_columns[4]] = generate_column_right_skewed(2, records_size)
    data[attribute_columns[5]] = generate_column_right_skewed(10, records_size)
    data[attribute_columns[6]] = generate_column_right_skewed(1000, records_size)
    data[attribute_columns[7]] = generate_column_right_skewed(10000, records_size)

    # Cap value to 30 to bound user contribution
    data[conv_amount] = [
        min(np.random.lognormal(mean=value, sigma=0.2).round(decimals=0), config.cap_value) for value in mean_values
    ]

    return pd.DataFrame(data)


def generate_advertiser_user_profile(converted_users):
    id_attribute = "device_id"
    attribute_columns = [
        "conv_profile_1",
        "conv_profile_2",
        "conv_profile_3",
        "conv_profile_4",
        "conv_profile_5",
        "conv_profile_6",
        "conv_profile_7",
        "conv_profile_8",
    ]
    segment = "conv_segment"

    converted_users_count = len(converted_users)

    data = {}
    data[id_attribute] = converted_users.tolist()
    data[attribute_columns[0]] = generate_column_uniform(2, converted_users_count)
    data[attribute_columns[1]] = generate_column_uniform(10, converted_users_count)
    data[attribute_columns[2]] = generate_column_uniform(1000, converted_users_count)
    data[attribute_columns[3]] = generate_column_uniform(10000, converted_users_count)
    data[attribute_columns[4]] = generate_column_right_skewed(2, converted_users_count)
    data[attribute_columns[5]] = generate_column_right_skewed(10, converted_users_count)
    data[attribute_columns[6]] = generate_column_right_skewed(
        1000, converted_users_count
    )
    data[attribute_columns[7]] = generate_column_right_skewed(
        10000, converted_users_count
    )
    data[segment] = [val // 1000 + 1 for val in range(converted_users_count)]

    return pd.DataFrame(data)


def create_synthetic_dataset(config: Dict[str, Any]):
    """
    Dataset constraints:
    a) Each user must contribute at most with one report per query
    b) We need  Q disjoint queries (say Q different product ids or some other feature) for an advertiser
    c) we have one advertiser
    d) the total conversion reports for each query is K (this means we probably need many many users)
    e) each user produces impressions for the product ids with a frequency R
    f) say a total of D days

    Some possible values:
    Q = 100, K=20K, R= [value that yields no impressions at all, value that yields at least one impression per day], D = 90
    """

    def create_data_for_query(product_id, publisher_user_profile, results):

        impressions = generate_ad_exposure_records(config, publisher_user_profile)
        conversions = generate_conversion_records(
            config, publisher_user_profile, impressions
        )

        # Process impressions and conversions
        impressions = impressions[["device_id", "exp_timestamp"]]
        conversions = conversions[["device_id", "conv_timestamp", "conv_amount"]]

        # Enforce constraints
        impressions = impressions.rename(
            columns={
                "device_id": "user_id",
                "exp_timestamp": "timestamp",
                "exp_attribute_9": "product_id",
            }
        )

        conversions = conversions.rename(
            columns={
                "device_id": "user_id",
                "conv_timestamp": "timestamp",
                "exp_attribute_9": "product_id",
                "conv_amount": "amount",
            }
        )

        impressions["advertiser_id"] = advertiser_id
        impressions["product_id"] = product_id

        conversions["advertiser_id"] = advertiser_id
        conversions["product_id"] = product_id

        # Set keys
        impressions["key"] = "_"
        conversions["key"] = "purchaseValue"

        # Set filters
        impressions["filter"] = "product_id=" + impressions["product_id"].astype(str)
        conversions["filter"] = "product_id=" + conversions["product_id"].astype(str)

        results[product_id] = {"impressions": impressions, "conversions": conversions}

    config = OmegaConf.create(config)
    advertiser_id = generate_uuid()

    processes = []
    manager = Manager()
    results = manager.dict()

    advertiser_id = generate_uuid()

    total_synthetic_impressions = []
    total_synthetic_conversions = []

    config.num_conversions_per_query = config.scheduled_batch_size * config.num_schedules

    # One conversion max allowed per user each query
    config.num_users = (
        config.num_conversions_per_query // config.conversion_rate_per_query
    )

    publisher_user_profile = generate_publisher_user_profile(config)

    for product_id in range(config.num_disjoint_queries):
        processes.append(
            Process(
                target=create_data_for_query,
                args=(product_id, publisher_user_profile, results),
            )
        )
        processes[product_id].start()

    for process in processes:
        process.join()

    total_synthetic_impressions = []
    total_synthetic_conversions = []

    for product_id, data in results.items():
        total_synthetic_impressions.append(data["impressions"])
        total_synthetic_conversions.append(data["conversions"])

    total_synthetic_impressions_df = pd.concat(total_synthetic_impressions)
    total_synthetic_conversions_df = pd.concat(total_synthetic_conversions)

    [a, b] = config.accuracy
    s = config.cap_value
    n = config.num_conversions_per_query

    def set_epsilon_given_accuracy(a, b, s, n):
      return s * math.log(1 / b) / (n * a)

    epsilon_per_query = set_epsilon_given_accuracy(a, b, s, n)
    total_synthetic_conversions_df["epsilon"] = epsilon_per_query

    total_synthetic_conversions_df["aggregatable_cap_value"] = config.cap_value

    # # Sort impressions and conversions
    total_synthetic_impressions_df = total_synthetic_impressions_df.sort_values(
        by=["timestamp"]
    )
    total_synthetic_conversions_df = total_synthetic_conversions_df.sort_values(
        by=["timestamp"]
    )

    total_synthetic_impressions_df.to_csv(
        "synthetic_impressions.csv", header=True, index=False
    )
    total_synthetic_conversions_df.to_csv(
        "synthetic_conversions.csv", header=True, index=False
    )


@app.command()
def create_dataset(omegaconf: str = "config.json"):
    omegaconf = OmegaConf.load(omegaconf)
    return create_synthetic_dataset(omegaconf)


if __name__ == "__main__":
    app()
