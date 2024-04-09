import math
import typer

import datetime
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

app = typer.Typer()


def generate_random_dates(start_date, num_days, num_samples):
    # start_seconds = 0
    start_seconds = int((start_date - datetime.datetime(1969, 12, 31)).total_seconds())
    # end_seconds = int((end_date - datetime.datetime(1969, 12, 31)).total_seconds())
    end_seconds = start_seconds + (num_days * 24 * 60 * 60)
    random_seconds = np.random.randint(
        start_seconds, end_seconds, size=num_samples, dtype=int
    )
    return random_seconds


def generate_publisher_user_profile(config):
    data = {}
    data["user_id"] = list(range(config.num_users))
    return pd.DataFrame(data)


def generate_impressions(start_date, num_days, config, publisher_user_profile):
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
    data["user_id"] = np.repeat(publisher_user_profile["user_id"], normal_distribution)
    data["timestamp"] = generate_random_dates(start_date, num_days, records_size)
    return pd.DataFrame(data)


def generate_conversions(product_id, publisher_user_profile, advertiser_id, config):
    start_date = datetime.datetime(2024, 1, 31)
    num_days = config.num_days - 31

    publisher_user_profile["means"] = 5
    num_converted_users = int(
        config.user_participation_rate_per_query * config.num_users
    )
    # For each converted user we generate <user_contributions_per_query> conversions for each scheduling cycle
    data = {}
    batch_size = num_converted_users * config.user_contributions_per_query
    records_size = batch_size * config.num_schedules
    data["timestamp"] = np.sort(
        generate_random_dates(start_date, num_days, records_size)
    )

    batch = (
        np.ones(num_converted_users).astype(int) * config.user_contributions_per_query
    )

    user_ids = []
    conv_amounts = []

    for _ in range(config.num_schedules):
        # Randomly select users to convert
        converted_devices = publisher_user_profile.sample(
            n=num_converted_users, replace=False
        )
        distinct_devices_batch = np.repeat(converted_devices["user_id"].values, batch)
        distinct_devices_mean_values = np.repeat(
            converted_devices["means"].values, batch
        )
        # We ensure that a user doesn't appear more than <user_contributions_per_query> times within a batch
        user_ids.append(distinct_devices_batch)
        conv_amounts.append(
            np.round(
                np.random.lognormal(
                    mean=distinct_devices_mean_values,
                    sigma=0.1,
                    size=batch_size,
                ),
                1,
            )
        )

    data["user_id"] = np.hstack(user_ids)
    data["amount"] = np.hstack(conv_amounts)

    # Cap value to bound user contribution
    data["amount"] = np.clip(data["amount"], a_min=1, a_max=config.cap_value)

    conversions = pd.DataFrame(data)
    conversions = conversions.rename(
        columns={
            "user_id": "user_id",
            "conv_timestamp": "timestamp",
            "conv_amount": "amount",
        }
    )
    conversions["advertiser_id"] = advertiser_id
    conversions["product_id"] = product_id

    conversions["key"] = "product_id=" + conversions["product_id"].astype(str)
    conversions["filter"] = ""
    return conversions


def create_microbenchmark(config: OmegaConf):
    advertiser_id = 1

    # <user_contributions_per_query> conversions allowed per user for each batch
    config.num_users = math.ceil(
        config.scheduled_batch_size
        / (
            config.user_contributions_per_query
            * config.user_participation_rate_per_query
        )
    )

    print("Number of users: ", config.num_users)
    publisher_user_profile = generate_publisher_user_profile(config)

    # Set Impressions
    # Give impressions a head start of 1 month so that conversions always have an available attribution window of 30 days
    impressions_start_date = datetime.datetime(2024, 1, 1)
    num_days = config.num_days - 1
    impressions = generate_impressions(
        impressions_start_date, num_days, config, publisher_user_profile
    )
    impressions["key"] = ""
    impressions["filter"] = ""
    impressions["advertiser_id"] = advertiser_id
    impressions = impressions.sort_values(["timestamp"])
    impressions.to_csv(
        f"impressions_knob1_{config.user_participation_rate_per_query}_knob2_{config.per_day_user_impressions_rate}.csv",
        header=True,
        index=False,
    )
    impressions = None  # Let garbage collector delete this

    # Set Conversions
    conversions = []
    for product_id in range(config.num_query_types):
        print("Processing distinct query: ", product_id)
        conversions.append(
            generate_conversions(
                product_id, publisher_user_profile, advertiser_id, config
            )
        )
    conversions = pd.concat(conversions)
    conversions = conversions.sort_values(["timestamp"])

    # Set epsilons
    def _set_epsilon():
        [a, b] = config.accuracy
        expected_result = 2500 # 500 * 5
        epsilon = config.cap_value * math.log(1 / b) / (a * expected_result)
        return epsilon

    conversions["epsilon"] = _set_epsilon()

    # Set cap value
    conversions["aggregatable_cap_value"] = config.cap_value
    conversions.to_csv(
        f"conversions_knob1_{config.user_participation_rate_per_query}_knob2_{config.per_day_user_impressions_rate}.csv",
        header=True,
        index=False,
    )


@app.command()
def create_dataset(
    omegaconf: str = "config.json",
    per_day_user_impressions_rate: float = None,
    user_participation_rate_per_query: float = None,
):
    config = OmegaConf.create(OmegaConf.load(omegaconf))

    if user_participation_rate_per_query:
        config.user_participation_rate_per_query = user_participation_rate_per_query
    if per_day_user_impressions_rate:
        config.per_day_user_impressions_rate = per_day_user_impressions_rate

    print(config)
    return create_microbenchmark(config)


if __name__ == "__main__":
    app()
