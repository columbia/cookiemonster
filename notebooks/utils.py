import re
import math
import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.express as px
from plotly.offline import iplot
from cookiemonster.utils import LOGS_PATH
from multiprocessing import Manager, Process
from experiments.ray.analysis import load_ray_experiment

CUSTOM_ORDER_BASELINES = ["ipa", "user_epoch_ara", "cookiemonster"]
CUSTOM_ORDER_RATES = ["0.1", "0.25", "0.5", "0.75", "1.0"]


def get_df(path):
    logs = LOGS_PATH.joinpath(path)
    df = load_ray_experiment(logs)
    return df


def get_budget_logs(row, results, i):
    scheduling_timestamps = [
        timestamp for [timestamp] in row["logs"]["scheduling_timestamps"]
    ]

    # Find the epochs "touched" in this experiment
    per_destination_touched_epochs = {}
    for [destination, epoch_min, epoch_max] in row["logs"]["epoch_range"]:
        if destination not in per_destination_touched_epochs:
            per_destination_touched_epochs[destination] = (math.inf, 0)

        epoch_range = per_destination_touched_epochs[destination]
        per_destination_touched_epochs[destination] = (
            min(epoch_range[0], epoch_min),
            max(epoch_range[1], epoch_max),
        )

    logs = row["logs"]["budget"]
    df = pd.DataFrame.from_records(
        logs,
        columns=[
            "timestamp",
            "destination",
            "user",
            "epochs_window",
            "budget_consumed",
            "status",
        ],
    )

    records = []
    for destination, destination_df in df.groupby(["destination"]):

        # Find the users "touched" in this experiment
        num_touched_users = destination_df["user"].nunique()
        epoch_range = per_destination_touched_epochs[destination[0]]
        num_touched_epochs = epoch_range[1] - epoch_range[0] + 1
        max_epoch_index = epoch_range[1] + 1
        users_epochs_dict = {}

        for _, log in destination_df.iterrows():
            user = log["user"]
            budget_per_epoch = log["budget_consumed"]
            for epoch, budget_consumed in budget_per_epoch.items():

                if user not in users_epochs_dict:
                    users_epochs_dict[user] = np.zeros(max_epoch_index)
                users_epochs_dict[user][int(epoch)] += budget_consumed

            if log["timestamp"] in scheduling_timestamps:

                # Convert to a |users| x |epochs| array
                user_epochs_np = np.array(
                    [users_epochs_dict[key] for key in sorted(users_epochs_dict.keys())]
                )
                sum_across_epochs = np.sum(user_epochs_np, axis=1)

                records.append(
                    {
                        "destination": destination[0],
                        "num_reports": log["timestamp"],
                        "max_avg_budget": np.max(
                            sum_across_epochs / num_touched_epochs
                        ),
                        "avg_max_budget": np.sum(np.max(user_epochs_np, axis=0))
                        / num_touched_epochs,
                        "avg_avg_budget": np.sum(sum_across_epochs)
                        / (num_touched_epochs * num_touched_users),
                        "status": log["status"],
                        "baseline": row["baseline"],
                        "num_days_per_epoch": row["num_days_per_epoch"],
                    }
                )
    rdf = pd.DataFrame.from_records(records).reset_index(names="queries_ran")
    rdf["queries_ran"] += 1
    results[i] = rdf


def get_microbenchmark_budget_logs(row, results, i):
    scheduling_timestamps = [
        timestamp for [timestamp] in row["logs"]["scheduling_timestamps"]
    ]

    # Obtain conversions and impressions rate from dataset paths
    pattern = r"_conv_rate_([0-9.]+)_impr_rate_([0-9.]+)\.csv"
    match = re.search(pattern, row["config"]["dataset"]["impressions_path"])
    if match:
        conv_rate = match.group(1)
        impr_rate = match.group(2)
    else:
        raise ValueError("Could not find conversion and impression rates in path")

    # Find the epochs "touched" in this experiment
    per_destination_touched_epochs = {}
    for [destination, epoch_min, epoch_max] in row["logs"]["epoch_range"]:
        if destination not in per_destination_touched_epochs:
            per_destination_touched_epochs[destination] = (math.inf, 0)

        epoch_range = per_destination_touched_epochs[destination]
        per_destination_touched_epochs[destination] = (
            min(epoch_range[0], epoch_min),
            max(epoch_range[1], epoch_max),
        )

    logs = row["logs"]["budget"]
    df = pd.DataFrame.from_records(
        logs,
        columns=[
            "timestamp",
            "destination",
            "user",
            "epochs_window",
            "budget_consumed",
            "status",
        ],
    )

    records = []
    for destination, destination_df in df.groupby(["destination"]):

        # Find the users "touched" in this experiment
        num_touched_users = destination_df["user"].nunique()
        max_user_id = destination_df["user"].max() + 1
        epoch_range = per_destination_touched_epochs[destination[0]]
        num_touched_epochs = epoch_range[1] - epoch_range[0] + 1
        max_epoch_index = epoch_range[1] + 1

        # Array bigger than the size of touched users but extra users are zeroed so they will be ignored
        users_epochs = np.zeros((max_user_id, max_epoch_index))

        for _, log in destination_df.iterrows():
            user = log["user"]
            budget_per_epoch = log["budget_consumed"]
            for epoch, budget_consumed in budget_per_epoch.items():
                users_epochs[user][int(epoch)] += budget_consumed

            if log["timestamp"] in scheduling_timestamps:
                sum_across_epochs = np.sum(users_epochs, axis=1)

                records.append(
                    {
                        "destination": destination[0],
                        "num_reports": log["timestamp"],
                        "max_avg_budget": np.max(
                            sum_across_epochs / num_touched_epochs
                        ),
                        "avg_max_budget": np.sum(np.max(users_epochs, axis=0))
                        / num_touched_epochs,
                        "avg_avg_budget": np.sum(sum_across_epochs)
                        / (num_touched_epochs * num_touched_users),
                        "status": log["status"],
                        "baseline": row["baseline"],
                        "num_days_per_epoch": row["num_days_per_epoch"],
                        "conversion_rate": conv_rate,
                        "impression_rate": impr_rate,
                    }
                )
    rdf = pd.DataFrame.from_records(records).reset_index(names="queries_ran")
    rdf["queries_ran"] += 1
    results[i] = rdf


def get_bias_logs(row, results, i):
    logs = row["logs"]["query_results"]
    df = pd.DataFrame.from_records(
        logs,
        columns=[
            "timestamp",
            "destination",
            "query_id",
            "true_output",
            "aggregation_output",
            "aggregation_noisy_output",
        ],
    )

    records = []
    for destination, destination_df in df.groupby(["destination"]):

        idp_accuracies = []
        num_queries_without_idp_bias = 0
        workload_size = len(destination_df)
        # print(workload_size)

        for _, log in destination_df.iterrows():
            # Handle IPA case
            if math.isnan(log["aggregation_output"]):
                idp_accuracies.append(0)
                continue

            num_queries_without_idp_bias += (
                log["true_output"] - log["aggregation_output"] == 0
            )

            # Aggregate bias across all queries ran in this workload/experiment
            idp_error = log["true_output"] - log["aggregation_output"]
            idp_accuracies.append(1 - (idp_error / log["true_output"]))

        records.append(
            {
                "destination": destination[0],
                "workload_size": workload_size,
                "fraction_queries_without_idp_bias": num_queries_without_idp_bias
                / workload_size,
                "workload_idp_accuracy": sum(idp_accuracies) / len(idp_accuracies),
                "baseline": row["baseline"],
                "num_days_per_epoch": row["num_days_per_epoch"],
                "initial_budget": float(row["config"]["user"]["initial_budget"]),
            }
        )
    results[i] = pd.DataFrame.from_records(records)


def analyze_results(path, type="budget", parallelize=True):
    dfs = []
    df = get_df(path)

    match type:
        case "budget":
            get_logs = get_budget_logs
        case "bias":
            get_logs = get_bias_logs
        case "microbenchmark_budget":
            get_logs = get_microbenchmark_budget_logs
        case _:
            raise ValueError(f"Unsupported type: {type}")

    if parallelize:
        processes = []
        results = Manager().dict()
        for i, row in df.iterrows():
            # print(i)
            processes.append(
                Process(
                    target=get_logs,
                    args=(row, results, i),
                )
            )
            processes[i].start()

        for process in processes:
            process.join()
    else:
        results = {}
        for i, row in df.iterrows():
            get_logs(row, results, i)

    for result in results.values():
        dfs.append(result)

    dfs = pd.concat(dfs)
    return dfs


def plot_budget_consumption(df, facet_row=None):
    # df["key"] = (
    #     df["baseline"] + "-days_per_epoch=" + df["num_days_per_epoch"].astype(str)
    # )

    category_orders = {
        "baseline": CUSTOM_ORDER_BASELINES,
    }

    match facet_row:
        case None:
            facet_row = None
        case "conversion_rate":
            category_orders.update({"conversion_rate" : CUSTOM_ORDER_RATES})
        case "impression_rate":
            category_orders.update({"impression_rate" : CUSTOM_ORDER_RATES})


    def max_avg_budget(df):
        fig = px.line(
            df,
            # x="num_reports",
            x="queries_ran",
            y="max_avg_budget",
            color="baseline",
            title=f"Cumulative Budget Consumption",
            width=1100,
            height=1200,
            markers=True,
            facet_col="destination",
            facet_row=facet_row,
            category_orders=category_orders,
        )
        return fig

    def avg_max_budget(df):
        fig = px.line(
            df,
            # x="num_reports",
            x="queries_ran",
            y="avg_max_budget",
            color="baseline",
            title=f"Cumulative Budget Consumption",
            width=1100,
            height=1200,
            markers=True,
            facet_col="destination",
            facet_row=facet_row,
            category_orders=category_orders,
        )
        return fig

    def avg_avg_budget(df):
        fig = px.line(
            df,
            # x="num_reports",
            x="queries_ran",
            y="avg_avg_budget",
            color="baseline",
            title=f"Cumulative Budget Consumption",
            width=1100,
            height=1200,
            markers=True,
            facet_col="destination",
            facet_row=facet_row,
            category_orders=category_orders,
        )
        return fig

    pio.show(max_avg_budget(df), renderer="png", include_plotlyjs=False)
    pio.show(avg_max_budget(df), renderer="png", include_plotlyjs=False)
    pio.show(avg_avg_budget(df), renderer="png", include_plotlyjs=False)

    # iplot(max_budget(df))
    # iplot(avg_budget(df))


def plot_accuracy(df, x_axis="workload_size"):

    df = df.sort_values(["workload_size", "initial_budget"])

    def fraction_queries_without_idp_bias(df):
        fig = px.line(
            df,
            x=x_axis,
            y="fraction_queries_without_idp_bias",
            color="baseline",
            title=f"Fraction of queries reaching accuracy target",
            width=1100,
            height=600,
            markers=True,
            facet_col="destination",
            # facet_row=facet_row,
            category_orders={
                "baseline": CUSTOM_ORDER_BASELINES,
            },
        )
        return fig

    def workload_idp_accuracy(df):
        fig = px.line(
            df,
            x=x_axis,
            y="workload_idp_accuracy",
            color="baseline",
            title=f"Avg. relative accuracy across queries in workload",
            width=1100,
            height=600,
            markers=True,
            facet_col="destination",
            # facet_row=facet_row,
            category_orders={
                "baseline": CUSTOM_ORDER_BASELINES,
            },
        )
        return fig

    iplot(fraction_queries_without_idp_bias(df))
    iplot(workload_idp_accuracy(df))


if __name__ == "__main__":
    path = "ray/synthetic/budget_consumption_varying_conversions_rate"
    df = analyze_results(path, type="budget", parallelize=False)
