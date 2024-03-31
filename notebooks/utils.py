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


def get_df(path):
    logs = LOGS_PATH.joinpath(path)
    df = load_ray_experiment(logs)
    return df


def analyze_budget_consumption(path):

    def get_logs(row, results, i):

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

            print(
                i,
                "Touched Users:",
                num_touched_users,
                "Max User id:",
                max_user_id,
                "Conversion rate:",
                conv_rate,
                "Impression rate:",
                impr_rate,
                "Total users:",
                max_user_id,
                "Min epoch:",
                epoch_range[0],
                "Max epoch:",
                epoch_range[1],
            )

            # Array bigger than the size of touched users but extra users are zeroed so they will be ignored
            users_epochs = np.zeros((max_user_id, max_epoch_index))

            for _, log in destination_df.iterrows():
                user = log["user"]
                budget_per_epoch = log["budget_consumed"]

                # print(budget_per_epoch)
                for epoch, budget_consumed in budget_per_epoch.items():
                    users_epochs[user][int(epoch)] += budget_consumed

                if log["timestamp"] in scheduling_timestamps:
                    # print(log["timestamp"])
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
                            # "optimization": row["optimization"],
                            "num_days_per_epoch": row["num_days_per_epoch"],
                            # "num_days_attribution_window": row[
                            # "num_days_attribution_window"
                            # ],
                            "conversion_rate": conv_rate,
                            "impression_rate": impr_rate,
                        }
                    )
        rdf = pd.DataFrame.from_records(records).reset_index(names="queries_ran")
        rdf["queries_ran"] += 1
        results[i] = rdf

    dfs = []
    df = get_df(path)

    # In Parallel
    processes = []
    results = Manager().dict()
    for i, row in df.iterrows():
        print(i)
        processes.append(
            Process(
                target=get_logs,
                args=(row, results, i),
            )
        )
        processes[i].start()

    for process in processes:
        process.join()

    # # Sequentially
    # results = {}
    # for i, row in df.iterrows():
    #     get_logs(row, results, i)

    for result in results.values():
        dfs.append(result)

    dfs = pd.concat(dfs)
    return dfs


def analyze_bias(path):

    def get_logs(row, results, i):
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

            total_nulls_error = 0
            total_dp_error = 0
            total_error = 0

            accuracy_per_query = []
            for _, log in destination_df.iterrows():
                # Aggregate bias across all queries ran in this workload/experiment
                nulls_error = log["true_output"] - log["aggregation_output"]
                nulls_accuracy = 1 - (nulls_error / log["true_output"])
                accuracy_per_query.append(nulls_accuracy)
                
                end2end_error = abs(log["aggregation_noisy_output"] - log["true_output"])
                end2end_accuracy = 1 - (end2end_error / log["true_output"])
                
                total_dp_error += abs(log["aggregation_noisy_output"] - log["aggregation_output"])
                total_error += abs(log["true_output"] - log["aggregation_noisy_output"])
                # print(log)

            records.append(
                {
                    "destination": destination[0],
                    "workload_nulls_error": workload_bias,
                    "workload_size": row["workload_size"],
                    "baseline": row["baseline"],
                    "optimization": row["optimization"],
                    "num_days_per_epoch": row["num_days_per_epoch"],
                    "num_days_attribution_window": row[
                        "num_days_attribution_window"
                    ],
                }
            )
        results[i] = pd.DataFrame.from_records(records)

    dfs = []
    df = get_df(path)

    # In Parallel
    processes = []
    results = Manager().dict()
    for i, row in df.iterrows():
        print(i)
        processes.append(
            Process(
                target=get_logs,
                args=(row, results, i),
            )
        )
        processes[i].start()

    for process in processes:
        process.join()

    # # Sequentially
    # results = {}
    # for i, row in df.iterrows():
    #     get_logs(row, results, i)

    for result in results.values():
        dfs.append(result)

    dfs = pd.concat(dfs)
    return dfs


def plot_budget_consumption(df, facet_row="conversion_rate"):
    # df["key"] = (
    #     df["baseline"] + "-days_per_epoch=" + df["num_days_per_epoch"].astype(str)
    # )
    custom_order_baselines = ["ipa", "user_epoch_ara", "cookiemonster"]
    custom_order_rates = (
        []
        if facet_row == "num_days_per_epoch"
        else ["0.1", "0.25", "0.5", "0.75", "1.0"]
    )

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
            category_orders={
                "baseline": custom_order_baselines,
                facet_row: custom_order_rates,
            },
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
            category_orders={
                "baseline": custom_order_baselines,
                facet_row: custom_order_rates,
            },
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
            category_orders={
                "baseline": custom_order_baselines,
                facet_row: custom_order_rates,
            },
        )
        return fig

    pio.show(max_avg_budget(df), renderer="png", include_plotlyjs=False)
    pio.show(avg_max_budget(df), renderer="png", include_plotlyjs=False)
    pio.show(avg_avg_budget(df), renderer="png", include_plotlyjs=False)

    # iplot(max_budget(df))
    # iplot(avg_budget(df))


def plot_bias(df):
    def bias(df):
        fig = px.line(
            df,
            x="workload_size",
            y="workload_bias",
            color="baseline",
            title=f"Total Workload Bias",
            width=1100,
            height=600,
            markers=True,
            facet_row="destination",
        )
        return fig

    iplot(bias(df))


if __name__ == "__main__":
    path = "ray/synthetic/budget_consumption_varying_conversions_rate"
    df = analyze_budget_consumption(path)
