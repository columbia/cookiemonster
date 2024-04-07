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
CUSTOM_ORDER_RATES = ["0.001", "0.01", "0.1", "1.0"]


def get_df(path):
    logs = LOGS_PATH.joinpath(path)
    df = load_ray_experiment(logs)
    return df


def get_budget_logs(row, results, i, **kwargs):
    scheduling_timestamps = [
        timestamp for [timestamp] in row["logs"]["scheduling_timestamps"]
    ]

    # Obtain conversions and impressions rate from dataset paths
    pattern = r"_knob1_([0-9.]+)_knob2_([0-9.]+)\.csv"
    match = re.search(pattern, row["config"]["dataset"]["impressions_path"])
    if match:
        knob1 = match.group(1)
        knob2 = match.group(2)

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
                if budget_consumed != math.inf and budget_consumed != "inf":
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
                        "max_max_budget": np.max(user_epochs_np),
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
                        "knob1": knob1,
                        "knob2": knob2,
                    }
                )
    rdf = pd.DataFrame.from_records(records).reset_index(names="queries_ran")
    rdf["queries_ran"] += 1
    results[i] = rdf


def get_bias_logs(row, results, i, **kwargs):
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
    t = kwargs.get("t", .90)
    for destination, destination_df in df.groupby(["destination"]):

        accuracies = []
        num_queries_with_null_reports = 0
        workload_size = len(destination_df)
        count_relatively_accurate = 0
        for _, log in destination_df.iterrows():
            true_sum = log["true_output"]
            biased_sum = log["aggregation_output"]
            sum_with_dp = log["aggregation_noisy_output"]

            relative_accuracy = sum_with_dp / true_sum
            if relative_accuracy >= t and relative_accuracy <= 1: # need to confirm what to do about exceeding true_sum here
                count_relatively_accurate += 1

            # Handle IPA case
            if math.isnan(biased_sum):
                accuracies.append(0)
            else:
                null_report_bias_error = true_sum - biased_sum
                num_queries_with_null_reports += (
                    null_report_bias_error == 0
                )

                # Aggregate bias across all queries ran in this workload/experiment
                accuracies.append(1 - (null_report_bias_error / true_sum))

        baseline = row["baseline"]
        num_days_per_epoch = row["num_days_per_epoch"]
        initial_budget = row["config"]["user"]["initial_budget"]

        records.append(
            {
                "destination": destination[0],
                "workload_size": workload_size,
                "fraction_queries_without_null_reports": num_queries_with_null_reports
                / workload_size,
                "average_accuracy": sum(accuracies) / len(accuracies),
                "fraction_relatively_accurate": count_relatively_accurate
                / workload_size,
                "baseline": baseline,
                "num_days_per_epoch": num_days_per_epoch,
                "initial_budget": float(initial_budget),
            }
        )
    results[i] = pd.DataFrame.from_records(records)


def analyze_results(path, type="budget", parallelize=True, **kwargs):
    dfs = []
    df = get_df(path)

    match type:
        case "budget":
            get_logs = get_budget_logs
        case "bias":
            get_logs = get_bias_logs
        # case "microbenchmark_budget":
        #     get_logs = get_microbenchmark_budget_logs
        case _:
            raise ValueError(f"Unsupported type: {type}")

    if parallelize:
        processes = []
        results = Manager().dict()
        for i, row in df.iterrows():
            processes.append(
                Process(
                    target=get_logs,
                    args=(row, results, i),
                    kwargs=kwargs
                )
            )
            processes[i].start()

        for process in processes:
            process.join()
    else:
        results = {}
        for i, row in df.iterrows():
            get_logs(row, results, i, **kwargs)

    for result in results.values():
        dfs.append(result)

    dfs = pd.concat(dfs)
    return dfs


def plot_budget_consumption_lines(df, facet_row=None):
    category_orders = {
        "baseline": CUSTOM_ORDER_BASELINES,
    }

    match facet_row:
        case None:
            facet_row = None
        case "knob1":
            category_orders.update({"knob1": CUSTOM_ORDER_RATES})
        case "knob2":
            category_orders.update({"knob2": CUSTOM_ORDER_RATES})

    def max_max_budget(df):
        fig = px.line(
            df,
            # x="num_reports",
            x="queries_ran",
            y="max_max_budget",
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

    pio.show(max_max_budget(df), renderer="png", include_plotlyjs=False)
    pio.show(max_avg_budget(df), renderer="png", include_plotlyjs=False)
    pio.show(avg_max_budget(df), renderer="png", include_plotlyjs=False)
    pio.show(avg_avg_budget(df), renderer="png", include_plotlyjs=False)

    # iplot(max_max_budget(df))
    # iplot(max_avg_budget(df))
    # iplot(avg_max_budget(df))
    # iplot(avg_avg_budget(df))


def plot_budget_consumption_bars(df, x_axis="knob1"):
    # df["key"] = (
    #     df["baseline"] + "-days_per_epoch=" + df["num_days_per_epoch"].astype(str)
    # )

    category_orders = {
        "baseline": CUSTOM_ORDER_BASELINES,
    }

    match x_axis:
        case None:
            x_axis = None
        case "knob1":
            category_orders.update({"knob1": CUSTOM_ORDER_RATES})
        case "knob2":
            category_orders.update({"knob2": CUSTOM_ORDER_RATES})

    last_query_ran = df["queries_ran"].max()
    dff = df.query("queries_ran == @last_query_ran")

    def max_max_budget(df):
        fig = px.bar(
            dff,
            x=x_axis,
            y="max_max_budget",
            color="baseline",
            title=f"Cumulative Budget Consumption",
            width=1100,
            height=400,
            barmode="group",
            facet_col="destination",
            # facet_row=facet_row,
            category_orders=category_orders,
        )
        return fig

    def max_avg_budget(df):
        fig = px.bar(
            dff,
            x=x_axis,
            y="max_avg_budget",
            color="baseline",
            title=f"Cumulative Budget Consumption",
            width=1100,
            height=400,
            barmode="group",
            facet_col="destination",
            # facet_row=facet_row,
            category_orders=category_orders,
        )
        return fig

    def avg_max_budget(df):
        fig = px.bar(
            dff,
            x=x_axis,
            y="avg_max_budget",
            color="baseline",
            title=f"Cumulative Budget Consumption",
            width=1100,
            height=400,
            barmode="group",
            facet_col="destination",
            # facet_row=facet_row,
            category_orders=category_orders,
        )
        return fig

    def avg_avg_budget(df):
        fig = px.bar(
            dff,
            x=x_axis,
            y="avg_avg_budget",
            color="baseline",
            title=f"Cumulative Budget Consumption",
            width=1100,
            height=400,
            barmode="group",
            facet_col="destination",
            category_orders=category_orders,
        )
        return fig

    pio.show(max_max_budget(df), renderer="png", include_plotlyjs=False)
    pio.show(max_avg_budget(df), renderer="png", include_plotlyjs=False)
    pio.show(avg_max_budget(df), renderer="png", include_plotlyjs=False)
    pio.show(avg_avg_budget(df), renderer="png", include_plotlyjs=False)

    # iplot(max_budget(df))
    # iplot(avg_budget(df))


def plot_null_reports_analysis(
    df: pd.DataFrame, x_axis: str = "workload_size", save_dir: str | None = None
):

    df = df.sort_values(["workload_size", "initial_budget"])

    def fraction_queries_without_dp_bias(df):
        fig = px.line(
            df,
            x=x_axis,
            y="fraction_queries_without_null_reports",
            color="baseline",
            title=f"Fraction of queries without null reports",
            width=1100,
            height=600,
            markers=True,
            range_y=[0, 1.2],
            facet_col="destination",
            category_orders={
                "baseline": CUSTOM_ORDER_BASELINES,
            },
        )
        return fig

    def workload_dp_accuracy(df):
        fig = px.line(
            df,
            x=x_axis,
            y="average_accuracy",
            color="baseline",
            title=f"Avg. relative accuracy across queries in workload",
            width=1100,
            height=600,
            markers=True,
            range_y=[0, 1.2],
            facet_col="destination",
            category_orders={
                "baseline": CUSTOM_ORDER_BASELINES,
            },
        )
        return fig

    frac_without_idp_bias_fig = fraction_queries_without_dp_bias(df)
    workload_idp_acc_fig = workload_dp_accuracy(df)

    if save_dir:
        advertiser = df["destination"].unique()[0]
        frac_without_idp_bias_fig.write_image(
            f"{save_dir}/{advertiser}_fraction_queries_with_null_reports.png"
        )
        workload_idp_acc_fig.write_image(
            f"{save_dir}/{advertiser}_average_relative_accuracy.png"
        )

    iplot(frac_without_idp_bias_fig)
    iplot(workload_idp_acc_fig)


if __name__ == "__main__":
    path = "ray/synthetic/budget_consumption_varying_conversions_rate"
    df = analyze_results(path, type="budget", parallelize=False)


# def get_microbenchmark_budget_logs(row, results, i, **kwargs):
#     scheduling_timestamps = [
#         timestamp for [timestamp] in row["logs"]["scheduling_timestamps"]
#     ]

#     # Obtain conversions and impressions rate from dataset paths
#     pattern = r"_knob1_([0-9.]+)_knob2_([0-9.]+)\.csv"
#     match = re.search(pattern, row["config"]["dataset"]["impressions_path"])
#     if match:
#         knob1 = match.group(1)
#         knob2 = match.group(2)
#     else:
#         raise ValueError("Could not find conversion and impression rates in path")

#     # Find the epochs "touched" in this experiment
#     per_destination_touched_epochs = {}
#     for [destination, epoch_min, epoch_max] in row["logs"]["epoch_range"]:
#         if destination not in per_destination_touched_epochs:
#             per_destination_touched_epochs[destination] = (math.inf, 0)

#         epoch_range = per_destination_touched_epochs[destination]
#         per_destination_touched_epochs[destination] = (
#             min(epoch_range[0], epoch_min),
#             max(epoch_range[1], epoch_max),
#         )

#     logs = row["logs"]["budget"]
#     df = pd.DataFrame.from_records(
#         logs,
#         columns=[
#             "timestamp",
#             "destination",
#             "user",
#             "budget_consumed",
#             "status",
#         ],
#     )

#     records = []
#     for destination, destination_df in df.groupby(["destination"]):

#         # Find the users "touched" in this experiment
#         num_touched_users = destination_df["user"].nunique()
#         max_user_id = int(destination_df["user"].max()) + 1
#         epoch_range = per_destination_touched_epochs[destination[0]]
#         num_touched_epochs = epoch_range[1] - epoch_range[0] + 1
#         max_epoch_index = epoch_range[1] + 1

#         # Array bigger than the size of touched users but extra users are zeroed so they will be ignored
#         users_epochs = np.zeros((max_user_id, max_epoch_index))

#         for _, log in destination_df.iterrows():
#             user = int(log["user"])
#             budget_per_epoch = log["budget_consumed"]
#             for epoch, budget_consumed in budget_per_epoch.items():
#                 assert budget_consumed != math.inf and budget_consumed != float('inf') and budget_consumed != 'inf'
#                 users_epochs[user][int(epoch)] += budget_consumed

#             if log["timestamp"] in scheduling_timestamps:
#                 sum_across_epochs = np.sum(users_epochs, axis=1)

#                 records.append(
#                     {
#                         "destination": destination[0],
#                         "num_reports": log["timestamp"],
#                         "max_avg_budget": np.max(
#                             sum_across_epochs / num_touched_epochs
#                         ),
#                         "max_max_budget": np.max(users_epochs),
#                         "avg_max_budget": np.sum(np.max(users_epochs, axis=0))
#                         / num_touched_epochs,
#                         "avg_avg_budget": np.sum(sum_across_epochs)
#                         / (num_touched_epochs * num_touched_users),
#                         "status": log["status"],
#                         "baseline": row["baseline"],
#                         "num_days_per_epoch": row["num_days_per_epoch"],
#                         "knob1": knob1,
#                         "knob2": knob2,
#                     }
#                 )
#     rdf = pd.DataFrame.from_records(records).reset_index(names="queries_ran")
#     rdf["queries_ran"] += 1
#     results[i] = rdf
