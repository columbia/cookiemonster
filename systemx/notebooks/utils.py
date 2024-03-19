import numpy as np
from typing import List
import pandas as pd

# import modin as pd
import plotly.express as px
from plotly.offline import iplot
from multiprocessing import Manager, Process

from systemx.utils import LOGS_PATH
from experiments.ray.analysis import load_ray_experiment


def get_df(path):
    logs = LOGS_PATH.joinpath(path)
    df = load_ray_experiment(logs)
    return df


def get_per_epoch_budget_consumption(filters: List[str], initial_budget):

    if filters:
        max_index = max([max(map(int, d.keys())) if d else 0 for d in filters])
        # print(max_index)
        dense_matrix = np.zeros((len(filters), max_index + 1))
        for i, d in enumerate(filters):
            if d:
                keys = np.array(list(map(int, d.keys())))
                values = initial_budget - np.array(list(d.values()))
                dense_matrix[i, keys] = values
        avg_consumed_budget_per_epoch = np.mean(dense_matrix, axis=0)
        sum_consumed_budget_per_epoch = np.sum(dense_matrix, axis=0)
        return avg_consumed_budget_per_epoch, sum_consumed_budget_per_epoch, max_index
    return np.zeros(1), np.zeros(1), 0


def analyze_budget_consumption(path, get_user_logs=True, get_destination_logs=True):
    df = get_df(path)

    def process_experiment_logs(row, experiments, get_user_logs, get_destination_logs):
        initial_budget = row["initial_budget"]
        optimization = row["optimization"]
        num_days_per_epoch = row["num_days_per_epoch"]
        b = row["remaining_budget_per_user_per_destination_per_epoch"]

        users_data = []
        destinations_data = {}

        total_num_users_converted = 0

        for i, user in enumerate(b):
            if get_user_logs:
                per_user_data = {
                    "user_id": i,
                    "num_unique_destinations_for_whom_converted": 0,
                    "filters": [],
                }

            if len(user):
                total_num_users_converted += 1

            for destination, filter in user.items():
                # print(destination, filter)
                if get_user_logs:
                    per_user_data["num_unique_destinations_for_whom_converted"] += 1
                    per_user_data["filters"].append(filter)

                if get_destination_logs:
                    per_destination_data = {"destination_id": destination}

                if destination not in destinations_data:
                    destinations_data[destination] = {
                        "destination_id": destination,
                        "set_user_ids_who_converted": set(),
                        "num_user_ids_who_converted": 0,
                        "filters": [],
                    }

                destinations_data[destination]["set_user_ids_who_converted"].add(i)
                destinations_data[destination]["num_user_ids_who_converted"] += 1
                destinations_data[destination]["filters"].append(filter)

            if get_user_logs:
                # Process filters for user
                uavg, usum, _ = get_per_epoch_budget_consumption(
                    per_user_data["filters"], initial_budget
                )
                per_user_data[
                    "avg_budget_consumption_per_epoch_across_destinations"
                ] = uavg
                per_user_data[
                    "sum_budget_consumption_per_epoch_across_destinations"
                ] = usum
                per_user_data[
                    "avg_budget_consumption_across_epochs_across_destinations"
                ] = uavg.mean()
                per_user_data[
                    "sum_budget_consumption_across_epochs_across_destinations"
                ] = usum.sum()
                del per_user_data["filters"]
                users_data.append(per_user_data)

        if get_destination_logs:
            # Process filters for destinations
            for _, per_destination_data in destinations_data.items():
                davg, dsum, mx_epoch = get_per_epoch_budget_consumption(
                    per_destination_data["filters"], initial_budget
                )
                per_destination_data[
                    "avg_budget_consumption_per_requested_epoch_across_converted_users"
                ] = davg
                per_destination_data[
                    "sum_budget_consumption_per_requested_epoch_across_users"
                ] = dsum
                per_destination_data[
                    "avg_budget_consumption_across_requested_epochs_across_converted_users"
                ] = davg.mean()
                per_destination_data[
                    "sum_budget_consumption_across_epochs_across_users"
                ] = dsum.sum()
                per_destination_data["max_epoch_requested_across_users"] = mx_epoch
                del per_destination_data["filters"]

                per_destination_data["num_unique_user_ids_who_converted"] = len(
                    per_destination_data["set_user_ids_who_converted"]
                )
                del per_destination_data["set_user_ids_who_converted"]

            destinations_data = list(destinations_data.values())

        users_data_df = pd.DataFrame.from_records(users_data)
        destinations_data_df = pd.DataFrame.from_records(destinations_data)

        experiments[(optimization, num_days_per_epoch)] = {
            "users": users_data_df,
            "destinations": destinations_data_df,
            "total_users_converted_across_destinations": total_num_users_converted,
        }

    processes = []
    manager = Manager()
    experiments = manager.dict()

    for i, row in df.iterrows():
        processes.append(
            Process(
                target=process_experiment_logs,
                args=(row, experiments, get_user_logs, get_destination_logs),
            )
        )
        processes[i].start()

    for process in processes:
        process.join()

    users_data_dfs = []
    destinations_data_dfs = []

    for key, result in experiments.items():
        (optimization, num_days_per_epoch) = key
        result["users"]["optimization"] = optimization
        result["users"]["num_days_per_epoch"] = num_days_per_epoch
        users_data_dfs.append(result["users"])
        result["destinations"]["optimization"] = optimization
        result["destinations"]["num_days_per_epoch"] = num_days_per_epoch
        destinations_data_dfs.append(result["destinations"])

        total_users_converted_across_destinations = result[
            "total_users_converted_across_destinations"
        ]
        # print(total_users_converted_across_destinations)

    custom_order = []

    users_data_df = pd.concat(users_data_dfs)
    users_data_df["optimization"] = pd.Categorical(
        users_data_df["optimization"], categories=custom_order, ordered=True
    )
    users_data_df = users_data_df.sort_values(["optimization"])

    destinations_data_df = pd.concat(destinations_data_dfs)
    destinations_data_df["optimization"] = pd.Categorical(
        destinations_data_df["optimization"], categories=custom_order, ordered=True
    )
    destinations_data_df["destination_id"] = destinations_data_df[
        "destination_id"
    ].astype(int)

    destinations_data_df = destinations_data_df.sort_values(
        ["destination_id", "optimization", "num_days_per_epoch"]
    )

    return (
        users_data_df,
        destinations_data_df,
        total_users_converted_across_destinations,
    )


def plot_budget_consumption(df):

    def plot_num_conversions_per_destination(df):
        fig = px.bar(
            df.query("optimization=='no_optimization' and num_days_per_epoch==1"),
            x="destination_id",
            y="num_user_ids_who_converted",
            title=f"Number of conversions per destination",
            width=1100,
            height=600,
        )
        return fig

    def plot_avg_budget_consumption_per_destination_across_epochs_across_users(df):
        fig = px.bar(
            df,
            x="destination_id",
            y="avg_budget_consumption_across_requested_epochs_across_converted_users",
            color="optimization",
            barmode="group",
            title=f"Avg. Budget Consumption per destination across requested epochs across converted users",
            width=1100,
            height=1600,
            facet_row="num_days_per_epoch",
        )
        fig.update_yaxes(title_text="avg budget")
        fig.update_layout(
            margin=dict(b=300),
        )
        #     fig.update_layout(
        #     annotations=[
        #         dict(
        #             text="• For num-days-per-epoch=1 we have 90 epochs and each time we request 30 epochs. <br>The no-optimization and monoepoch optimization behave similarly because reports request more than 1 epoch so monoepoch cannot thrive. <br>Monoepoch is slightly better  because reports in day 1 (epoch 1) necessarily request for only 1 epoch instead of 30. <br>Multiepoch performs obviously the best because instead of paying for 30 epochs we pay for just 1 - the dataset is extremely sparse <br>and in most cases we find relevant impressions in just one epoch. <br>•  For num-days-per-epoch=15 we have 6 epochs and each time we  request 2 epochs. <br>Yet again monoepoch doesn't thrive because most requests are for >1 epoch. <br>However, it is still slightly better than no-optimization because reports in epoch 1 necessarily request for only 1 epoch instead of 2. <br>Multiepoch is obviously still better because it pays for one epoch instead of 2. <br>However, the gap is smaller since even the other baselines do not mpay as much now. <br>• For num-days-per-epoch=30 we have 3 epochs and each time we request 1. <br>Monoepoch and multiepoch are identical. They are better than no optimization since they pay bsaed on their individual sensitivity rather than the global one.",
        #             # showarrow=True,
        #             # arrowsize=40,
        #             x = 0,
        #             y = -0.25,
        #             xref='paper',
        #             yref='paper',
        #             xanchor='left',
        #             yanchor='bottom',
        #             font=dict(size=13, color="black"),
        #             align="left",
        #             textangle=0,
        #             width=1000,
        #             height=200,
        #         )
        #     ]
        # )
        return fig

    def plot_avg_budget_consumption_across_destinations(df):
        # Average across destinations
        dff = (
            df.groupby(["optimization", "num_days_per_epoch"], observed=False)[
                "sum_budget_consumption_across_epochs_across_users"
            ]
            .mean()
            .reset_index(name="mean")
        )

        fig = px.bar(
            dff,
            x="optimization",
            y="mean",
            title=f"Avg. Budget Consumption across destinations",
            width=1100,
            height=1000,
            facet_row="num_days_per_epoch",
        )
        return fig

    def plot_total_budget_consumption_per_epoch_across_destinations(df):
        def align_and_sum(group):
            if not group.empty:
                max_shape = max(arr.shape for arr in group)
                padded_arrays = [
                    np.pad(arr, (0, max_shape[0] - arr.shape[0]), mode="constant")
                    for arr in group
                ]
                summed_array = np.sum(padded_arrays, axis=0)
                return summed_array

        def explode_within_group(group):
            return group.explode(
                "sum_budget_consumption_per_requested_epoch_across_users"
            )

        g = df.groupby(["optimization", "num_days_per_epoch"], observed=False)
        g = g["sum_budget_consumption_per_requested_epoch_across_users"].apply(
            align_and_sum
        )
        g = g.reset_index(
            name="sum_budget_consumption_per_requested_epoch_across_users"
        )
        g = (
            g.groupby(["optimization", "num_days_per_epoch"], observed=False)[
                "sum_budget_consumption_per_requested_epoch_across_users"
            ]
            .apply(explode_within_group)
            .reset_index()
        )
        g["sum_budget_consumption_per_requested_epoch_across_users"] = g[
            "sum_budget_consumption_per_requested_epoch_across_users"
        ].astype(float)

        fig = px.bar(
            g,
            x="level_2",
            y="sum_budget_consumption_per_requested_epoch_across_users",
            color="optimization",
            barmode="group",
            title=f"Total. Budget Consumption per requested epoch across destinations",
            width=1100,
            height=1600,
            facet_row="num_days_per_epoch",
        )
        return fig

    def plot_total_budget_consumption_per_destination_across_epochs_across_users(df):
        fig = px.bar(
            df,
            x="destination_id",
            y="sum_budget_consumption_across_epochs_across_users",
            color="optimization",
            barmode="group",
            title=f"Total Budget Consumption per destination across users across epochs",
            width=1100,
            height=1600,
            facet_row="num_days_per_epoch",
        )
        return fig

    def plot_total_budget_consumption_across_destinations(df):
        dff = (
            df.groupby(["optimization", "num_days_per_epoch"], observed=False)[
                "sum_budget_consumption_across_epochs_across_users"
            ]
            .sum()
            .reset_index(name="sum")
        )
        fig = px.bar(
            dff,
            x="optimization",
            y="sum",
            title=f"Total Budget Consumption across destinations across users across epochs",
            width=1100,
            height=1600,
            facet_row="num_days_per_epoch",
        )
        return fig

    iplot(plot_num_conversions_per_destination(df))
    iplot(plot_avg_budget_consumption_per_destination_across_epochs_across_users(df))
    iplot(plot_avg_budget_consumption_across_destinations(df))
    iplot(plot_total_budget_consumption_per_epoch_across_destinations(df))
    iplot(plot_total_budget_consumption_per_destination_across_epochs_across_users(df))
    iplot(plot_total_budget_consumption_across_destinations(df))


# max_epoch_requested_across_destinations_across_users = destinations_data_df[
#     "max_epoch_requested_across_users"
# ].max()

# df = destinations_data_df[["optimization"]]
# df[
#     "avg_budget_across_destinations_across_epochs_across_users"
# ] = destinations_data_df[
#     "sum_budget_consumption_across_requested_epochs_across_converted_users"
# ] / (
#     total_users_converted_across_destinations
#     * max_epoch_requested_across_destinations_across_users
# )
