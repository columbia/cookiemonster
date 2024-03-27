import re
import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.express as px
from plotly.offline import iplot
from systemx.utils import LOGS_PATH
from multiprocessing import Manager, Process
from experiments.ray.analysis import load_ray_experiment


def get_df(path):
    logs = LOGS_PATH.joinpath(path)
    df = load_ray_experiment(logs)
    return df


def analyze_budget_consumption(path):

    def get_logs(row, results, i):
        logs = row["logs"]["budget"]

        pattern = r"_conv_rate_([0-9.]+)_impr_rate_([0-9.]+)\.csv"
        match = re.search(pattern, row["config"]["dataset"]["impressions_path"])
        if match:
            conv_rate = match.group(1)
            impr_rate = match.group(2)

        per_destination_epoch_range = row["logs"]["epoch_range"]

        df = pd.DataFrame.from_records(
            logs,
            columns=[
                "timestamp",
                "destination",
                "user",
                "epochs_window",
                "attribution_window",
                "budget_consumed",
                "status",
            ],
        )

        records = []
        for destination, destination_df in df.groupby(["destination"]):

            num_touched_users = destination_df["user"].nunique()
            max_user_id = destination_df["user"].max()+1
            epoch_range = per_destination_epoch_range[str(destination[0])]
            min_epoch = epoch_range["min"]
            max_epoch = epoch_range["max"]
            num_epochs = max_epoch - min_epoch + 1

            print(i, "Touched Users:", num_touched_users, "Total users:", max_user_id, "Min epoch:", min_epoch,"Max epoch:", max_epoch)

            # Array bigger than the size of touched users but I will ignore them
            users = np.zeros(max_user_id)

            for _, log in destination_df.iterrows():
                users[log["user"]] += log["budget_consumed"]

                records.append(
                    {
                        "destination": destination[0],
                        "num_reports": log["timestamp"],
                        "max_budget_conusmed": np.max(users) / num_epochs,
                        "avg_budget_consumed": np.sum(users) / (num_epochs * num_touched_users),
                        "status": log["status"],
                        "baseline": row["baseline"],
                        "optimization": row["optimization"],
                        "num_days_per_epoch": row["num_days_per_epoch"],
                        "num_days_attribution_window": row[
                            "num_days_attribution_window"
                        ],
                        "converstion_rate": conv_rate,
                        "impression_rate": impr_rate,
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
    #     if i == 1:
    #         get_logs(row, results, i)

    for result in results.values():
        dfs.append(result)

    return pd.concat(dfs)


def analyze_bias(path):

    def get_logs(row):
        logs = row["logs"]["bias"]
        df = pd.DataFrame.from_records(
            logs, columns=["timestamp", "destination", "query_id", "bias"]
        )

        records = []
        for destination, destination_df in df.groupby(["destination"]):

            workload_bias = destination_df["bias"].sum()
            # print(log)
            records.append(
                {
                    "destination": destination[0],
                    "workload_bias": workload_bias,
                    "workload_size": row["workload_size"],
                    "baseline": row["baseline"],
                    "optimization": row["optimization"],
                    "num_days_per_epoch": row["num_days_per_epoch"],
                    "num_days_attribution_window": row["num_days_attribution_window"],
                }
            )
        return pd.DataFrame.from_records(records)

    dfs = []
    df = get_df(path)
    for _, row in df.iterrows():
        dfs.append(get_logs(row))

    return pd.concat(dfs)


def plot_budget_consumption(df):
    # df["key"] = (
    #     df["baseline"] + "-days_per_epoch=" + df["num_days_per_epoch"].astype(str)
    # )

    custom_order = ["ipa", "user_epoch_ara", "systemx"]

    def max_budget(df):
        fig = px.line(
            df,
            x="num_reports",
            y="max_budget_conusmed",
            color="baseline",
            title=f"Cumulative Budget Consumption",
            width=1100,
            height=600,
            # markers=True,
            facet_col="destination",
            facet_row="converstion_rate",
            category_orders={"baseline": custom_order},
        )
        return fig

    def avg_budget(df):
        fig = px.line(
            df,
            x="num_reports",
            y="avg_budget_consumed",
            color="baseline",
            title=f"Cumulative Budget Consumption",
            width=1100,
            height=600,
            # markers=True,
            facet_col="destination",
            facet_row="converstion_rate",
            category_orders={"baseline": custom_order},
        )
        return fig

    # figures = (
    #     df.groupby("destination_id")
    #     .apply(plot_budget_consumption_across_time, include_groups=False)
    #     .reset_index(name="figures")["figures"]
    # )
    # iplot(figures.values[0])

    pio.show(max_budget(df), renderer="png", include_plotlyjs=False)
    pio.show(avg_budget(df), renderer="png", include_plotlyjs=False)

    # max_budget(df).show()
    # avg_budget(df).show()
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
