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


def analyze_budget_consumption(path):
    df = get_df(path)

    def get_logs(row):
        logs = []
        for destination, destination_logs in row["destination_logs"].items():
            cumulative_budget_consumed = 0
            for i, log in enumerate(destination_logs):
                cumulative_budget_consumed += log["total_budget_consumed"]
                logs.append(
                    {
                        "destination_id": destination,
                        "conversion_timestamp": i,
                        "total_budget_consumed": log["total_budget_consumed"],
                        "cumulative_budget_consumed": cumulative_budget_consumed,
                        "user_id": log["user_id"],
                        "epoch_window": log["epoch_window"],
                        "status": log["status"],
                        "baseline": row["baseline"],
                        "optimization": row["optimization"],
                        "num_days_per_epoch": row["num_days_per_epoch"],
                        "num_days_attribution_window": row[
                            "num_days_attribution_window"
                        ],
                    }
                )
        return pd.DataFrame.from_records(logs)

    dfs = []
    for _, row in df.iterrows():
        dfs.append(get_logs(row))

    return pd.concat(dfs)


def plot_budget_consumption(df):
    df["key"] = (
        df["baseline"] + "-days_per_epoch=" + df["num_days_per_epoch"].astype(str)
    )

    def plot_budget_consumption_across_time(df):
        fig = px.line(
            df,
            x="conversion_timestamp",
            y="cumulative_budget_consumed",
            color="key",
            title=f"Total Budget Consumption",
            width=1100,
            height=1600,
            # facet_row="num_days_per_epoch",
        )
        return fig

    figures = (
        df.groupby("destination_id")
        .apply(plot_budget_consumption_across_time, include_groups=False)
        .reset_index(name="figures")["figures"]
    )

    # for figure in figures.values:
    # iplot(figure)
    iplot(figures.values[0])
