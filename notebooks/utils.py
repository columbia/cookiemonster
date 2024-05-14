import re
import math
from typing import Callable
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.offline import iplot
from cookiemonster.utils import LOGS_PATH
from experiments.ray.analysis import load_ray_experiment

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

pd.set_option("future.no_silent_downcasting", True)

CUSTOM_ORDER_BASELINES = ["ipa", "cookiemonster_base", "cookiemonster"]
CUSTOM_ORDER_RATES = ["0.001", "0.01", "0.1", "1.0"]


class Bias:
    def __init__(self) -> None:
        self.values = []
        self.count = 0
        self.undefined_errors_counter = 0
        self.no_run_counter = 0


def get_df(path):
    logs = LOGS_PATH.joinpath(path)
    df = load_ray_experiment(logs)
    return df


def save_df(df, path, filename, include_index=False):
    save_dir = LOGS_PATH.joinpath(path, filename)
    df.to_csv(save_dir, header=True, index=include_index)


def analyze_results(path, type="budget"):
    df = get_df(path)

    match type:
        case "budget":
            get_logs = get_budget_logs
        case "bias":
            get_logs = get_bias_logs
        case "filters_state":
            get_logs = get_filters_state_logs
        case _:
            raise ValueError(f"Unsupported type: {type}")

    dfs = []
    for _, row in df.iterrows():
        df = get_logs(row)
        config = row["config"]
        df["baseline"] = config["user"]["baseline"]
        df["initial_budget"] = config["user"]["initial_budget"]
        df["workload_size"] = config["dataset"]["workload_size"]
        df["num_days_per_epoch"] = config["dataset"]["num_days_per_epoch"]
        df["num_days_attribution_window"] = config["dataset"][
            "num_days_attribution_window"
        ]
        # df = df.sort_values(
        #     ["num_days_per_epoch", "num_days_attribution_window", "initial_budget"]
        # )
        dfs.append(df)
    return pd.concat(dfs)


def get_budget_logs(row):
    config = row["config"]
    global_statistics = row["global_statistics"]

    # Obtain conversions and impressions rate from dataset paths
    pattern = r"_knob1_([0-9.]+)_knob2_([0-9.]+)\.csv"
    match = re.search(pattern, config["dataset"]["impressions_path"])
    (knob1, knob2) = (match.group(1), match.group(2)) if match else ("", "")

    budget_events = row["event_logs"]["budget"]
    df = pd.DataFrame.from_records(
        budget_events,
        columns=[
            "destination",
            "timestamp",
            "budget_metrics",
        ],
    )

    # Explode metric dict keys to columns
    budget_metrics_keys = set().union(*df["budget_metrics"])
    for key in budget_metrics_keys:
        df[key] = df["budget_metrics"].apply(lambda x: x.get(key))
    df.drop("budget_metrics", axis=1, inplace=True)

    # Finalize metrics by dividing them with the right denominators
    def finalize_metrics(row):
        destination = row["destination"]
        stats = global_statistics[destination]
        num_devices = stats["num_unique_device_filters_touched"]
        num_epochs = stats["num_epochs_touched"]
        row["sum_max"] /= num_devices
        row["max_sum"] /= num_epochs
        row["sum_sum"] /= num_devices * num_epochs
        return row

    df = df.astype({"destination": "str", "timestamp": "int"})
    df = df.apply(finalize_metrics, axis=1)

    df = df.rename(
        columns={"sum_max": "avg_of_max", "max_sum": "max_of_avg", "sum_sum": "avg"}
    )
    df.index.name = "index"

    if knob1:
        df["knob1"] = float(knob1)
        df["knob2"] = float(knob2)
        df.sort_values(["knob1", "knob2"], inplace=True)
        df = df.astype({"knob1": str, "knob2": str})
    return df


def get_bias_logs(row):
    config = row["config"]

    # Obtain conversions and impressions rate from dataset paths
    pattern = r"_knob1_([0-9.]+)_knob2_([0-9.]+)\.csv"
    match = re.search(pattern, config["dataset"]["impressions_path"])
    (knob1, knob2) = (match.group(1), match.group(2)) if match else ("", "")

    bias_events = row["event_logs"]["bias"]
    df = pd.DataFrame.from_records(
        bias_events,
        columns=[
            "timestamp",
            "destination",
            "query_id",
            "epsilon",
            "sensitivity",
            "bias_data",
        ],
    )
    # Explode bias data dict keys to columns
    bias_data_keys = set().union(*df["bias_data"])
    for key in bias_data_keys:
        df[key] = df["bias_data"].apply(lambda x: x.get(key))
    df.drop("bias_data", axis=1, inplace=True)

    # Compute RMSRE for the queries of each destination
    def compute_bias_stats(group):
        queries_rmsre = Bias()

        for _, row in group.iterrows():
            if math.isnan(row.aggregation_output):
                queries_rmsre.undefined_errors_counter += 1
            elif not row.true_output:
                queries_rmsre.no_run_counter += 1
            else:
                x = abs(row.true_output - row.aggregation_output) ** 2 + 2 * (
                    row.sensitivity**2
                ) / (row.epsilon**2)
                y = row.true_output**2
                queries_rmsre.values.append(math.sqrt(x / y))

        queries_rmsre.values += [np.nan] * queries_rmsre.undefined_errors_counter
        return pd.DataFrame(
            {
                "destination": group.name,
                "queries_rmsres": queries_rmsre.values,
            }
        )

    df = df.groupby("destination").apply(compute_bias_stats).reset_index(drop=True)
    df = df.explode("queries_rmsres")
    if knob1:
        df["knob1"] = float(knob1)
        df["knob2"] = float(knob2)
        df.sort_values(["knob1", "knob2"], inplace=True)
        df = df.astype({"knob1": str, "knob2": str})
    return df


def get_filters_state_logs(row):
    config = row["config"]
    filters_state = row["event_logs"]["filters_state"]
    # Obtain conversions and impressions rate from dataset paths
    pattern = r"_knob1_([0-9.]+)_knob2_([0-9.]+)\.csv"
    match = re.search(pattern, config["dataset"]["impressions_path"])
    (knob1, knob2) = (match.group(1), match.group(2)) if match else ("", "")

    df = pd.DataFrame.from_records(
        filters_state,
        columns=["filters_state"],
    )

    filters_state_keys = set().union(*df["filters_state"])
    filters_state_values = {}
    for key in filters_state_keys:
        filters_state_values[key] = df["filters_state"].apply(lambda x: x.get(key))

    df = pd.concat([df, pd.DataFrame(filters_state_values)], axis=1)

    df = df.drop(columns=["filters_state"], axis=1).reset_index()
    df = pd.melt(
        df, id_vars=["index"], var_name="destination", value_name="budget_consumption"
    ).reset_index(drop=True)
    df = df.drop(columns=["index"], axis=1)
    df = df.astype({"destination": "str"})
    df = df.explode("budget_consumption")

    if knob1:
        df["knob1"] = float(knob1)
        df["knob2"] = float(knob2)
        df.sort_values(["knob1", "knob2"], inplace=True)
        df = df.astype({"knob1": str, "knob2": str})
    return df


def save_data(path):

    # Filters state
    df = analyze_results(path, "filters_state")
    df = df.drop(columns=["workload_size", "num_days_attribution_window"], axis=1)
    df = df.explode("budget_consumption")
    save_df(df, path, "filters_state.csv")

    # Budget Consumption
    df = analyze_results(path, "budget")
    save_df(df, path, "budgets.csv", include_index=True)

    # Bias
    df = analyze_results(path, "bias")
    # max_ = df["queries_rmsres"].max() * 2
    # df.fillna({"queries_rmsres": max_}, inplace=True)
    save_df(df, path, "rmsres.csv")


def focus(
    df, workload_size, epoch_size, knob1, knob2, attribution_window, initial_budget=0
):
    # Pick a subset of the experiments
    focus = ""
    if workload_size:
        focus = f"workload size {workload_size}"
        df = df.query("requested_workload_size == @workload_size")
    if epoch_size:
        focus = f"epoch size {epoch_size}"
        df = df.query("num_days_per_epoch == @epoch_size")
    if knob1:
        focus = f"knob1 {knob1}"
        df = df.query("knob1 == @knob1")
    if knob2:
        focus = f"knob2 {knob2}"
        df = df.query("knob2 == @knob2")
    if attribution_window:
        focus = f"attribution_window {attribution_window}"
        df = df.query("num_days_attribution_window == @attribution_window")
    if initial_budget:
        focus = f"initial_budget {initial_budget}"
        df = df.query("initial_budget == @initial_budget")
    return df, focus


def plot_budget_consumption_cdf(
    path,
    knob1=0,
    knob2=0,
    epoch_size=0,
    workload_size=0,
    attribution_window=0,
    by_destination=False,
    facet_row=None,
    category_orders={},
    log_y=False,
):
    df = analyze_results(path, "filters_state")
    df, focus_ = focus(df, workload_size, epoch_size, knob1, knob2, attribution_window)
    # df = df.explode("budget_consumption")

    fig = px.ecdf(
        df,
        y="budget_consumption",
        color="baseline",
        title=f"CDF for epoch-devices budget consumption {focus_})",
        width=1100,
        height=600,
        orientation="h",
        log_y=log_y,
        facet_row=facet_row,
        facet_col="destination" if by_destination else None,
        category_orders={
            "baseline": CUSTOM_ORDER_BASELINES,
            **category_orders,
        },
    )
    fig.write_image("budget_consumption_cdf.png")
    # iplot(fig)


def plot_budget_consumption_boxes(
    path,
    x_axis="num_days_per_epoch",
    knob1=0,
    knob2=0,
    epoch_size=0,
    workload_size=0,
    attribution_window=0,
    by_destination=False,
    facet_row=None,
    category_orders={},
    log_y=False,
):

    df = analyze_results(path, "filters_state")
    df, focus_ = focus(df, workload_size, epoch_size, knob1, knob2, attribution_window)
    df[x_axis] = df[x_axis].astype(str)
    # df = df.explode("budget_consumption")

    df[x_axis] = df[x_axis].astype(str)
    fig = px.box(
        df,
        x=x_axis,
        y="budget_consumption",
        color="baseline",
        title=f"Budget Consumption {focus_}",
        width=1100,
        height=600,
        log_y=log_y,
        facet_row=facet_row,
        facet_col="destination" if by_destination else None,
        category_orders={
            "baseline": CUSTOM_ORDER_BASELINES,
            **category_orders,
        },
    )

    iplot(fig)


def plot_budget_consumption_lines(
    path,
    knob1=0,
    knob2=0,
    epoch_size=0,
    workload_size=0,
    attribution_window=0,
    by_destination=False,
    facet_row=None,
    category_orders={},
    log_y=False,
    extra_df_prep: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
):

    df = analyze_results(path, "budget")
    df, focus_ = focus(df, workload_size, epoch_size, knob1, knob2, attribution_window)
    if extra_df_prep:
        df = extra_df_prep(df)

    kwargs = {
        "data_frame": df,
        "x": df.index,
        "title": f"Cumulative Budget Consumption {focus_}",
        "color": "baseline",
        "width": 1100,
        "height": 1000,
        "markers": True,
        # "range_y": [0, 1],
        "log_y": log_y,
        "facet_row": facet_row,
        "facet_col": "destination" if by_destination else None,
        "category_orders": {
            "baseline": CUSTOM_ORDER_BASELINES,
            **category_orders,
        },
    }

    iplot(px.line(y="max_max", **kwargs))
    # iplot(px.line(y="max_of_avg", **kwargs))
    # iplot(px.line(y="avg_of_max", **kwargs))
    iplot(px.line(y="avg", **kwargs))


def plot_budget_consumption_bars(
    path,
    x_axis="knob1",
    knob1=0,
    knob2=0,
    epoch_size=0,
    workload_size=0,
    attribution_window=0,
    by_destination=False,
    facet_row=None,
    category_orders={},
    log_y=False,
):

    df = analyze_results(path, "budget")
    df, focus_ = focus(df, workload_size, epoch_size, knob1, knob2, attribution_window)
    df = df.query("index == @df.index.max()")
    kwargs = {
        "data_frame": df,
        "x": x_axis,
        "title": f"Cumulative Budget Consumption {focus_}",
        "color": "baseline",
        "width": 1100,
        "height": 400,
        "barmode": "group",
        "log_y": log_y,
        "facet_row": facet_row,
        "facet_col": "destination" if by_destination else None,
        "category_orders": {
            "baseline": CUSTOM_ORDER_BASELINES,
            **category_orders,
        },
    }

    iplot(px.bar(y="max_max", **kwargs))
    # iplot(px.bar(y="max_of_avg", **kwargs))
    # iplot(px.bar(y="avg_of_max", **kwargs))
    iplot(px.bar(y="avg", **kwargs))


def plot_rmsre_boxes(
    path,
    x_axis="num_days_per_epoch",
    knob1=0,
    knob2=0,
    epoch_size=0,
    workload_size=0,
    attribution_window=0,
    by_destination=False,
    facet_row=None,
    category_orders={},
    log_y=False,
):

    df = analyze_results(path, "bias")
    df, focus_ = focus(df, workload_size, epoch_size, knob1, knob2, attribution_window)
    # df = df.explode("queries_rmsres")
    max_ = df["queries_rmsres"].max() * 2
    df.fillna({"queries_rmsres": max_}, inplace=True)
    df[x_axis] = df[x_axis].astype(str)

    fig = px.box(
        df,
        x=x_axis,
        y="queries_rmsres",
        color="baseline",
        title=f"RMSRE {focus_}",
        width=1100,
        height=600,
        log_y=log_y,
        facet_row=facet_row,
        facet_col="destination" if by_destination else None,
        category_orders={
            "baseline": CUSTOM_ORDER_BASELINES,
            **category_orders,
        },
    )

    iplot(fig)


def plot_rmsre_cdf(
    path,
    knob1=0,
    knob2=0,
    epoch_size=0,
    workload_size=0,
    attribution_window=0,
    initial_budget=0,
    by_destination=False,
    facet_row=None,
    category_orders={},
    log_y=False,
):

    df = analyze_results(path, "bias")
    df, focus_ = focus(
        df, workload_size, epoch_size, knob1, knob2, attribution_window, initial_budget
    )
    # df = df.explode("queries_rmsres")
    max_ = df["queries_rmsres"].max() * 2
    df.fillna({"queries_rmsres": max_}, inplace=True)

    fig = px.ecdf(
        df,
        y="queries_rmsres",
        color="baseline",
        title=f"CDF for E2E RMSRE({focus_})",
        width=1100,
        height=600,
        orientation="h",
        log_y=log_y,
        facet_row=facet_row,
        facet_col="destination" if by_destination else None,
        category_orders={
            "baseline": CUSTOM_ORDER_BASELINES,
            **category_orders,
        },
    )

    iplot(fig)


if __name__ == "__main__":
    save_data("ray/microbenchmark/varying_knob1")
    save_data("ray/microbenchmark/varying_knob2")
    # save_data("ray/patcg/varying_epoch_granularity_aw_7")
    # save_data("ray/patcg/varying_initial_budget")
