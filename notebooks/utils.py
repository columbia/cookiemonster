import re
import math
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.offline import iplot
from cookiemonster.utils import LOGS_PATH
from multiprocessing import Manager, Process
from experiments.ray.analysis import load_ray_experiment

pd.set_option("future.no_silent_downcasting", True)

CUSTOM_ORDER_BASELINES = ["ipa", "user_epoch_ara", "cookiemonster"]
CUSTOM_ORDER_RATES = ["0.001", "0.01", "0.1", "1.0"]


class Bias:
    def __init__(self) -> None:
        self.values = []
        self.count = 0
        self.undefined_errors_counter = 0


def get_df(path):
    logs = LOGS_PATH.joinpath(path)
    df = load_ray_experiment(logs)
    return df


def analyze_results(path, type="budget"):
    df = get_df(path)

    match type:
        case "budget":
            get_logs = get_budget_logs
        case "bias":
            get_logs = get_bias_logs
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
        df = df.sort_values(["num_days_per_epoch", "num_days_attribution_window", "initial_budget"])
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
    df["knob1"] = knob1
    df["knob2"] = knob2
    return df


def get_bias_logs(row):
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

    result_df = (
        df.groupby("destination").apply(compute_bias_stats).reset_index(drop=True)
    )
    return result_df


def plot_budget_consumption_lines(df, facet_row=None, height=600, log_y=False):
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


    kwargs = {
        "data_frame": df,
        "x": df.index,
        "title": f"Cumulative Budget Consumption",
        "color": "baseline",
        "width": 1100,
        "height": height,
        "markers": True,
        "log_y": log_y,
        "range_y": [0, 1],
        "facet_col": "destination",
        "facet_row": facet_row,
        "category_orders": category_orders,
    }

    iplot(px.line(y="max_max", **kwargs))
    iplot(px.line(y="max_of_avg", **kwargs))
    iplot(px.line(y="avg_of_max", **kwargs))
    iplot(px.line(y="avg", **kwargs))


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


    kwargs = {
        "data_frame": df.query("index == @df.index.max()"),
        "x": x_axis,
        "title": f"Cumulative Budget Consumption",
        "color": "baseline",
        "width": 1100,
        "height": 400,
        "log_y": True,
        "barmode": "group",
        "facet_col": "destination",
        # "facet_row": facet_row,
        "category_orders": category_orders,
    }

    iplot(px.bar(y="max_max", **kwargs))
    iplot(px.bar(y="max_of_avg", **kwargs))
    iplot(px.bar(y="avg_of_max", **kwargs))
    iplot(px.bar(y="avg", **kwargs))


def plot_bias_rmsre(
    df, x_axis="num_days_per_epoch", by_destination=True, log_y=True, category_orders={}
):
    # Set values for the potentially undefined errors of baselines
    df = df.explode("queries_rmsres")
    max_ = df["queries_rmsres"].max() * 2
    df.fillna({"queries_rmsres": max_}, inplace=True)

    # Compute the average RMSRE per grouping key
    group_key = [x_axis, "baseline"]
    group_key += ["destination"] if by_destination else []

    # df = df.groupby(group_key)['queries_rmsres'].agg(['mean', 'std']).reset_index()
    # df = df.rename(columns={"mean": "avg_rmsre"})
    def rmsre(df):
        fig = px.box(
            df,
            x=x_axis,
            # error_y="std",
            # y="avg_rmsre",
            y="queries_rmsres",
            color="baseline",
            title=f"RMSRE",
            width=1100,
            height=600,
            # barmode="group",
            log_y=log_y,
            facet_col="destination" if by_destination else None,
            category_orders={
                "baseline": CUSTOM_ORDER_BASELINES,
                **category_orders,
            },
        )
        return fig

    iplot(rmsre(df))


def plot_bias_rmsre_cdf(
    df,
    workload_size=0,
    epoch_size=0,
    by_destination=True,
    log_y=False,
    category_orders={},
):

    # Pick a subset of the experiments
    focus = ""
    if workload_size:
        focus = f"workload size {workload_size}"
        df = df.query("requested_workload_size == @workload_size")
    if epoch_size:
        focus = f"epoch size {epoch_size}"
        df = df.query("num_days_per_epoch == @epoch_size")

    df = df.explode("queries_rmsres")
    max_ = df["queries_rmsres"].max() * 2
    df.fillna({"queries_rmsres": max_}, inplace=True)

    def ecdf(df):
        fig = px.ecdf(
            df,
            y="queries_rmsres",
            color="baseline",
            title=f"CDF for E2E RMSRE({focus})",
            width=1100,
            height=600,
            orientation="h",
            log_y=log_y,
            facet_col="destination" if by_destination else None,
            category_orders={
                "baseline": CUSTOM_ORDER_BASELINES,
                **category_orders,
            },
        )
        return fig

    iplot(ecdf(df))


if __name__ == "__main__":
    path = "ray/microbenchmark/varying_knob1"
    df = analyze_results(path, type="budget")
