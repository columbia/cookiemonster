import re
import math
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.offline import iplot
from cookiemonster.utils import LOGS_PATH
from experiments.ray.analysis import load_ray_experiment

pd.set_option("future.no_silent_downcasting", True)

CUSTOM_ORDER_BASELINES = ["ipa", "cookiemonster_base", "cookiemonster"]
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


def save_df(df, path, filename):
    save_dir = LOGS_PATH.joinpath(path, filename)
    df.to_csv(save_dir, header=True, index=False)


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
        df = df.sort_values(
            ["num_days_per_epoch", "num_days_attribution_window", "initial_budget"]
        )
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


def get_filters_state_logs(row):
    filters_state = row["event_logs"]["filters_state"]

    df = pd.DataFrame.from_records(
        filters_state,
        columns=["filters_state"],
    )

    filters_state_keys = set().union(*df["filters_state"])
    for key in filters_state_keys:
        df[key] = df["filters_state"].apply(lambda x: x.get(key))

    df = df.drop(columns=["filters_state"], axis=1).reset_index()
    df = pd.melt(
        df, id_vars=["index"], var_name="destination", value_name="budget_consumption"
    ).reset_index(drop=True)
    df = df.drop(columns=["index"], axis=1)
    df = df.astype({"destination": "str"})
    return df

def plot_budget_consumption_cdf(
    path,
    workload_size=0,
    epoch_size=0,
    by_destination=False,
    category_orders={},
):

    category_orders = {
        "baseline": CUSTOM_ORDER_BASELINES,
    }

    def ecdf(df):
        fig = px.ecdf(
            df,
            y="budget_consumption",
            color="baseline",
            title=f"CDF for epoch-devices budget consumption {focus})",
            width=1100,
            height=600,
            orientation="h",
            facet_col="destination" if by_destination else None,
            category_orders={
                "baseline": CUSTOM_ORDER_BASELINES,
                **category_orders,
            },
        )
        return fig

    df = analyze_results(path, "filters_state")

    # Pick a subset of the experiments
    focus = ""
    if workload_size:
        focus = f"workload size {workload_size}"
        df = df.query("requested_workload_size == @workload_size")
    if epoch_size:
        focus = f"epoch size {epoch_size}"
        df = df.query("num_days_per_epoch == @epoch_size")

    df = df.explode("budget_consumption")

    save_df(df, path, "budget_consumption_cdf.csv")
    iplot(ecdf(df))

def plot_budget_consumption_boxes(
    path,
    x_axis="num_days_per_epoch",
    by_destination=True,
    log_y=True,
    category_orders={},
):

    category_orders = {
        "baseline": CUSTOM_ORDER_BASELINES,
    }

    def budget_consumption(df):
        fig = px.box(
            df,
            x=x_axis,
            y="budget_consumption",
            color="baseline",
            title=f"Budget Consumption",
            width=1100,
            height=600,
            log_y=log_y,
            facet_col="destination" if by_destination else None,
            category_orders={
                "baseline": CUSTOM_ORDER_BASELINES,
                **category_orders,
            },
        )
        return fig

    df = analyze_results(path, "filters_state")

    df = df.explode("budget_consumption")

    save_df(df, path, "budget_consumption_boxes.csv")
    iplot(budget_consumption(df))


def plot_budget_consumption_lines(path, facet_row=None, height=600):
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

    df = analyze_results(path, "budget")

    kwargs = {
        "data_frame": df,
        "x": df.index,
        "title": f"Cumulative Budget Consumption",
        "color": "baseline",
        "width": 1100,
        "height": height,
        "markers": True,
        "log_y": False,
        "range_y": [0, 1],
        "facet_col": "destination",
        "facet_row": facet_row,
        "category_orders": category_orders,
    }

    save_df(df, path, "budget_lines.csv")
    iplot(px.line(y="max_max", **kwargs))
    # iplot(px.line(y="max_of_avg", **kwargs))
    # iplot(px.line(y="avg_of_max", **kwargs))
    iplot(px.line(y="avg", **kwargs))


def plot_budget_consumption_bars(path, x_axis="knob1", log_y=False, height=400):
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

    df = analyze_results(path, "budget")
    df = df.query("index == @df.index.max()")
    kwargs = {
        "data_frame": df,
        "x": x_axis,
        "title": f"Cumulative Budget Consumption",
        "color": "baseline",
        "width": 1100,
        "height": height,
        "log_y": log_y,
        "barmode": "group",
        "facet_col": "destination",
        "category_orders": category_orders,
    }

    save_df(df, path, "budget_bars.csv")
    iplot(px.bar(y="max_max", **kwargs))
    # iplot(px.bar(y="max_of_avg", **kwargs))
    # iplot(px.bar(y="avg_of_max", **kwargs))
    iplot(px.bar(y="avg", **kwargs))


def plot_rmsre_boxes(
    path,
    x_axis="num_days_per_epoch",
    by_destination=True,
    log_y=True,
    category_orders={},
):
    def rmsre(df):
        fig = px.box(
            df,
            x=x_axis,
            y="queries_rmsres",
            color="baseline",
            title=f"RMSRE",
            width=1100,
            height=600,
            log_y=log_y,
            facet_col="destination" if by_destination else None,
            category_orders={
                "baseline": CUSTOM_ORDER_BASELINES,
                **category_orders,
            },
        )
        return fig

    df = analyze_results(path, "bias")
    df = df.explode("queries_rmsres")
    max_ = df["queries_rmsres"].max() * 2
    df.fillna({"queries_rmsres": max_}, inplace=True)

    save_df(df, path, "rmsre_boxes.csv")
    iplot(rmsre(df))


def plot_bias_rmsre_cdf(
    path,
    workload_size=0,
    epoch_size=0,
    by_destination=True,
    log_y=False,
    category_orders={},
):
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

    df = analyze_results(path, "bias")

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

    save_df(df, path, "rmsre_cdf.csv")
    iplot(ecdf(df))


if __name__ == "__main__":
    path = "ray/microbenchmark/varying_knob1"
    df = analyze_results(path, type="budget")
