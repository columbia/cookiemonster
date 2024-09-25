import pandas as pd

from plotting.macros import *
from plotting.plot_template import *


def microbenchmark_plot_budget_consumption_bars(knob, file_path, output_path):
    args = []
    df = pd.read_csv(file_path)
    last_query_ran = df["index"].max()
    df = df.query("index == @last_query_ran")

    # max max metric
    args.append(
        {
            "df": df,
            "metric": "max_max",
            "x_axis": knob,
            "x_axis_title": KNOB1_AXIS if knob == "knob1" else KNOB2_AXIS,
            "y_axis_title": BUDGET_CONSUMPTION_Y_MAX,
            "ordering": (knob, "str"),
            "log_y": False,
            "showlegend": False,
        }
    )

    # avg metric
    args.append(
        {
            "df": df,
            "metric": "avg",
            "x_axis": knob,
            "x_axis_title": KNOB1_AXIS if knob == "knob1" else KNOB2_AXIS,
            "y_axis_title": BUDGET_CONSUMPTION_Y_AVG_LOG,
            "ordering": (knob, "str"),
            "log_y": True,
            "showlegend": False,
        }
    )

    args = [args[1], args[0]]
    args[0]["showlegend"] = True

    figs = [(bars, arg) for arg in args]
    figs_args = {
        "axis_title_font_size": {"x": 18, "y": 18},
        "axis_tick_font_size": {"x": 14, "y": 14},
        "legend": {
            "yanchor": "top",
            "y": 1.2,
            "xanchor": "left",
            "x": 0.2,
            "orientation": "h",
        },
        "output_path": output_path,
        "height": 300,
        "width": 1500,
    }
    make_plots(figs, cols=2, **figs_args)


def microbenchmark_plot_all_budget_consumption_bars(
    file_path_knob_1, file_path_knob_2, output_path
):
    args = []

    # Knob 1
    df = pd.read_csv(file_path_knob_1)
    last_query_ran = df["index"].max()
    df = df.query("index == @last_query_ran")

    # avg metric
    args.append(
        {
            "df": df,
            "metric": "avg",
            "x_axis": "knob1",
            "x_axis_title": KNOB1_AXIS,
            "y_axis_title": BUDGET_CONSUMPTION_Y_AVG_LOG,
            "ordering": ("knob1", "str"),
            "log_y": True,
            "showlegend": False,
        }
    )

    # max max metric
    args.append(
        {
            "df": df,
            "metric": "max_max",
            "x_axis": "knob1",
            "x_axis_title": KNOB1_AXIS,
            "y_axis_title": BUDGET_CONSUMPTION_Y_MAX,
            "ordering": ("knob1", "str"),
            "log_y": False,
            "showlegend": False,
        }
    )

    # Knob 2
    df = pd.read_csv(file_path_knob_2)
    last_query_ran = df["index"].max()
    df = df.query("index == @last_query_ran")

    # avg metric
    args.append(
        {
            "df": df,
            "metric": "avg",
            "x_axis": "knob2",
            "x_axis_title": KNOB2_AXIS,
            "y_axis_title": BUDGET_CONSUMPTION_Y_AVG_LOG,
            "ordering": ("knob2", "str"),
            "log_y": True,
            "showlegend": False,
        }
    )

    # max max metric
    args.append(
        {
            "df": df,
            "metric": "max_max",
            "x_axis": "knob2",
            "x_axis_title": KNOB2_AXIS,
            "y_axis_title": BUDGET_CONSUMPTION_Y_MAX,
            "ordering": ("knob2", "str"),
            "log_y": False,
            "showlegend": False,
        }
    )

    # Shared attributes
    args[0]["showlegend"] = True
    figs = [(bars, arg) for arg in args]
    figs_args = {
        "axis_title_font_size": {"x": 18, "y": 18},
        "axis_tick_font_size": {"x": 14, "y": 14},
        "legend": {
            "yanchor": "top",
            "y": 1.2,
            "xanchor": "left",
            "x": 0.2,
            "orientation": "h",
        },
        "output_path": output_path,
        "height": 300,
        "width": 1500,
    }
    make_plots(figs, cols=4, **figs_args)
