import pandas as pd
from plotting.macros import *
from plotting.plot_template import *


def criteo_plot_experiments_side_by_side(path1, path2, output_path, variable="num_days_per_epoch", value=7, x_axis_title=NUM_DAYS_PER_EPOCH_X):
    df = pd.read_csv(f"{path1}/rmsres.csv")

    # print(df["queries_rmsres"].max())
    # df["queries_rmsres"] = df["queries_rmsres"].apply(lambda x: x if x < 0.4751914699759363 else None)
    # df.to_csv(f"{path1}/rmsres.csv", index=False)
    df = df[df[variable] != 90]
    df = df[df["baseline"] != "ipa"]

    args1 = {
        "df": df,
        "metric": "queries_rmsres",
        "x_axis": variable,
        "x_axis_title": x_axis_title,
        "y_axis_title": RMSRE_Y,
        "ordering": (variable, "str"),
        "log_y": False,
        # "y_range": [0, 6],
        "showlegend": False,
    }

    df = pd.read_csv(f"{path1}/rmsres.csv")
    df = df[df[variable] != 90]
    args2 = {
        "df": df,
        "metric": "queries_rmsres",
        "x_axis": variable,
        "x_axis_title": RMSRE_CDF_X,
        "y_axis_title": RMSRE_Y,
        "ordering": (variable, "str"),
        "log_y": False,
        "x_range": [1, 100],
        "showlegend": True,
        "marker_pos": 0.98,
    }

    df = pd.read_csv(f"{path1}/filters_state.csv")
    df = df.query(f"{variable} == {value}")

    args3 = {
        "df": df,
        "metric": "budget_consumption",
        "x_axis": None,
        "x_axis_title": BUDGET_CDF_X,
        "y_axis_title": BUDGET_CDF_Y,
        "ordering": None,
        "log_y": False,
        "x_range": [1, 100],
        "showlegend": False,
    }

    df = pd.read_csv(f"{path2}/filters_state.csv")
    df = df.query(f"{variable} == {value}")

    args4 = {
        "df": df,
        "metric": "budget_consumption",
        "x_axis": None,
        "x_axis_title": BUDGET_CDF_X,
        "y_axis_title": BUDGET_CDF_Y,
        "ordering": None,
        "log_y": False,
        "x_range": [1, 100],
        "showlegend": False,
    }

    figs = [(boxes, args1), (cdf, args2), (cdf, args3), (cdf, args4)]
    figs_args = {
        "axis_title_font_size": {"x": 18, "y": 18},
        "axis_tick_font_size": {"x": 14, "y": 14},
        "legend": {
            "yanchor": "top",
            "y": 1.3,
            "xanchor": "left",
            "x": 0.2,
            "orientation": "h",
        },
        "output_path": output_path,
        "height": 300,
        "width": 1500,
    }

    make_plots(figs, cols=len(figs), **figs_args)
