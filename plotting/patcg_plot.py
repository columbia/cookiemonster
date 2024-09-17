import math

import pandas as pd

from plotting.macros import *
from plotting.plot_template import *


def patcg_plot_experiments_side_by_side(
    path,
    output_path,
    variable="num_days_per_epoch",
    value=7,
    x_axis_title=NUM_DAYS_PER_EPOCH_X,
):

    df = pd.read_csv(f"{path}/rmsres.csv")
    df = df.query(f"{variable} in [1 , 7, 14, 21, 30]")
    # df = df[df[variable] != 28]
    # df = df[df["baseline"] != "ipa"]

    # RMSRE BOXES
    args1 = {
        "df": df,
        "metric": "queries_rmsres",
        "x_axis": variable,
        "x_axis_title": x_axis_title,
        "y_axis_title": RMSRE_Y,
        "ordering": (variable, "str"),
        "log_y": False,
        "showlegend": False,
        "show_nqueries": True,
        "vspace": 0.02,
    }

    df = pd.read_csv(f"{path}/rmsres.csv")

    # RMSRE CDF
    df = df.query(f"{variable} == {value}")

    args2 = {
        "df": df,
        "metric": "queries_rmsres",
        "x_axis": None,
        "x_axis_title": RMSRE_CDF_X,
        "y_axis_title": RMSRE_Y,
        "ordering": None,
        "log_y": False,
        "x_range": [1, 100],
        "showlegend": True,
        "marker_pos": 0.98,
    }

    # AVG BUDGET CONSUMPTION LINES
    df = pd.read_csv(f"{path}/budgets.csv")
    df1 = df.query(f"{variable} == {value}")
    df_rmsre = pd.read_csv(f"{path}/rmsres.csv").query(
        f"{variable} == {value} and baseline == 'ipa'"
    )

    print(
        f"Avg budget consumption plot: IPA executed {len(df_rmsre.queries_rmsres.dropna())} queries"
    )

    args3 = {
        "df": df1,
        "metric": "avg",
        "x_axis": "index",
        "x_axis_title": BUDGET_CONSUMPTION_X,
        "y_axis_title": BUDGET_CONSUMPTION_Y_AVG,
        "ordering": None,
        "log_y": False,
        "x_range": [0, 80],
        "showlegend": False,
        "marker_pos": 0.98,
        "mode": "markers",
    }

    # # MAX BUDGET CONSUMPTION BARS
    # last_query_ran = df["index"].max()
    # df = df[df[variable] != 28]
    # df = df.query("index == @last_query_ran")
    # args4 = {
    #     "df": df,
    #     "metric": "max_max",
    #     "x_axis": variable,
    #     "x_axis_title": x_axis_title,
    #     "y_axis_title": BUDGET_CONSUMPTION_Y_MAX,
    #     "ordering": (variable, "str"),
    #     "log_y": False,
    #     "showlegend": False,
    # }

    # figs = [(boxes, args1), (bars, args4), (cdf, args2), (lines, args3)]
    figs = [(lines, args3), (cdf, args2), (boxes, args1)]
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

    make_plots(figs, cols=3, **figs_args)
