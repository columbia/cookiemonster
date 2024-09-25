import pandas as pd

from plotting.macros import *
from plotting.plot_template import *


def criteo_plot_experiments_side_by_side(
    unaugmented_results,
    augmented_impressions_results,
    output_path,
    variable,
    value,
    x_axis_title,
):
    df = pd.read_csv(f"{unaugmented_results}/rmsres.csv")
    # df = df[df[variable] != 90]
    df = df.query(f"{variable} in [1, 14, 30, 60]")
    # non_ipa_df = df[df["baseline"] != "ipa"]

    args1 = {
        "df": df,
        "metric": "queries_rmsres",
        "x_axis": variable,
        "x_axis_title": x_axis_title,
        "y_axis_title": RMSRE_Y,
        "ordering": (variable, "str"),
        "log_y": False,
        "showlegend": False,
        "show_nqueries": "percentage",
        "vspace": 0.015,
        "hspace": 9,
    }

    # df = df.assign(
    #     queries_rmsres=df["queries_rmsres"].apply(
    #         lambda x: x if x < 0.4751914699759363 else None
    #     )
    # )

    args2 = {
        "df": df,
        "metric": "queries_rmsres",
        "x_axis": variable,
        "x_axis_title": RMSRE_CDF_X,
        "y_axis_title": RMSRE_Y,
        "ordering": (variable, "str"),
        "log_y": False,
        "x_range": [1, 100],
        "showlegend": False,
        "marker_pos": 0.98,
    }

    # df = pd.read_csv(
    #     f"{unaugmented_results}/filters_state.csv.tar.gz", compression="gzip"
    # )
    df = pd.read_csv(f"{unaugmented_results}/filters_state.csv")
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
        "unit": "devices",
    }

    args4 = get_aug_impressions_args(
        unaugmented_results, augmented_impressions_results, variable, value
    )

    # figs = [
    #     (boxes, args1),
    #     (cdf, args2),
    #     (cdf, args3),
    #     (augmented_impressions_cdf, args4),
    # ]

    figs = [
        (cdf, args3),
        (cdf, args2),
        (boxes, args1),
        (augmented_impressions_cdf, args4),
    ]

    figs_args = {
        "axis_title_font_size": {"x": 18, "y": 18},
        "axis_tick_font_size": {"x": 14, "y": 14},
        "legend": {
            "entrywidthmode": "fraction",
            "entrywidth": 0.5,
            "yanchor": "top",
            "y": 2,
            "xanchor": "left",
            "x": 0,
            "orientation": "h",
            "font": {"size": 18},
        },
        # "output_path": f"{unaugmented_results}/../all_experiments_side_by_side.pdf",
        "output_path": output_path,
        "height": 300,
        "width": 1200,
    }

    make_plots(figs, cols=len(figs), **figs_args)


def get_aug_impressions_args(path1, path2, variable, value):
    # odf = pd.read_csv(f"{path1}/filters_state.csv.tar.gz", compression="gzip")
    odf = pd.read_csv(f"{path1}/filters_state.csv")
    odf = odf.query(f"{variable} == {value}")
    odf = odf.drop(columns=["initial_budget"])

    odf = odf.assign(
        baseline=odf.apply(
            lambda r: r.baseline + "_0",
            axis=1,
        )
    )

    # df = pd.read_csv(f"{path2}/filters_state.csv.tar.gz", compression="gzip")
    df = pd.read_csv(f"{path2}/filters_state.csv")
    df = df.query(f"{variable} == {value}")
    df = df.drop(columns=["initial_budget"])

    df = pd.concat([odf, df])
    df = df.rename(columns={"filters_state.csv": "destination"})

    args = {
        "df": df,
        "metric": "budget_consumption",
        "x_axis": None,
        "x_axis_title": BUDGET_CDF_X,
        "y_axis_title": BUDGET_CDF_Y,
        "ordering": None,
        "log_y": False,
        "x_range": [1, 100],
        "showlegend": True,
    }

    return args


def criteo_plot_experiments_side_by_side_submission(
    path1,
    path2,
    output_path,
    variable="num_days_per_epoch",
    value=7,
    x_axis_title=NUM_DAYS_PER_EPOCH_X,
):
    df = pd.read_csv(f"{path1}/rmsres.csv")

    # print(df["queries_rmsres"].max())
    # df["queries_rmsres"] = df["queries_rmsres"].apply(lambda x: x if x < 0.4751914699759363 else None)
    # df.to_csv(f"{path1}/rmsres.csv", index=False)
    df = df[df[variable] != 90]
    # df = df[df["baseline"] != "ipa"]

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
