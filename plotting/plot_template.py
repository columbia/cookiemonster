import math
from plotting.macros import *

import numpy as np
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

pio.kaleido.scope.mathjax = None


def make_plots(
    figs,
    cols,
    legend,
    axis_title_font_size,
    axis_tick_font_size,
    output_path,
    height=None,
    width=None,
    titles=None,
):

    fig = make_subplots(rows=1, cols=cols,
                        horizontal_spacing=0.08, subplot_titles=titles)

    # Create the figures
    traces = []
    for func, args in figs:
        traces.append(func(**args))

    # Add the figures to the subplots
    for i, (func, args) in enumerate(figs):
        for trace in func(**args):
            fig.add_trace(trace, row=1, col=i + 1)

    for i in range(cols):
        fig.update_xaxes(
            title=figs[i][1].get("x_axis_title"),
            tickfont=dict(size=axis_tick_font_size.get("x")),
            title_font=dict(size=axis_title_font_size.get("x")),
            showgrid=True,
            range=figs[i][1].get("x_range"),
            row=1,
            col=i + 1,
        )
        fig.update_yaxes(
            title=figs[i][1].get("y_axis_title"),
            title_standoff=0,
            tickfont=dict(size=axis_tick_font_size.get("y")),
            title_font=dict(size=axis_title_font_size.get("y")),
            type="log" if figs[i][1].get("log_y") else "linear",
            range=figs[i][1].get("y_range"),
            showgrid=True,
            row=1,
            col=i + 1,
        )

    fig.update_layout(
        legend={
            "title": None,
            "font": {"size": 20},
            "traceorder": "reversed",
            **legend,
        },
        template=TEMPLATE,
        showlegend=True,
        height=height,
        width=width,
        barmode="group",
        boxmode="group",
        margin=dict(t=0, b=0),
    )
    # fig.show()
    fig.write_image(f"{output_path}", engine="kaleido")


def bars(
    df,
    metric,
    x_axis,
    ordering,
    showlegend=True,
    **kwargs,
):
    if ordering:
        df = ensure_ordering(df, ordering[0], ordering[1])

    traces = []
    for baseline in baselines_order:
        csv_name = csv_mapping[baseline]
        group = df.query("baseline == @csv_name")
        trace = go.Bar(
            x=group[x_axis],
            y=group[metric],
            legendgroup=baseline,
            name=baseline,
            marker_color=color_discrete_map[baseline],
            showlegend=showlegend,
            marker_line_color="black",
            marker_pattern_shape=pattern_shape_map[baseline],
            opacity=0.8,
        )
        traces.append(trace)
    return traces


def lines(
    df,
    metric,
    x_axis,
    ordering,
    marker_pos=1/2,
    showlegend=True,
    **kwargs,
):
    if ordering:
        df = ensure_ordering(df, ordering[0], ordering[1])

    traces = []
    for baseline in baselines_order:
        csv_name = csv_mapping[baseline]
        group = df.query("baseline == @csv_name")
        if group.empty:
            continue
        traces.append(go.Scatter(
            x=group[x_axis],
            y=group[metric],
            legendgroup=baseline,
            name=baseline,
            marker_color=color_discrete_map[baseline],
            showlegend=True,
            mode="lines",
            line=dict(dash=lines_map[baseline]),
        ))

        # # Add markers
        # sample_group = group.iloc[[int(len(group) * marker_pos)]]
        # traces.append(go.Scatter(
        #     x=sample_group[x_axis],
        #     y=sample_group[metric],
        #     legendgroup=baseline,
        #     name=baseline,
        #     marker_color=color_discrete_map[baseline],
        #     showlegend=showlegend,
        #     marker_symbol=symbol_map[baseline],
        #     mode="markers",
        #     marker=dict(size=12),
        # ))

    return traces


def boxes(
    df,
    metric,
    x_axis,
    ordering,
    showlegend=True,
    **kwargs,
):
    if ordering:
        df = ensure_ordering(df, ordering[0], ordering[1])

    traces = []
    for baseline in baselines_order:
        csv_name = csv_mapping[baseline]
        group = df.query("baseline == @csv_name")
        trace = go.Box(
            x=group[x_axis],
            y=group[metric],
            legendgroup=baseline,
            name=baseline,
            marker_color=color_discrete_map[baseline],
            showlegend=showlegend,
            boxpoints=False,
            # marker_line_color="black",
            opacity=1,
            # fillcolor=color_discrete_map[baseline],
            # marker_pattern_shape=pattern_shape_map[baseline],
        )
        traces.append(trace)
    return traces


def cdf(
    df,
    metric,
    x_axis,
    ordering,
    showlegend=True,
    marker_pos=1/2,
    unit=None,
    **kwargs,
):
    if ordering:
        df = ensure_ordering(df, ordering[0], ordering[1])

    traces = []
    for baseline in baselines_order:
        csv_name = csv_mapping[baseline]
        group = df.query("baseline == @csv_name")
        group = group.sort_values(by=[metric])
        len_values = group.shape[0]
        start = 1

        if group.empty:
            continue

        group.dropna(inplace=True)
        stop = group.shape[0]
        values = np.sort(group[metric].values)
        cumulative_probabilities = np.arange(
            start, stop + 1) / float(len_values) * 100

        traces.append(go.Scatter(
            x=cumulative_probabilities,
            y=values,
            legendgroup=baseline,
            name=f"{baseline} ({len_values} {unit})" if unit else baseline,
            marker_color=color_discrete_map[baseline],
            marker_symbol=symbol_map[baseline],
            showlegend=showlegend,
            mode="lines",
            line=dict(dash=lines_map[baseline]),
        ))

        if stop != len_values:
            # Add X marker to show IPA stopped
            sample_pos = int(len(cumulative_probabilities) * marker_pos)
            sample_x = cumulative_probabilities[-1]
            sample_y = values[-1]

            traces.append(go.Scatter(
                x=[sample_x],
                y=[sample_y],
                legendgroup=baseline,
                name=f"{baseline} ({len_values} {unit})" if unit else baseline,
                marker_color=color_discrete_map[baseline],
                showlegend=False,
                marker_symbol=symbol_map[baseline],
                mode="markers",
                marker=dict(size=11),
            ))
    return traces


def augmented_impressions_cdf(
    df,
    metric,
    x_axis,
    ordering,
    showlegend=True,
    marker_pos=1/2,
    **kwargs,
):
    if ordering:
        df = ensure_ordering(df, ordering[0], ordering[1])

    baselines = []
    aug_csv_mapping = {}
    aug_color_discrete_map = {**color_discrete_map}
    aug_symbol_map = {}
    aug_lines_map = {**lines_map}
    aug_legendranks = [0]*6

    symbols = ["square", "circle", "x", "triangle-up"]
    for i, impressions in enumerate([0, 3, 6, 9]):
        baseline, csv_baseline = (COOKIEMONSTER, "cookiemonster")
        title = f"{baseline}+{impressions} impressions/conversion" if impressions else baseline
        csv = f"{csv_baseline}_{impressions}"
        baselines.append(title)
        aug_csv_mapping[title] = csv
        aug_color_discrete_map[title] = aug_color_discrete_map[baseline]
        aug_symbol_map[title] = symbols[i]
        aug_lines_map[title] = aug_lines_map[baseline]
        if not i:
            aug_legendranks[i] = i
        else:
            aug_legendranks[i] = 2*i - 1

    for i, (baseline, csv_baseline) in enumerate([(COOKIEMONSTER_BASE, "cookiemonster_base"), (IPA, "ipa")]):
        impressions = 0
        title = baseline
        csv = f"{csv_baseline}_{impressions}"
        baselines.append(title)
        aug_csv_mapping[title] = csv
        aug_legendranks[i + 4] = 2*(i + 1)

    traces = []
    for i, baseline in enumerate(baselines):
        csv_name = aug_csv_mapping[baseline]
        group = df.query("baseline == @csv_name")
        group = group.sort_values(by=[metric])
        len_values = group.shape[0]
        start = 1

        if group.empty:
            continue

        group.dropna(inplace=True)
        stop = group.shape[0]
        values = np.sort(group[metric].values)
        cumulative_probabilities = np.arange(
            start, stop + 1) / float(len_values) * 100

        lines_only = (not baseline.startswith(COOKIEMONSTER)
                      ) or baseline == COOKIEMONSTER

        traces.append(go.Scatter(
            x=cumulative_probabilities,
            y=values,
            legendgroup=baseline,
            name=baseline,  # 381000 devices; 109 devices for IPA
            marker_color=aug_color_discrete_map[baseline],
            showlegend=lines_only,
            legendrank=-1*aug_legendranks[i] if lines_only else 1000,
            mode="lines",
            line=dict(dash=aug_lines_map[baseline]),
        ))

        if baseline.startswith(COOKIEMONSTER) and baseline != COOKIEMONSTER:
            def get_index(l: int, i: int) -> int:
                return math.floor(l * i/4)

            sample_x = [
                cumulative_probabilities[get_index(
                    len(cumulative_probabilities), i)]
                for i in range(1, 2)
            ]
            sample_y = [
                values[get_index(len(cumulative_probabilities), i)]
                for i in range(1, 2)
            ]

            name = baseline.replace(COOKIEMONSTER, "")
            traces.append(go.Scatter(
                x=sample_x,
                y=sample_y,
                legendgroup=name,
                name=name,
                marker_color=aug_color_discrete_map[baseline],
                marker_symbol=aug_symbol_map[baseline],
                showlegend=showlegend,
                legendrank=-1*aug_legendranks[i],
                mode="lines+markers",
                marker=dict(size=12),
            ))

    return traces
