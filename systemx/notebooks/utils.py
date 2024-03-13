import numpy as np
import pandas as pd
import plotly.express as px
from plotly.offline import iplot

from systemx.utils import LOGS_PATH
from experiments.ray.analysis import load_ray_experiment

def get_df(path):
    logs = LOGS_PATH.joinpath(path)
    df = load_ray_experiment(logs)
    return df


def get_avg_budget_consumption(df):
    results = []
    for _, row  in df.iterrows():
        initial_budget = row["initial_budget"]
        b = row["remaining_budgets_per_epoch"]
        
        max_index = max([ max(map(int, d.keys())) if d else 0 for d in b] )
        print(max_index)
        dense_matrix = np.zeros((len(b), max_index + 1))
        for i, d in enumerate(b):
            if d:
                keys = np.array(list(map(int, d.keys())))
                values = initial_budget - np.array(list(d.values()))
                dense_matrix[i, keys] = values

        avg_consumed_budget_per_epoch = np.mean(dense_matrix, axis=0)
        avg_consumed_budget_across_epochs = np.mean(avg_consumed_budget_per_epoch)
        results.append(
            pd.DataFrame([{
                "optimization": row["optimization"],
                "initial_budget": initial_budget,
                "avg_consumed_budget_per_epoch": avg_consumed_budget_per_epoch,
                "avg_consumed_budget_across_epochs": avg_consumed_budget_across_epochs,
                }]
            )
        )
    
    return pd.concat(results)


def plot_avg_budget_across_epochs(df):
    df = df.sort_values(["optimization"])
    fig = px.bar(
        df,
        x="optimization",
        y="avg_consumed_budget_across_epochs",
        color="optimization",
        title=f"Avg. Budget Consumption across epochs/advertisers",
        width=900,
        height=500,
    )
    return fig


def analyze_budget_consumption(path):
    df = get_df(path)
    dff = get_avg_budget_consumption(df)
    iplot(plot_avg_budget_across_epochs(dff))
    return dff