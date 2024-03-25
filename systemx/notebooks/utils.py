import pandas as pd
import plotly.express as px
from plotly.offline import iplot
from systemx.utils import LOGS_PATH
from experiments.ray.analysis import load_ray_experiment


def get_df(path):
    logs = LOGS_PATH.joinpath(path)
    df = load_ray_experiment(logs)
    return df


def analyze_budget_consumption(path):

    def get_logs(row):
        
        budget_logs_per_destination = row["logs"]["budget"]
        records = []
        
        for destination, destination_logs in budget_logs_per_destination.items():    

            cumulative_avg_budget_per_user = {}
            users = set()
            for log in destination_logs:
                users.add(log["user_id"])
            num_users = len(users)
            
            for log in destination_logs:

                timestamp = log["timestamp"]
                user_id = log["user_id"]
                total_budget_consumed = log["total_budget_consumed"]

                if user_id not in cumulative_avg_budget_per_user:
                    cumulative_avg_budget_per_user[user_id] = 0
                cumulative_avg_budget_per_user[user_id] += total_budget_consumed
                

                max_avg_budget_consumed = max(cumulative_avg_budget_per_user.values()) / 93
                avg_avg_budget_consumed = sum(cumulative_avg_budget_per_user.values()) / num_users
                
                records.append(
                        {
                            "destination_id": destination,
                            "num_reports": timestamp,
                            "max_avg_budget_conusmed": max_avg_budget_consumed,
                            "avg_avg_budget_consumed": avg_avg_budget_consumed,
                            "status": log["status"],
                            "baseline": row["baseline"],
                            "optimization": row["optimization"],
                            "num_days_per_epoch": row["num_days_per_epoch"],
                            "num_days_attribution_window": row[
                                "num_days_attribution_window"
                            ],
                        }
                )

        return pd.DataFrame.from_records(records)


    dfs = []
    df = get_df(path)
    for _, row in df.iterrows():
        dfs.append(get_logs(row))

    return pd.concat(dfs)


def analyze_bias(path):
    df = get_df(path)
    bias_logs_per_destination = df["logs"]["bias"]

    def get_logs(row):
        logs = []
        for destination, destination_logs in bias_logs_per_destination.items():
            total_bias = 0
            for i, log in enumerate(destination_logs):
                total_bias += log["bias"]
                logs.append(
                    {
                        "destination_id": destination,
                        "conversion_timestamp": i,
                        "bias": total_bias,
                        # "user_id": log["user_id"],
                        # "epoch_window": log["epoch_window"],
                        "status": log["status"],
                        "baseline": row["baseline"],
                        "optimization": row["optimization"],
                        "num_days_per_epoch": row["num_days_per_epoch"],
                        # "num_days_attribution_window": row[
                        # "num_days_attribution_window"
                        # ],
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
