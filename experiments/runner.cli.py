import os
import time
import typer
import multiprocessing
from copy import deepcopy
from ray_runner import grid_run
from cookiemonster.utils import BUDGET, QUERY_RESULTS

app = typer.Typer()


def experiments_start_and_join(experiments):
    for p in experiments:
        time.sleep(5)
        p.start()
    for p in experiments:
        p.join()


def get_path(path_base, knob1, knob2):
    return f"{path_base}_knob1_{knob1}_knob2_{knob2}.csv"


def budget_consumption_vary_knob1(dataset, ray_session_dir):

    logs_dir = f"{dataset}/budget_consumption_varying_knob1"

    experiments = []

    impressions_path_base = f"{dataset}/impressions"
    conversions_path_base = f"{dataset}/conversions"

    knob1s = [0.001, 0.01, 0.1, 1.0]
    knob2 = 0.1

    config = {
        "baseline": ["ipa", "user_epoch_ara", "cookiemonster"],
        "dataset_name": f"{dataset}",
        "num_days_per_epoch": [7],  # [1, 15, 30],
        "num_days_attribution_window": 30,
        "workload_size": [5],
        "scheduling_batch_size_per_query": 10000,
        "initial_budget": [1],  # TODO: check that I can safely change this to 1
        "logs_dir": logs_dir,
        "loguru_level": "INFO",
        "ray_session_dir": ray_session_dir,
        "logging_keys": [BUDGET],
    }

    for knob1 in knob1s:
        config["impressions_path"] = get_path(impressions_path_base, knob1, knob2)
        config["conversions_path"] = get_path(conversions_path_base, knob1, knob2)
        experiments.append(
            multiprocessing.Process(
                target=lambda config: grid_run(**config), args=(deepcopy(config),)
            )
        )

    experiments_start_and_join(experiments)
    # analyze(f"ray/{logs_dir}")


def budget_consumption_vary_knob2(dataset, ray_session_dir):

    logs_dir = f"{dataset}/budget_consumption_varying_knob2"

    experiments = []

    impressions_path_base = f"{dataset}/impressions"
    conversions_path_base = f"{dataset}/conversions"

    knob1 = 0.1
    knob2s = [0.001, 0.01, 0.1, 1.0]

    config = {
        "baseline": ["ipa", "user_epoch_ara", "cookiemonster"],
        "dataset_name": f"{dataset}",
        "num_days_per_epoch": [7],
        "num_days_attribution_window": 30,
        "workload_size": [5],
        "scheduling_batch_size_per_query": 10000,
        "initial_budget": [1],  # TODO: check that I can safely change this to 1
        "logs_dir": logs_dir,
        "loguru_level": "INFO",
        "ray_session_dir": ray_session_dir,
        "logging_keys": [BUDGET],
    }

    for knob2 in knob2s:
        config["impressions_path"] = get_path(impressions_path_base, knob1, knob2)
        config["conversions_path"] = get_path(conversions_path_base, knob1, knob2)
        experiments.append(
            multiprocessing.Process(
                target=lambda config: grid_run(**config), args=(deepcopy(config),)
            )
        )

    experiments_start_and_join(experiments)
    # analyze(f"ray/{logs_dir}")


def budget_consumption_vary_epoch_granularity(dataset, ray_session_dir):

    logs_dir = f"{dataset}/budget_consumption_varying_epoch_granularity"

    impressions_path_base = f"{dataset}/impressions"
    conversions_path_base = f"{dataset}/conversions"

    knob1 = 0.1
    knob2 = 0.1

    config = {
        "baseline": ["ipa", "user_epoch_ara", "cookiemonster"],
        "dataset_name": f"{dataset}",
        "impressions_path": get_path(impressions_path_base, knob1, knob2),
        "conversions_path": get_path(conversions_path_base, knob1, knob2),
        "num_days_per_epoch": [1, 7, 14, 21, 28],
        "num_days_attribution_window": 30,
        "workload_size": [4],
        "scheduling_batch_size_per_query": 10000,
        "initial_budget": [1],  # TODO: check that I can safely change this to 1
        "logs_dir": logs_dir,
        "loguru_level": "INFO",
        "ray_session_dir": ray_session_dir,
        "logging_keys": [BUDGET],
    }

    grid_run(**config)
    # analyze(f"ray/{logs_dir}")


def bias_vary_workload_size(dataset, ray_session_dir):

    logs_dir = f"{dataset}/bias_varying_workload_size2"

    impressions_path_base = f"{dataset}/impressions"
    conversions_path_base = f"{dataset}/conversions"

    knob1 = 0.1
    knob2 = 0.1

    config = {
        "baseline": ["ipa", "user_epoch_ara", "cookiemonster"],
        "dataset_name": f"{dataset}",
        "impressions_path": get_path(impressions_path_base, knob1, knob2),
        "conversions_path": get_path(conversions_path_base, knob1, knob2),
        "num_days_per_epoch": [7],
        "num_days_attribution_window": 30,
        "workload_size": [1, 10, 20, 30, 40, 50],
        "scheduling_batch_size_per_query": 10000,
        "initial_budget": [0.5],
        "logs_dir": logs_dir,
        "loguru_level": "INFO",
        "ray_session_dir": ray_session_dir,
        "logging_keys": [QUERY_RESULTS],
    }

    grid_run(**config)
    # analyze(f"ray/{logs_dir}")


def criteo_bias_vary_workload_size(dataset, ray_session_dir):
    """
    Varying Workload methodology:
      1. Generate conversions for the largest 6 advertisers (advertisers with the most queries in their query pool)
      2. Run the varying workload with initial budget set to 1 across workload sizes of 1, 5, 20, 15, and 20.
    * 3. Run the varying workload with initial budget set to 1 across workload sizes of 25, 30, 35, 40, 45.
      4. Generate conversions for the middle 6 advertisers
      5. Run the varying workload with initial budget set to 1 across workload sizes of 1, 3, 6, 9, 10.
      6. Generate conversions for the smallest 6 advertisers
      7. Run the varying workload with initial budget set to 1 across workload sizes of 1, 2, 3, 4.
    """

    logs_dir = f"{dataset}/bias_varying_workload_size"

    impressions_path_base = f"{dataset}/{dataset}_query_pool_impressions.csv"
    conversions_path_base = f"{dataset}/{dataset}_query_pool_conversions.csv"

    config = {
        "baseline": ["ipa", "user_epoch_ara", "cookiemonster"],
        "dataset_name": f"{dataset}",
        "impressions_path": impressions_path_base,
        "conversions_path": conversions_path_base,
        "num_days_per_epoch": [7],
        "num_days_attribution_window": 30,
        "workload_size": [25, 30, 35, 40, 45],  # [1, 5, 10, 15, 20],
        "max_scheduling_batch_size_per_query": 20_000,
        "min_scheduling_batch_size_per_query": 1_500,
        "initial_budget": [1],
        "logs_dir": logs_dir,
        "loguru_level": "INFO",
        "ray_session_dir": ray_session_dir,
        "logging_keys": [QUERY_RESULTS, BUDGET],
    }

    grid_run(**config)


def bias_vary_initial_budget(dataset, ray_session_dir):

    logs_dir = f"{dataset}/bias_varying_initial_budget"

    impressions_path_base = f"{dataset}/impressions"
    conversions_path_base = f"{dataset}/conversions"

    knob1 = 0.1
    knob2 = 0.1

    config = {
        "baseline": ["ipa", "user_epoch_ara", "cookiemonster"],
        "dataset_name": f"{dataset}",
        "impressions_path": get_path(impressions_path_base, knob1, knob2),
        "conversions_path": get_path(conversions_path_base, knob1, knob2),
        "num_days_per_epoch": [7],
        "num_days_attribution_window": 30,
        "workload_size": [50],
        "scheduling_batch_size_per_query": 10000,
        "initial_budget": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "logs_dir": logs_dir,
        "loguru_level": "INFO",
        "ray_session_dir": ray_session_dir,
        "logging_keys": [QUERY_RESULTS],
    }
    grid_run(**config)
    # analyze(f"ray/{logs_dir}")


@app.command()
def run(
    exp: str = "budget_consumption_vary_conversions_rate",
    dataset: str = "synthetic",
    ray_session_dir: str = "",
    loguru_level: str = "INFO",
):
    os.environ["LOGURU_LEVEL"] = loguru_level
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    globals()[f"{exp}"](dataset, ray_session_dir)


if __name__ == "__main__":
    app()
