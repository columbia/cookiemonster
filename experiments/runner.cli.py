import os
import time
import typer
import multiprocessing
from copy import deepcopy
from ray_runner import grid_run
from cookiemonster.utils import BUDGET, BIAS

app = typer.Typer()


def experiments_start_and_join(experiments):
    for p in experiments:
        time.sleep(5)
        p.start()
    for p in experiments:
        p.join()


## ----------------- MICROBENCHMARK ----------------- ##


def get_path(path_base, knob1, knob2):
    return f"{path_base}_knob1_{knob1}_knob2_{knob2}.csv"


def microbenchmark_varying_knob1(ray_session_dir):
    dataset = "microbenchmark"
    logs_dir = f"{dataset}/varying_knob1"

    experiments = []

    impressions_path_base = f"{dataset}/impressions"
    conversions_path_base = f"{dataset}/conversions"

    knob1s = [0.001, 0.01, 0.1, 1.0]
    knob2 = 0.1

    config = {
        "baseline": ["ipa", "cookiemonster_base", "cookiemonster"],
        "dataset_name": f"{dataset}",
        "num_days_per_epoch": [7],
        "num_days_attribution_window": [30],
        "workload_size": [5],
        "min_scheduling_batch_size_per_query": 1000,
        "max_scheduling_batch_size_per_query": 1000,
        "initial_budget": [1],
        "logs_dir": logs_dir,
        "loguru_level": "INFO",
        "ray_session_dir": ray_session_dir,
        "logging_keys": [BUDGET, BIAS],
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


def microbenchmark_varying_knob2(ray_session_dir):
    dataset = "microbenchmark"
    logs_dir = f"{dataset}/varying_knob2"

    experiments = []

    impressions_path_base = f"{dataset}/impressions"
    conversions_path_base = f"{dataset}/conversions"

    knob1 = 0.1
    knob2s = [0.001, 0.01, 0.1, 1.0]

    config = {
        "baseline": ["ipa", "cookiemonster_base", "cookiemonster"],
        "dataset_name": f"{dataset}",
        "num_days_per_epoch": [7],
        "num_days_attribution_window": [30],
        "workload_size": [5],
        "min_scheduling_batch_size_per_query": 1000,
        "max_scheduling_batch_size_per_query": 1000,
        "initial_budget": [1],
        "logs_dir": logs_dir,
        "loguru_level": "INFO",
        "ray_session_dir": ray_session_dir,
        "logging_keys": [BUDGET, BIAS],
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


def microbenchmark_varying_epoch_granularity(ray_session_dir):
    dataset = "microbenchmark"
    logs_dir = f"{dataset}/varying_epoch_granularity"

    impressions_path_base = f"{dataset}/impressions"
    conversions_path_base = f"{dataset}/conversions"

    knob1 = 0.1
    knob2 = 0.1

    config = {
        "baseline": ["ipa", "cookiemonster_base", "cookiemonster"],
        "dataset_name": f"{dataset}",
        "impressions_path": get_path(impressions_path_base, knob1, knob2),
        "conversions_path": get_path(conversions_path_base, knob1, knob2),
        "num_days_per_epoch": [1, 7, 14, 21, 28],
        "num_days_attribution_window": [30],
        "workload_size": [5],
        "min_scheduling_batch_size_per_query": 1000,
        "max_scheduling_batch_size_per_query": 1000,
        "initial_budget": [1],  # TODO: check that I can safely change this to 1
        "logs_dir": logs_dir,
        "loguru_level": "INFO",
        "ray_session_dir": ray_session_dir,
        "logging_keys": [BUDGET, BIAS],
    }

    grid_run(**config)
    # analyze(f"ray/{logs_dir}")


## ----------------- CRITEO ----------------- ##

def criteo_bias_varying_epoch_size(ray_session_dir):
    dataset = "criteo"
    logs_dir = f"{dataset}/bias_varying_epoch_size"
    impressions_path = f"{dataset}/{dataset}_query_pool_impressions.csv"
    conversions_path = f"{dataset}/{dataset}_query_pool_conversions.csv"

    config = {
        "baseline": ["ipa", "cookiemonster_base", "cookiemonster"],
        "dataset_name": f"{dataset}",
        "impressions_path": impressions_path,
        "conversions_path": conversions_path,
        "num_days_per_epoch": [1, 7, 14, 21, 28, 30, 60, 90],
        "num_days_attribution_window": [30],
        "workload_size": [970],
        "max_scheduling_batch_size_per_query": 450,
        "min_scheduling_batch_size_per_query": 300,
        "initial_budget": [1],
        "logs_dir": logs_dir,
        "loguru_level": "INFO",
        "ray_session_dir": ray_session_dir,
        "logging_keys": [BIAS, BUDGET],
    }

    grid_run(**config)


## ----------------- PATCG ----------------- ##


def patcg_varying_epoch_granularity(ray_session_dir):
    dataset = "patcg"
    logs_dir = f"{dataset}/varying_epoch_granularity_aw_30"

    impressions_path = f"{dataset}/v375_{dataset}_impressions.csv"
    conversions_path = f"{dataset}/v375_{dataset}_conversions.csv"

    config = {
        "baseline": ["ipa", "cookiemonster_base", "cookiemonster"],
        "dataset_name": f"{dataset}",
        "impressions_path": impressions_path,
        "conversions_path": conversions_path,
        "num_days_per_epoch": [21, 28, 30, 60],
        "num_days_attribution_window": [30],
        "workload_size": [80],
        "max_scheduling_batch_size_per_query": 303009,
        "min_scheduling_batch_size_per_query": 280000,
        "initial_budget": [1],
        "logs_dir": logs_dir,
        "loguru_level": "INFO",
        "ray_session_dir": ray_session_dir,
        "logging_keys": [BUDGET, BIAS],
    }

    grid_run(**config)
    config["num_days_per_epoch"] = [1, 7, 14]
    grid_run(**config)

def patcg_varying_initial_budget(ray_session_dir):
    dataset = "patcg"
    logs_dir = f"{dataset}/varying_initial_budget"

    impressions_path = f"{dataset}/v375_{dataset}_impressions.csv"
    conversions_path = f"{dataset}/v375_{dataset}_conversions.csv"

    config = {
        "baseline": ["ipa", "cookiemonster_base", "cookiemonster"],
        "dataset_name": f"{dataset}",
        "impressions_path": impressions_path,
        "conversions_path": conversions_path,
        "num_days_per_epoch": [7],
        "num_days_attribution_window": [7],
        "workload_size": [80],
        "max_scheduling_batch_size_per_query": 303009,
        "min_scheduling_batch_size_per_query": 280000,
        "initial_budget": [1, 2, 4, 6, 8, 10],
        "logs_dir": logs_dir,
        "loguru_level": "INFO",
        "ray_session_dir": ray_session_dir,
        "logging_keys": [BUDGET, BIAS],
    }

    grid_run(**config)



def patcg_bias_varying_attribution_window(ray_session_dir):
    dataset = "patcg"
    logs_dir = f"{dataset}/bias_varying_attribution_window"

    impressions_path = f"{dataset}/v375_{dataset}_impressions.csv"
    conversions_path = f"{dataset}/v375_{dataset}_conversions.csv"

    config = {
        "baseline": ["ipa", "cookiemonster_base", "cookiemonster"],
        "dataset_name": f"{dataset}",
        "impressions_path": impressions_path,
        "conversions_path": conversions_path,
        "num_days_per_epoch": [7],
        "num_days_attribution_window": [1, 7, 14, 21, 28],
        "workload_size": [80],
        "max_scheduling_batch_size_per_query": 303009,
        "min_scheduling_batch_size_per_query": 280000,
        "initial_budget": [1],
        "logs_dir": logs_dir,
        "loguru_level": "INFO",
        "ray_session_dir": ray_session_dir,
        "logging_keys": [BUDGET, BIAS],
    }

    grid_run(**config)


@app.command()
def run(
    exp: str = "budget_consumption_vary_conversions_rate",
    ray_session_dir: str = "",
    loguru_level: str = "INFO",
):
    os.environ["LOGURU_LEVEL"] = loguru_level
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    globals()[f"{exp}"](ray_session_dir)


if __name__ == "__main__":
    app()
