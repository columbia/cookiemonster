import os
import time
import typer
from copy import deepcopy
from ray_runner import grid_run
import multiprocessing

app = typer.Typer()

def experiments_start_and_join(experiments):
    for p in experiments:
        time.sleep(5)
        p.start()
    for p in experiments:
        p.join()

def get_path(path_base, conversions_rate, impression_rate):
    return f"{path_base}_conv_rate_{conversions_rate}_impr_rate_{impression_rate}.csv"

def budget_consumption_vary_conversions_rate(dataset):
    
    logs_dir = f"{dataset}/budget_consumption"

    experiments = []
    
    impressions_path_base = f"{dataset}/{dataset}_impressions"
    conversions_path_base = f"{dataset}/{dataset}_conversions"

    impression_rate = 0.036

    conversions_rate = 0.1
    config = {
        "baseline": ["ipa", "user_epoch_ara", "systemx"],
        "optimization": ["multiepoch"],
        "dataset_name": f"{dataset}",
        "impressions_path": get_path(impressions_path_base, conversions_rate, impression_rate),
        "conversions_path": get_path(conversions_path_base, conversions_rate, impression_rate),
        "num_days_per_epoch": [1],  # [1, 15, 30],
        "num_days_attribution_window": 30,
        "workload_size": [4],
        "scheduling_batch_size_per_query": 20000,
        "initial_budget": [100000000],
        "logs_dir": logs_dir,
        "loguru_level": "INFO",
        "mlflow_experiment_id": "",
    }

    experiments.append(
        multiprocessing.Process(
            target=lambda config: grid_run(**config), args=(deepcopy(config),)
        )
    )
    
    conversions_rate = 1
    config["impressions_path"] = get_path(impressions_path_base, conversions_rate, impression_rate)
    config["conversions_path"] = get_path(conversions_path_base, conversions_rate, impression_rate)

    experiments.append(
        multiprocessing.Process(
            target=lambda config: grid_run(**config), args=(deepcopy(config),)
        )
    )
    experiments_start_and_join(experiments)

    # analyze(f"ray/{logs_dir}")


@app.command()
def run(
    exp: str = "budget_consumption_vary_conversions_rate",
    dataset: str = "synthetic",
    loguru_level: str = "INFO",
):
    os.environ["LOGURU_LEVEL"] = loguru_level
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    globals()[f"{exp}"](dataset)


if __name__ == "__main__":
    app()
