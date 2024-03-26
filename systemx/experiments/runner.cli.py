import os
import typer
from ray_runner import grid_run

app = typer.Typer()


def budget_consumption(dataset):
    logs_dir = f"{dataset}/optimizations"
    config = {
        "baseline": ["ipa", "user_epoch_ara", "systemx"],
        "optimization": ["multiepoch"],
        "dataset_name": f"{dataset}",
        "impressions_path": f"{dataset}/{dataset}_impressions.csv",
        "conversions_path": f"{dataset}/{dataset}_conversions.csv",
        "num_days_per_epoch": [1],  # [1, 15, 30],
        "num_days_attribution_window": 30,
        "workload_size": [4],
        "scheduling_batch_size_per_query": 20000,
        "initial_budget": [100000000],
        "logs_dir": logs_dir,
        "loguru_level": "INFO",
        "mlflow_experiment_id": "",
    }

    logs = grid_run(**config)

    # analyze(f"ray/{logs_dir}")


@app.command()
def run(
    exp: str = "budget_consumption",
    dataset: str = "synthetic",
    loguru_level: str = "INFO",
):
    os.environ["LOGURU_LEVEL"] = loguru_level
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    globals()[f"{exp}"](dataset)


if __name__ == "__main__":
    app()
