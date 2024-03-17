import os
import typer

from ray_runner import grid_run

app = typer.Typer()


def optimizations_criteo(dataset):
    logs_dir = f"{dataset}/optimizations"
    config = {
        "optimization": ["0", "1", "2"],
        "dataset_name": "criteo",
        "impressions_path": "criteo/criteo_impressions.csv",
        "conversions_path": "criteo/criteo_conversions.csv",
        "num_days_per_epoch": [1, 30],
        "num_days_attribution_window": 30,
        "initial_budget": [1],
        "logs_dir": logs_dir,
        "loguru_level": "INFO",
        "mlflow_experiment_id": "",
    }

    logs = grid_run(**config)

    # analyze(f"ray/{logs_dir}")


@app.command()
def run(
    exp: str = "optimizations",
    dataset: str = "criteo",
    loguru_level: str = "INFO",
):
    os.environ["LOGURU_LEVEL"] = loguru_level
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    globals()[f"{exp}_{dataset}"](dataset)


if __name__ == "__main__":
    app()
