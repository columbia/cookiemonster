import os
import typer
import argprase

from ray_runner import grid_run

app = typer.Typer()


# def optimizations_criteo(dataset):
#     logs_dir = f"{dataset}/optimizations"
#     config = {
#         "optimization": ["0", "1", "2"],
#         "dataset_name": "criteo",
#         "impressions_path": "criteo/criteo_impressions.csv",
#         "conversions_path": "criteo/criteo_conversions.csv",
#         "initial_budget": [1],
#         "logs_dir": logs_dir,
#         "loguru_level": "INFO",
#         "mlflow_experiment_id": "",
#     }

#     logs = grid_run(**config)

    # analyze(f"ray/{logs_dir}")

def optimizations(dataset) : 
    logs_dir = f"{dataset}/optimizations"
    config = {
        "optimization": ["0", "1", "2"],
        "dataset_name": "{dataset}",
        "impressions_path": "{dataset}/{dataset}_impressions.csv",
        "conversions_path": "{dataset}/{dataset}_conversions.csv",
        "initial_budget": [1],
        "logs_dir": logs_dir,
        "loguru_level": "INFO",
        "mlflow_experiment_id": "",
    }

    logs = grid_run(**config)


@app.command()
def run(
    exp: str = "optimizations",
    dataset: str = "criteo",
    loguru_level: str = "INFO",
):
    os.environ["LOGURU_LEVEL"] = loguru_level
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    globals()[f"{exp}"](dataset)


if __name__ == "__main__":
    parser = argprase.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="Run experiments on synthetic dataset")
    args = parser.parse_args()

    if args.synthetic:
        app(dataset = "synthetic")
    else :
        app()
