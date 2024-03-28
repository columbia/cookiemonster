from typing import Any, Dict, List

import ray
from ray import tune
from ray import train
from loguru import logger
from rich.pretty import pretty_repr

from cookiemonster.utils import RAY_LOGS, get_data_path
from cookiemonster.run_evaluation import Evaluation


def run_and_report(config: dict, replace=False) -> None:
    logs = Evaluation(config).run()
    if logs:
        train.report(logs)


def grid_run(
    baseline: List[str],
    optimization: List[str],
    dataset_name: str,
    impressions_path: str,
    conversions_path: str,
    num_days_per_epoch: List[int],
    num_days_attribution_window: int,
    workload_size: List[int],
    scheduling_batch_size_per_query: int,
    initial_budget: float,
    logs_dir: str,
    loguru_level: str,
    mlflow_experiment_id: str,
):

    config = {
        "sensitivity_metric": "L1",
        "scheduling_batch_size_per_query": scheduling_batch_size_per_query,
        "user": {
            "sensitivity_metric": "L1",
            "baseline": tune.grid_search(baseline),
            "optimization": tune.grid_search(optimization),
            "initial_budget": tune.grid_search(initial_budget),
        },
        "dataset": {
            "name": dataset_name,
            "impressions_path": get_data_path(impressions_path),
            "conversions_path": get_data_path(conversions_path),
            "num_days_per_epoch": tune.grid_search(num_days_per_epoch),
            "num_days_attribution_window": num_days_attribution_window,
            "workload_size": tune.grid_search(workload_size),
        },
        "logs": {
            "verbose": False,
            "save": True,
            "save_dir": "",
            "mlflow": False,
            "mlflow_experiment_id": mlflow_experiment_id,
            "loguru_level": loguru_level,
        },
    }

    logger.info(f"Tune config: {pretty_repr(config)}")

    experiment_analysis = tune.run(
        run_and_report,
        config=config,
        resources_per_trial={"cpu": 1},
        local_dir=str(RAY_LOGS.joinpath(logs_dir)),
        resume=False,
        verbose=1,
        callbacks=[
            CustomLoggerCallback(),
            tune.logger.JsonLoggerCallback(),
        ],
        progress_reporter=ray.tune.CLIReporter(
            metric_columns=[],
            parameter_columns={
                "dataset/name": "dataset",
                "user/baseline": "baseline",
            },
            max_report_frequency=60,
        ),
    )
    # all_trial_paths = experiment_analysis._get_trial_paths()
    # experiment_dir = Path(all_trial_paths[0]).parent


class CustomLoggerCallback(tune.logger.LoggerCallback):
    """Custom logger interface"""

    def __init__(self, metrics=[]) -> None:
        self.metrics = []
        self.metrics.extend(metrics)
        super().__init__()

    def log_trial_result(self, iteration: int, trial: Any, result: Dict):
        logger.info([f"{key}: {result[key]}" for key in self.metrics])
        return

    def on_trial_complete(self, iteration: int, trials: List, trial: Any, **info):
        return
