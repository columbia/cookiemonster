import ray
from ray import tune
from ray import train
from loguru import logger
from typing import Any, Dict, List
from rich.pretty import pretty_repr

from cookiemonster.run_evaluation import Evaluation
from cookiemonster.utils import RAY_LOGS, get_data_path


def run_and_report(config: dict, replace=False) -> None:
    logs = Evaluation(config).run()
    if logs:
        train.report(logs)


def grid_run(
    baseline: List[str],
    dataset_name: str,
    impressions_path: str,
    conversions_path: str,
    num_days_per_epoch: List[int],
    num_days_attribution_window: List[int],
    workload_size: List[int],
    min_scheduling_batch_size_per_query: int,
    max_scheduling_batch_size_per_query: int,
    initial_budget: float,
    logs_dir: str,
    loguru_level: str,
    ray_session_dir: str,
    logging_keys: List[str],
    ray_init: bool = True,
):

    if ray_session_dir and ray_init:
        ray.init(_temp_dir=ray_session_dir, log_to_driver=False)

    config = {
        "user": {
            "sensitivity_metric": "L1",
            "baseline": tune.grid_search(baseline),
            "initial_budget": tune.grid_search(initial_budget),
        },
        "dataset": {
            "name": dataset_name,
            "impressions_path": get_data_path(impressions_path),
            "conversions_path": get_data_path(conversions_path),
            "num_days_per_epoch": tune.grid_search(num_days_per_epoch),
            "num_days_attribution_window": tune.grid_search(
                num_days_attribution_window
            ),
            "workload_size": tune.grid_search(workload_size),
        },
        "logs": {
            "save": False,
            "save_dir": "",
            "logging_keys": logging_keys,
            "loguru_level": loguru_level,
        },
        "aggregation_service": "local_laplacian",
        "aggregation_policy": {
            "type": "count_conversion_policy",
            "min_interval": min_scheduling_batch_size_per_query,
            "max_interval": max_scheduling_batch_size_per_query,
        },
    }

    logger.info(f"Tune config: {pretty_repr(config)}")

    experiment_analysis = tune.run(
        run_and_report,
        config=config,
        resources_per_trial={"cpu": 1},
        storage_path=str(RAY_LOGS.joinpath(logs_dir)),
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
