import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Union

from systemx.budget import BasicBudget
from systemx.budget_accountant import BudgetAccountant


REPO_ROOT = Path(__file__).parent.parent
LOGS_PATH = REPO_ROOT.joinpath("logs")
RAY_LOGS = LOGS_PATH.joinpath("ray")


kNulledReport = "Null"

IPA = "ipa"
USER_EPOCH_ARA = "user_epoch_ara"
SYSTEMX = "systemx"
MONOEPOCH = "monoepoch"
MULTIEPOCH = "multiepoch"


def maybe_initialize_filters(
    filters_per_origin,
    destination: str,
    attribution_epochs: Tuple[int, int],
    initial_budget: float,
):
    if destination not in filters_per_origin:
        filters_per_origin[destination] = BudgetAccountant()
    destination_filter = filters_per_origin[destination]

    destination_filter.maybe_initialize(attribution_epochs, initial_budget)
    return destination_filter


def compute_global_sensitivity(sensitivity_metric, aggregatable_cap_value):
    match sensitivity_metric:
        case "L1":
            global_sensitivity = aggregatable_cap_value
        case _:
            raise ValueError(f"Unsupported sensitivity metric: {sensitivity_metric}")
    assert global_sensitivity is not None
    return global_sensitivity


def get_data_path(path):
    return str(REPO_ROOT.joinpath(f"data/{path}"))


def load_logs(log_path: str, relative_path=True) -> dict:
    full_path = Path(log_path)
    if relative_path:
        full_path = LOGS_PATH.joinpath(log_path)
    with open(full_path, "r") as f:
        logs = json.load(f)
    return logs


def process_logs(logs: Dict[str, List[Dict[str, Any]]], config: Dict[str, Any]) -> dict:

    proceessed_logs = {
        "destination_logs": logs,
        "baseline": config["user"]["baseline"],
        "optimization": config["user"]["optimization"],
        "num_days_per_epoch": config["dataset"]["num_days_per_epoch"],
        "num_days_attribution_window": config["dataset"]["num_days_attribution_window"],
        "initial_budget": config["user"]["initial_budget"],
        "dataset": config["dataset"]["name"],
        "config": config,
    }
    return proceessed_logs


def save_logs(log_dict, save_dir):
    log_path = LOGS_PATH.joinpath(save_dir).joinpath(
        f"{datetime.now().strftime('%m%d-%H%M%S')}_{str(uuid.uuid4())[:6]}/result.json"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as fp:
        json_object = json.dumps(log_dict, indent=4)
        fp.write(json_object)
