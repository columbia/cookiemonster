import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any


REPO_ROOT = Path(__file__).parent.parent
LOGS_PATH = REPO_ROOT.joinpath("logs")
RAY_LOGS = LOGS_PATH.joinpath("ray")

kInsufficientBudgetError = "InsufficientBudgetError"
kOk = "OK"
kNulledReport = "Null"

IPA = "ipa"
USER_EPOCH_ARA = "user_epoch_ara"
SYSTEMX = "systemx"
MONOEPOCH = "monoepoch"
MULTIEPOCH = "multiepoch"


def epoch_window_to_list(epoch_window: Tuple[int, int]) -> List[int]:
    return list(range(epoch_window[0], epoch_window[1] + 1))


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
