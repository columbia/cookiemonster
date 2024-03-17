import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any


REPO_ROOT = Path(__file__).parent.parent
LOGS_PATH = REPO_ROOT.joinpath("logs")
RAY_LOGS = LOGS_PATH.joinpath("ray")


def attribution_window_to_list(attribution_window: Tuple[int, int]) -> List[int]:
    return list(range(attribution_window[0], attribution_window[1] + 1))


def get_data_path(path):
    return str(REPO_ROOT.joinpath(f"data/{path}"))


def load_logs(log_path: str, relative_path=True) -> dict:
    full_path = Path(log_path)
    if relative_path:
        full_path = LOGS_PATH.joinpath(log_path)
    with open(full_path, "r") as f:
        logs = json.load(f)
    return logs


def process_logs(
    log: List[Dict[str, Dict[str, float]]], config: Dict[str, Any]
) -> dict:

    destination_mappings = {}
    remaining_budget_per_user_per_destination_per_epoch = []

    for user in log:
        user_data = {}
        for destination, filter in user.items():
            if destination not in destination_mappings:
                destination_mappings[destination] = len(destination_mappings)

            user_data[destination_mappings[destination]] = filter

        remaining_budget_per_user_per_destination_per_epoch.append(user_data)

    proceessed_logs = {
        "remaining_budget_per_user_per_destination_per_epoch": remaining_budget_per_user_per_destination_per_epoch,
        "initial_budget": config["user"]["initial_budget"],
        "optimization": config["user"]["optimization"],
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
