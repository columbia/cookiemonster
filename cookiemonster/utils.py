import json
import math
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

from cookiemonster.budget_accountant import BudgetAccountant

REPO_ROOT = Path(__file__).parent.parent
LOGS_PATH = REPO_ROOT.joinpath("logs")
RAY_LOGS = LOGS_PATH.joinpath("ray")


IPA = "ipa"
COOKIEMONSTER = "cookiemonster"
COOKIEMONSTER_BASE = "cookiemonster_base"

BIAS = "bias"
BUDGET = "budget"
FILTERS_STATE = "filters_state"
MLFLOW = "mlflow"


class EpochsWindow:
    def __init__(self) -> None:
        self.epochs = (math.inf, 0)

    def update(self, epochs_window):
        (a, b) = epochs_window
        self.epochs = (
            min(a, self.epochs[0]),
            max(b, self.epochs[1]),
        )

    def get_epochs(self):
        return self.epochs

    def len(self):
        return self.epochs[1] - self.epochs[0] + 1


class GlobalStatistics:
    def __init__(self, baseline) -> None:
        self.baseline = baseline
        self.per_destination_global_stats: Dict[str, Any] = {}

    def update(self, conversion):
        destination = str(conversion.destination)
        epochs_window = conversion.epochs_window
        device_filter = 0 if self.baseline == IPA else conversion.user_id

        if destination not in self.per_destination_global_stats:
            self.per_destination_global_stats[destination] = {
                "epochs_window": EpochsWindow(),
                "unique_device_filters": set(),
            }

        global_stats = self.per_destination_global_stats[destination]
        global_stats["epochs_window"].update(epochs_window)
        global_stats["unique_device_filters"].add(device_filter)

    def dump(self):
        output = {}
        for destination, stats in self.per_destination_global_stats.items():
            output[destination] = {}
            output[destination]["num_unique_device_filters_touched"] = len(
                stats["unique_device_filters"]
            )
            output[destination]["num_epochs_touched"] = stats["epochs_window"].len()
        return output


def maybe_initialize_filters(
    filters_per_origin,
    destination: str,
    attribution_epochs: Tuple[int, int],
    initial_budget: float,
):
    if destination not in filters_per_origin:
        filters_per_origin[destination] = BudgetAccountant(initial_budget)
    destination_filter = filters_per_origin[destination]

    destination_filter.maybe_initialize_filter(attribution_epochs)
    return destination_filter


def get_data_path(path):
    return str(REPO_ROOT.joinpath(f"data/{path}"))


def load_logs(log_path: str, relative_path=True) -> dict:
    full_path = Path(log_path)
    if relative_path:
        full_path = LOGS_PATH.joinpath(log_path)
    with open(full_path, "r") as f:
        logs = json.load(f)
    return logs


def save_logs(log_dict, save_dir):
    log_path = LOGS_PATH.joinpath(save_dir).joinpath(
        f"{datetime.now().strftime('%m%d-%H%M%S')}_{str(uuid.uuid4())[:6]}/result.json"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as fp:
        json_object = json.dumps(log_dict, indent=4)
        fp.write(json_object)
