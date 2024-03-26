import math
from typing import Dict, Any, Union
from systemx.events import Impression, Conversion
from systemx.budget_accountant import BudgetAccountantResult


def log_budget_helper(
    logger: "EventLogger",
    event: Union[Impression, Conversion],
    user_id: Any,
    filter_result: BudgetAccountantResult,
):
    logger.log(
        "budget",
        event.id,
        event.destination,
        user_id,
        event.epochs_window,
        event.attribution_window,
        filter_result.total_budget_consumed,
        filter_result.status,
    )


class EventLogger:
    def __init__(self):
        # TODO: Possibly add support for mlflow logging
        self.logs: Dict[str, Any] = {}

    def log(self, key, *data):
        if key not in self.logs:
            self.logs[key] = []
        self.logs[key].append(data)

    def log_range(self, key, destination, a, b):
        if key not in self.logs:
            self.logs[key] = {}
        if destination not in self.logs[key]:
            self.logs[key][destination] = {"min": math.inf, "max": 0}
        logs_key_destination = self.logs[key][destination]
        logs_key_destination["min"] = min(logs_key_destination["min"], a)
        logs_key_destination["max"] = max(logs_key_destination["max"], b)

    def __add__(self, other: "EventLogger") -> "EventLogger":
        new_logger = EventLogger()
        new_logger.logs = {**self.logs, **other.logs}
        return new_logger
