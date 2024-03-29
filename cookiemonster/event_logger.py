from typing import Dict, Any, Union
from cookiemonster.events import Impression, Conversion
from cookiemonster.budget_accountant import BudgetAccountantResult


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
        filter_result.budget_consumed,
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

    def __add__(self, other: "EventLogger") -> "EventLogger":
        new_logger = EventLogger()
        new_logger.logs = {**self.logs, **other.logs}
        return new_logger
