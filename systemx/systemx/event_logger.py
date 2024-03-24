from typing import Dict, Any, Union, Tuple
from systemx.events import Impression, Conversion
from systemx.budget_accountant import BudgetAccountantResult


class EventLogger:
    def __init__(self):
        # TODO: Possibly add support for mlflow logging
        self.logs: Dict[str, Any] = {"budget": {}, "bias": {}}

    def log_event_budget_internal(
        self,
        timestamp: int,
        destination: str,
        user_id: int,
        epochs_window: Tuple[int, int],
        attribution_window: Tuple[int, int],
        total_budget_consumed: float,
        status: str,
    ):
        logs = self.logs["budget"]

        if destination not in logs:
            logs[destination] = []

        logs[destination].append(
            {
                "timestamp": timestamp,
                "user_id": user_id,
                "epochs_window": epochs_window,
                "attribution_window": attribution_window,
                "total_budget_consumed": total_budget_consumed,
                "status": status,
            }
        )

    def log_event_budget(
        self,
        event: Union[Impression, Conversion],
        user_id: Any,
        filter_result: BudgetAccountantResult,
    ):
        self.log_event_budget_internal(
            event.timestamp,
            event.destination,
            user_id,
            event.epochs_window,
            event.attribution_window,
            filter_result.total_budget_consumed,
            filter_result.status,
        )


    def log_event_bias(self, timestamp: int, destination: str, query_id: str, bias: float):
        logs = self.logs["bias"]

        if destination not in logs:
            logs[destination] = []

        logs[destination].append(
            {"timestamp": timestamp, "query_id": query_id, "bias": bias}
        )

    def __add__(self, other: "EventLogger") -> "EventLogger":
        new_logger = EventLogger()
        new_logger.logs = {**self.logs, **other.logs}
        return new_logger
