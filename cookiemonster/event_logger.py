from typing import Dict, Any


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
