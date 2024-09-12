from typing import Any, Dict


class EventLogger:
    def __init__(self):
        self.logs: Dict[str, Any] = {}

    def log(self, key, *data):
        if key not in self.logs:
            self.logs[key] = []
        self.logs[key].append(data)

    def log_one(self, key, data):
        # Avoid building a tuple of length 1 when we just want to log one element
        if key not in self.logs:
            self.logs[key] = []
        self.logs[key].append(data)

    def store(self, key, data):
        self.logs[key] = data

    def get(self, key):
        return self.logs.get(key, None)

    def __add__(self, other: "EventLogger") -> "EventLogger":
        new_logger = EventLogger()
        new_logger.logs = {**self.logs, **other.logs}
        return new_logger
