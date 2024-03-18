from loguru import logger
from omegaconf import OmegaConf
import os
from termcolor import colored
import typer
from typing import Dict, Any, List

from systemx.user import User
from systemx.report import Report
from systemx.dataset import Dataset
from systemx.utils import process_logs, save_logs
from systemx.aggregation_service import AggregationService
from systemx.aggregation_policy import AggregationPolicy

app = typer.Typer()


class Evaluation:
    def __init__(self, config: Dict[str, Any]):
        self.config = OmegaConf.create(config)
        self.dataset = Dataset.create(self.config.dataset)

        self.users: Dict[str, User] = {}
        self.reports: Dict[str, List[Report]] = {}
        self.summary_reports: List[Dict[str, Dict[str, float]]] = []
        self.aggregation_service = AggregationService.create(self.config.aggregation_service)
        self.aggregation_policy = AggregationPolicy.create(self.config.aggregation_policy)

    def run(self):
        """Reads events from a dataset and asks users to process them"""

        count = 0
        event_reader = self.dataset.event_reader()
        while res := next(event_reader):
            count += 1
            (user_id, event) = res

            logger.info(colored(str(event), "blue"))

            if user_id not in self.users:
                self.users[user_id] = User(user_id, self.config.user)

            report = self.users[user_id].process_event(event)

            if report:
                if event.destination not in self.reports:
                    self.reports[event.destination] = []
                self.reports[event.destination].append(report)

            if self.aggregation_policy.should_calculate_summary_reports(event):
                self.summary_reports.append(self.aggregation_service.create_summary_reports(self.reports))
                self.reports = {}

        # Collects budget consumption per user per destination epoch
        logs = process_logs(
            [user.get_logs() for user in self.users.values()],
            OmegaConf.to_object(self.config),
        )
        if self.config.logs.save:
            save_dir = self.config.logs.save_dir if self.config.logs.save_dir else ""
            save_logs(logs, save_dir)

        return logs


@app.command()
def run_evaluation(
    omegaconf: str = "config/config.json",
    loguru_level: str = "INFO",
):
    os.environ["LOGURU_LEVEL"] = loguru_level

    omegaconf = OmegaConf.load(omegaconf)
    return Evaluation(omegaconf).run()


if __name__ == "__main__":
    app()
