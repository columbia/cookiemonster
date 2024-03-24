from loguru import logger
from omegaconf import OmegaConf
import os
from termcolor import colored
import typer
from typing import Dict, Any, List

from systemx.report import Report
from systemx.dataset import Dataset
from systemx.utils import process_logs, save_logs
from systemx.aggregation_service import AggregationService
from systemx.aggregation_policy import AggregationPolicy
from systemx.user import User, get_logs_across_users

app = typer.Typer()


class QueryBatch:
    def __init__(self, query_id, epsilon) -> None:
        self.query_id = query_id
        self.epsilon = epsilon
        self.reports = []


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

        event_reader = self.dataset.event_reader()
        while res := next(event_reader):
            (user_id, event) = res

            logger.info(colored(str(event), "blue"))

            if user_id not in self.users:
                self.users[user_id] = User(user_id, self.config.user)

            report = self.users[user_id].process_event(event)

            if report:
                assert not report.empty()
                if event.destination not in self.reports:
                    self.reports[event.destination] = {}

                assert len(report.histogram.keys()) == 1
                query_id = list(report.histogram.keys())[0]

                if query_id not in self.reports[event.destination]:
                    self.reports[event.destination][query_id] = []

                self.reports[event.destination][query_id].append(report)

            if self.aggregation_policy.should_calculate_summary_reports(event):
                self.summary_reports.append(self.aggregation_service.create_summary_reports(self.reports))
                self.reports = {}

        # End of reports - batch

        # Collects budget consumption per user per destination epoch
        logs = process_logs(
            get_logs_across_users(),
            OmegaConf.to_object(self.config),
        )
        if self.config.logs.save:
            save_dir = self.config.logs.save_dir if self.config.logs.save_dir else ""
            save_logs(logs, save_dir)

        return logs


@app.command()
def run_evaluation(
    omegaconf: str = "systemx/config/config.json",
    loguru_level: str = "INFO",
):
    os.environ["LOGURU_LEVEL"] = loguru_level

    omegaconf = OmegaConf.load(omegaconf)
    return Evaluation(omegaconf).run()

if __name__ == "__main__":
    app()
