import os
import typer
from loguru import logger
from typing import Dict, Any, List
from omegaconf import OmegaConf

from termcolor import colored
from user import User
from report import Report
from dataset import Dataset

app = typer.Typer()


class Evaluation:
    def __init__(self, config: Dict[str, Any]):
        self.config = OmegaConf.create(config)
        self.dataset = Dataset.create(self.config.dataset)

        self.users: Dict[str, User] = {}
        self.reports: Dict[str, List[Report]] = {}

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
                if event.destination not in self.reports:
                    self.reports[event.destination] = []
                self.reports[event.destination].append(report)

            # TODO: add possibly with simpy another process per destination
            # that receives reports, categorizes them per query and schedules
            # sending them to TEE for aggregation


@app.command()
def run_evaluation(
    omegaconf: str = "evaluation/config/config.json",
    loguru_level: str = "INFO",
):
    os.environ["LOGURU_LEVEL"] = loguru_level

    omegaconf = OmegaConf.load(omegaconf)
    logs = Evaluation(omegaconf).run()
    return logs


if __name__ == "__main__":
    app()
