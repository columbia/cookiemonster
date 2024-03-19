import os
import typer
from loguru import logger
from typing import Dict, Any, List
from termcolor import colored
from omegaconf import OmegaConf

from systemx.report import Report
from systemx.dataset import Dataset
from systemx.utils import process_logs, save_logs
from systemx.user import User, get_logs_across_users

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
