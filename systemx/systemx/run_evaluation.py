import os
import math
import typer
from loguru import logger
from typing import Dict, Any
from termcolor import colored
from omegaconf import OmegaConf

from systemx.dataset import Dataset
from systemx.event_logger import EventLogger
from systemx.budget_accountant import BudgetAccountant
from systemx.user import User, get_log_events_across_users, ConversionResult
from systemx.utils import process_logs, save_logs, IPA, maybe_initialize_filters

app = typer.Typer()


class QueryBatch:
    def __init__(self, query_id, epsilon) -> None:
        self.query_id = query_id
        self.epsilon = epsilon
        self.values = []
        self.unbiased_values = []

    def size(self):
        return len(self.values)


class Evaluation:
    def __init__(self, config: Dict[str, Any]):
        self.config = OmegaConf.create(config)
        self.dataset = Dataset.create(self.config.dataset)
        self.users: Dict[str, User] = {}

        self.logger = EventLogger()

        # filters shared across users for IPA
        self.global_filters_per_origin: Dict[str, BudgetAccountant] = {}

        self.per_destination_per_query_batch: Dict[str, Dict[str, QueryBatch]] = {}

    def run(self):
        """Reads events from a dataset and asks users to process them"""

        event_reader = self.dataset.event_reader()
        while res := next(event_reader):
            (user_id, event) = res

            logger.info(colored(str(event), "blue"))

            if user_id not in self.users:
                self.users[user_id] = User(user_id, self.config.user)

            result = self.users[user_id].process_event(event)

            if isinstance(result, ConversionResult):
                # Add report to its corresponding batch
                report = result.final_report
                unbiased_report = result.unbiased_final_report
                assert not report.empty() and not unbiased_report.empty()

                if event.destination not in self.per_destination_per_query_batch:
                    self.per_destination_per_query_batch[event.destination] = {}

                per_query_batch = self.per_destination_per_query_batch[
                    event.destination
                ]

                for query_id, value in report.histogram.items():
                    if query_id not in per_query_batch:
                        per_query_batch[query_id] = QueryBatch(query_id, event.epsilon)
                    else:
                        # All reports for the same query should have the same global epsilon
                        assert per_query_batch[query_id].epsilon == event.epsilon
                    per_query_batch[query_id].values.append(value)
                    per_query_batch[query_id].unbiased_values.append(
                        unbiased_report.histogram[query_id]
                    )

                # Check if the new report triggers scheduling / aggregation
                for query_id in report.histogram.keys():
                    batch = per_query_batch[query_id]

                    if batch.size() == self.config.scheduling_batch_size_per_query:
                        # In case of IPA the advertiser consumes worst-case budget from all the requested epochs in their global filter (Central DP)
                        if self.config.baseline == IPA:
                            origin_filters = maybe_initialize_filters(
                                self.global_filters_per_origin,
                                event.destination,
                                event.epochs_window,
                                float(self.config.initial_budget),
                            )
                            filter_result = origin_filters.pay_all_or_nothing(
                                event.epochs_window, event.epsilon
                            )
                            if not filter_result.succeeded():
                                # Not enough budget to run this query - don't schedule the batch
                                self.logger.log_event(
                                    event.timestamp,
                                    event.destination,
                                    query_id,
                                    math.inf,
                                )
                                logger.info(colored(f"Query bias: {bias}", "red"))
                                # # Reset the batch
                                # del per_query_batch[query_id]
                                continue

                        # Schedule the batch
                        # TODO: move this to aggregation service
                        logger.info(
                            colored(f"Scheduling query batch {query_id}", "green")
                        )
                        query_output = sum(batch.values)
                        unbiased_query_output = sum(batch.unbiased_values)
                        bias = math.abs(query_output - unbiased_query_output)
                        logger.info(
                            colored(
                                f"Query bias: {bias}, true output: {query_output}",
                                "green",
                            )
                        )
                        self.logger.log_event(
                            event.timestamp, event.destination, query_id, bias
                        )

                        # Reset the batch
                        del per_query_batch[query_id]

        logs = process_logs(
            self.logger.logs + get_log_events_across_users(),
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
