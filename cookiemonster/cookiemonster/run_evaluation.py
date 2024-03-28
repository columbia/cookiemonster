import os
import math
import typer
from loguru import logger
from typing import Dict, Any
from termcolor import colored
from omegaconf import OmegaConf

from cookiemonster.dataset import Dataset
from cookiemonster.event_logger import EventLogger
from cookiemonster.budget_accountant import BudgetAccountant
from cookiemonster.user import User, get_log_events_across_users, ConversionResult
from cookiemonster.utils import process_logs, save_logs, IPA, maybe_initialize_filters

app = typer.Typer()


class QueryBatch:
    def __init__(self, query_id, epsilon) -> None:
        self.epochs_window = (math.inf, 0)
        self.query_id = query_id
        self.epsilon = epsilon
        self.values = []
        self.unbiased_values = []

    def size(self):
        return len(self.values)

    def upate_epochs_window(self, epochs_window):
        (a, b) = epochs_window
        self.epochs_window = (
            min(a, self.epochs_window[0]),
            max(b, self.epochs_window[1]),
        )


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

                    batch = per_query_batch[query_id]
                    batch.values.append(value)
                    batch.unbiased_values.append(unbiased_report.histogram[query_id])
                    batch.upate_epochs_window(event.epochs_window)

                # Check if the new report triggers scheduling / aggregation
                for query_id in report.histogram.keys():
                    batch = per_query_batch[query_id]

                    if batch.size() == self.config.scheduling_batch_size_per_query:

                        self.logger.log("scheduling_timestamps", event.id)
                        self.logger.log(
                            "epoch_range", event.destination, *batch.epochs_window
                        )

                        # In case of IPA the advertiser consumes worst-case budget from all the requested epochs in their global filter (Central DP)
                        if self.config.user.baseline == IPA:
                            origin_filters = maybe_initialize_filters(
                                self.global_filters_per_origin,
                                event.destination,
                                batch.epochs_window,
                                float(self.config.user.initial_budget),
                            )
                            filter_result = origin_filters.pay_all_or_nothing(
                                batch.epochs_window, batch.epsilon
                            )
                            self.logger.log(
                                "budget",
                                event.id,
                                event.destination,
                                0,
                                batch.epochs_window,
                                filter_result.budget_consumed,
                                filter_result.status,
                            )

                            if not filter_result.succeeded():
                                # Not enough budget to run this query - don't schedule the batch
                                self.logger.log(
                                    "bias",
                                    event.id,
                                    event.destination,
                                    query_id,
                                    math.inf,
                                )
                                logger.info(colored(f"Query bias: {math.inf}", "red"))
                                # Reset the batch
                                del per_query_batch[query_id]
                                continue

                        # Schedule the batch
                        # TODO: move this to aggregation service
                        logger.info(
                            colored(f"Scheduling query batch {query_id}", "green")
                        )
                        query_output = sum(batch.values)
                        unbiased_query_output = sum(batch.unbiased_values)
                        bias = abs(query_output - unbiased_query_output)
                        logger.info(
                            colored(
                                f"Query bias: {bias}, true output: {query_output}",
                                "green",
                            )
                        )

                        self.logger.log(
                            "bias", event.id, event.destination, query_id, bias
                        )

                        # Reset the batch
                        del per_query_batch[query_id]

        merged_event_loggers = self.logger + get_log_events_across_users()
        logs = process_logs(
            merged_event_loggers.logs,
            OmegaConf.to_object(self.config),
        )
        if self.config.logs.save:
            save_dir = self.config.logs.save_dir if self.config.logs.save_dir else ""
            save_logs(logs, save_dir)

        return logs


@app.command()
def run_evaluation(
    omegaconf: str = "cookiemonster/config/config.json",
    loguru_level: str = "INFO",
):
    os.environ["LOGURU_LEVEL"] = loguru_level

    omegaconf = OmegaConf.load(omegaconf)
    return Evaluation(omegaconf).run()


if __name__ == "__main__":
    app()
