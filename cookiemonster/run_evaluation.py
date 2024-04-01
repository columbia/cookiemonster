import os
import typer
from loguru import logger
from typing import Dict, Any
from termcolor import colored
from omegaconf import OmegaConf

from cookiemonster.dataset import Dataset
from cookiemonster.query_batch import QueryBatch
from cookiemonster.event_logger import EventLogger
from cookiemonster.budget_accountant import BudgetAccountant
from cookiemonster.aggregation_policy import AggregationPolicy
from cookiemonster.aggregation_service import AggregationService
from cookiemonster.user import User, get_log_events_across_users, ConversionResult
from cookiemonster.utils import (
    process_logs,
    save_logs,
    IPA,
    maybe_initialize_filters,
    compute_global_sensitivity,
    BUDGET,
    QUERY_RESULTS
)


app = typer.Typer()


class Evaluation:
    def __init__(self, config: Dict[str, Any]):
        self.config = OmegaConf.create(config)
        self.dataset = Dataset.create(self.config.dataset)
        self.users: Dict[str, User] = {}

        self.logger = EventLogger()

        # filters shared across users for IPA
        self.global_filters_per_origin: Dict[str, BudgetAccountant] = {}

        self.per_destination_per_query_batch: Dict[str, Dict[str, QueryBatch]] = {}

        self.aggregation_policy = AggregationPolicy.create(
            self.config.aggregation_policy
        )

        self.aggregation_service = AggregationService.create(
            self.config.aggregation_service
        )

    def run(self):
        """Reads events from a dataset and asks users to process them"""
        event_reader = self.dataset.event_reader()
        while res := next(event_reader):
            (user_id, event) = res

            logger.info(colored(str(event), "blue"))

            if user_id not in self.users:
                self.users[user_id] = User(user_id, self.config)

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

                # Compute global sensitivity based on the aggregatable cap value
                global_sensitivity = compute_global_sensitivity(
                    self.config.user.sensitivity_metric, event.aggregatable_cap_value
                )
                # Support for only scalar reports for now
                assert len(report.histogram) == 1

                for query_id, value in report.histogram.items():
                    if query_id not in per_query_batch:
                        per_query_batch[query_id] = QueryBatch(
                            query_id, event.epsilon, global_sensitivity
                        )
                    else:
                        # All reports for the same query should have the same global epsilon and sensitivity
                        assert per_query_batch[query_id].global_epsilon == event.epsilon
                        assert (
                            per_query_batch[query_id].global_sensitivity
                            == global_sensitivity
                        )

                    per_query_batch[query_id].update(
                        value, unbiased_report.histogram[query_id], event.epochs_window
                    )

                # Check if the new report triggers scheduling / aggregation
                for query_id in report.histogram.keys():
                    batch = per_query_batch[query_id]
                    if self.aggregation_policy.should_calculate_summary_reports(batch):
                        self._calculate_summary_reports(
                            query_id=query_id,
                            batch=batch,
                            id=event.id,
                            destination=event.destination,
                        )

                        # Reset the batch
                        del per_query_batch[query_id]

        # handle the tail for those queries that have enough events for DP, but are not
        # the preferred batch size
        for (
            destination,
            per_query_batch,
        ) in self.per_destination_per_query_batch.values():
            for query_id, batch in per_query_batch.values():
                if self.aggregation_policy.should_calculate_summary_reports(
                    batch, tail=True
                ):
                    self._calculate_summary_reports(
                        query_id=query_id,
                        batch=batch,
                        id=-1,
                        destination=destination,
                    )

        merged_event_loggers = self.logger + get_log_events_across_users()
        logs = process_logs(
            merged_event_loggers.logs,
            OmegaConf.to_object(self.config),
        )
        if self.config.logs.save:
            save_dir = self.config.logs.save_dir if self.config.logs.save_dir else ""
            save_logs(logs, save_dir)

        return logs

    def _calculate_summary_reports(
        self, *, query_id: str, batch: QueryBatch, id: int, destination: str
    ) -> None:

        self.logger.log("scheduling_timestamps", id)
        self.logger.log("epoch_range", destination, *batch.epochs_window)

        # In case of IPA the advertiser consumes worst-case budget from all the
        # requested epochs in their global filter (Central DP)
        if self.config.user.baseline == IPA:
            origin_filters = maybe_initialize_filters(
                self.global_filters_per_origin,
                destination,
                batch.epochs_window,
                float(self.config.user.initial_budget),
            )
            filter_result = origin_filters.pay_all_or_nothing(
                batch.epochs_window, batch.global_epsilon
            )
            if BUDGET in self.config.logs.logging_keys:
                self.logger.log(
                    BUDGET,
                    id,
                    destination,
                    0,
                    batch.epochs_window,
                    filter_result.budget_consumed,
                    filter_result.status,
                )

            if not filter_result.succeeded():
                # Not enough budget to run this query - don't schedule the batch
                if QUERY_RESULTS in self.config.logs.logging_keys:
                    self.logger.log(
                        QUERY_RESULTS,
                        id,
                        destination,
                        query_id,
                        None,
                        None,
                        None,
                    )
                logger.info(colored(f"IPA can't run query", "red"))
                return

        # Schedule the batch
        aggregation_result = self.aggregation_service.create_summary_report(batch)
        logger.info(colored(f"Scheduling query batch {query_id}", "green"))
        logger.info(
            colored(
                f"true_output: {aggregation_result.true_output}, aggregation_output: {aggregation_result.aggregation_output}, aggregation_noisy_output: {aggregation_result.aggregation_noisy_output}",
                "green",
            )
        )

        if QUERY_RESULTS in self.config.logs.logging_keys:
            self.logger.log(
                QUERY_RESULTS,
                id,
                destination,
                query_id,
                aggregation_result.true_output,
                aggregation_result.aggregation_output,
                aggregation_result.aggregation_noisy_output,
            )


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
