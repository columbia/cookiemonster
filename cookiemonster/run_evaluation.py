import os
import typer
from loguru import logger
from termcolor import colored
from omegaconf import OmegaConf
from typing import Dict, Any, List

from cookiemonster.dataset import Dataset
from cookiemonster.query_batch import QueryBatch
from cookiemonster.event_logger import EventLogger
from cookiemonster.user import User, ConversionResult
from cookiemonster.budget_accountant import BudgetAccountant
from cookiemonster.aggregation_policy import AggregationPolicy
from cookiemonster.aggregation_service import AggregationService
from cookiemonster.utils import (
    GlobalStatistics,
    IPA,
    BIAS,
    BUDGET,
    FILTERS_STATE,
    save_logs,
    maybe_initialize_filters,
    compute_global_sensitivity,
)

app = typer.Typer()


class Evaluation:
    def __init__(self, config: Dict[str, Any]):
        self.config = OmegaConf.create(config)
        self.dataset = Dataset.create(self.config.dataset)
        self.users: Dict[str, User] = {}

        self.logger = EventLogger()
        self.global_statistics = GlobalStatistics(self.config.user.baseline)

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
        for i, res in enumerate(self.dataset.event_reader()):
            (user_id, event) = res

            if i % 100000 == 0:
                logger.info(colored(str(event), "blue"))

            if user_id not in self.users:
                self.users[user_id] = User(user_id, self.config)

            result = self.users[user_id].process_event(event)

            if isinstance(result, ConversionResult):
                self.global_statistics.update(event)

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
                            query_id,
                            event.epsilon,
                            global_sensitivity,
                            biggest_id=event.id,
                        )
                    else:
                        # All conversion requests for the same query should have the same epsilon and sensitivity
                        assert per_query_batch[query_id].global_epsilon == event.epsilon
                        assert (
                            per_query_batch[query_id].global_sensitivity
                            == global_sensitivity
                        )

                    per_query_batch[query_id].update(
                        value,
                        unbiased_report.histogram[query_id],
                        event.epochs_window,
                        biggest_id=event.id,
                    )

                # Check if the new report triggers scheduling / aggregation
                for query_id in report.histogram.keys():
                    batch = per_query_batch[query_id]
                    if self.aggregation_policy.should_calculate_summary_reports(batch):
                        self._calculate_summary_reports(
                            query_id=query_id,
                            batch=batch,
                            destination=event.destination,
                        )

                        # Reset the batch
                        del per_query_batch[query_id]

        # Handle the tail for those queries that have enough events for DP, but are not the preferred batch size
        for (
            destination,
            per_query_batch,
        ) in self.per_destination_per_query_batch.items():
            for query_id, batch in per_query_batch.items():
                if self.aggregation_policy.should_calculate_summary_reports(
                    batch, tail=True
                ):
                    self._calculate_summary_reports(
                        query_id=query_id,
                        batch=batch,
                        destination=destination,
                    )

        self._log_all_filters_state()

        logs = {}
        logs["global_statistics"] = self.global_statistics.dump()
        logs["event_logs"] = self.logger.logs
        logs["config"] = OmegaConf.to_object(self.config)
        if self.config.logs.save:
            save_dir = self.config.logs.save_dir if self.config.logs.save_dir else ""
            save_logs(logs, save_dir)

        return logs

    def _try_consume_budget_for_ipa(self, destination, batch):
        # In case of IPA the advertiser consumes worst-case budget from all the requested epochs in their global filter (Central DP)
        if self.config.user.baseline == IPA:
            origin_filters = maybe_initialize_filters(
                self.global_filters_per_origin,
                destination,
                batch.epochs_window.get_epochs(),
                float(self.config.user.initial_budget),
            )
            filter_result = origin_filters.pay_all_or_nothing(
                batch.epochs_window.get_epochs(), batch.global_epsilon
            )
            if not filter_result.succeeded():
                logger.info(colored(f"IPA can't run query", "red"))
                return False
        return True

    def _calculate_summary_reports(
        self, *, query_id: str, batch: QueryBatch, destination: str
    ) -> None:

        true_output = None
        aggregation_output = None
        aggregation_noisy_output = None

        status = True
        if self.config.user.baseline == IPA:
            status = self._try_consume_budget_for_ipa(destination, batch)

        if status:
            # Schedule the batch
            aggregation_result = self.aggregation_service.create_summary_report(batch)
            true_output = aggregation_result.true_output
            aggregation_output = aggregation_result.aggregation_output
            aggregation_noisy_output = aggregation_result.aggregation_noisy_output
            logger.info(
                colored(
                    f"Scheduling query batch {query_id}, true_output: {true_output}, aggregation_output: {aggregation_output}, noisy_output: {aggregation_noisy_output}",
                    "green",
                )
            )

        # Log budgeting metrics and accuracy related data
        if BUDGET in self.config.logs.logging_keys:
            budget_metrics = {"max_max": 0, "sum_max": 0, "max_sum": 0, "sum_sum": 0}

            def get_budget_metrics(filters_per_origin):
                if destination in filters_per_origin:
                    budget_accountant = filters_per_origin[destination]
                    max_ = budget_accountant.get_max_consumption_across_blocks()
                    sum_ = budget_accountant.get_sum_consumption_across_blocks()

                    budget_metrics["max_max"] = max(budget_metrics["max_max"], max_)
                    budget_metrics["max_sum"] = max(budget_metrics["max_sum"], sum_)
                    budget_metrics["sum_max"] += max_
                    budget_metrics["sum_sum"] += sum_

            if self.global_filters_per_origin:
                get_budget_metrics(self.global_filters_per_origin)
            else:
                for user in self.users.values():
                    get_budget_metrics(user.filters_per_origin)

            self.logger.log(BUDGET, destination, batch.biggest_id, budget_metrics)

        if BIAS in self.config.logs.logging_keys:
            self.logger.log(
                BIAS,
                batch.biggest_id,
                destination,
                query_id,
                batch.global_epsilon,
                batch.global_sensitivity,
                {
                    "true_output": true_output,
                    "aggregation_output": aggregation_output,
                    "aggregation_noisy_output": aggregation_noisy_output,
                },
            )

    def _log_all_filters_state(self):
        if BUDGET in self.config.logs.logging_keys:
            per_destination_device_epoch_filters: Dict[str, List[float]] = {}

            def get_filters_state(filters_per_origin):
                # Log filters state for each destination
                for destination, budget_accountant in filters_per_origin.items():
                    if destination not in per_destination_device_epoch_filters:
                        per_destination_device_epoch_filters[destination] = []

                    avg_consumed_budget = (
                        budget_accountant.get_avg_consumption_across_blocks()
                    )
                    per_destination_device_epoch_filters[destination].append(
                        avg_consumed_budget
                    )

            if self.global_filters_per_origin:
                get_filters_state(self.global_filters_per_origin)
            else:
                for user in self.users.values():
                    get_filters_state(user.filters_per_origin)

            self.logger.log(FILTERS_STATE, per_destination_device_epoch_filters)


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
