import os
from typing import Any, Dict, List

import mlflow
import numpy as np
import typer
from coolname import generate_slug
from loguru import logger
from omegaconf import OmegaConf
from ray.air.integrations.mlflow import setup_mlflow
from termcolor import colored

from cookiemonster.aggregation_policy import AggregationPolicy
from cookiemonster.aggregation_service import AggregationService
from cookiemonster.attribution import LastTouch, LastTouchWithCount
from cookiemonster.budget_accountant import BudgetAccountant
from cookiemonster.dataset import Dataset
from cookiemonster.event_logger import EventLogger
from cookiemonster.query_batch import QueryBatch
from cookiemonster.user import ConversionResult, User
from cookiemonster.utils import (BIAS, BUDGET, FILTERS_STATE, IPA, LOGS_PATH,
                                 MLFLOW, GlobalStatistics,
                                 compute_global_sensitivity,
                                 maybe_initialize_filters, save_logs)

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

    def setup_mlfow(self):
        mlflow.set_tracking_uri(LOGS_PATH.joinpath("mlflow"))

        exp_name = self.config.logs.experiment_name
        try:
            experiment_id = mlflow.create_experiment(name=exp_name)
        except mlflow.exceptions.MlflowException:
            experiment_id = mlflow.get_experiment_by_name(exp_name).experiment_id

        run_name = (
            self.config.logs.trial_name
            if self.config.logs.trial_name
            else generate_slug(2)
        )
        mlflow.start_run(run_name=run_name, experiment_id=experiment_id)
        mlflow.log_params(OmegaConf.to_object(self.config))

    def run(self):
        """Reads events from a dataset and asks users to process them"""

        if MLFLOW in self.config.logs.logging_keys:
            self.setup_mlfow()

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

                # TODO: do we really need this?
                assert not report.empty() and not unbiased_report.empty()

                if event.destination not in self.per_destination_per_query_batch:
                    self.per_destination_per_query_batch[event.destination] = {}

                per_query_batch = self.per_destination_per_query_batch[
                    event.destination
                ]

                # We only support single-query reports for now, with potentially multiple buckets
                # At aggregation time you need to know how many buckets you are interested in
                # Otherwise, the absence or presence of a bucket can leak info about a single record
                # Maybe one day add support for arbitrary buckets too, e.g. for histogram queries?

                # TODO(P1): store this in the report itself
                global_sensitivity = event.aggregatable_cap_value

                # if self.config.user.bias_detection_knob:
                #     # impression_buckets = ["empty", "main"]
                #     # Global L1 sensitivity, removing a record can either remove kappa from "empty", or at most cap from "main"
                #     # (but not both at the same time)
                #     # assert self.config.user.sensitivity_metric == "L1"
                #     # global_sensitivity = max(
                #     #     event.aggregatable_cap_value,
                #     #     self.config.user.bias_detection_knob,
                #     # )

                #     attribution_function = LastTouchWithCount(
                #         sensitivity_metric=self.config.user.sensitivity_metric,
                #         attribution_cap=event.aggregatable_cap_value,
                #         kappa=self.config.user.bias_detection_knob,
                #     )

                # else:
                #     attribution_function = LastTouchWithCount(
                #         sensitivity_metric=self.config.user.sensitivity_metric,
                #         attribution_cap=event.aggregatable_cap_value,
                #     )

                #     # impression_buckets = [""]
                #     # # Compute global sensitivity based on the aggregatable cap value
                #     # global_sensitivity = compute_global_sensitivity(
                #     #     self.config.user.sensitivity_metric,
                #     #     event.aggregatable_cap_value,
                #     # )

                # global_sensitivity = attribution_function.compute_global_sensitivity()

                # query_id, value = report.get_query_value(impression_buckets)
                # _, unbiased_value = unbiased_report.get_query_value(impression_buckets)

                query_id = report.get_query_id()
                value = report.get_value()
                unbiased_value = unbiased_report.get_value()

                # for query_id, value in report.histogram.items():
                if query_id not in per_query_batch:
                    per_query_batch[query_id] = QueryBatch(
                        query_id,
                        event.epsilon,
                        global_sensitivity,  # TODO: allow one sensitivity per report in the batch
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
                    unbiased_value,
                    event.epochs_window,
                    biggest_id=event.id,
                )

                # Check if the new report triggers scheduling / aggregation
                # for query_id in report.histogram.keys():
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

        print(logs["global_statistics"])

        # mlflow.log_metrics(logs["global_statistics"])

        return logs

    def _try_consume_budget_for_ipa(self, destination, batch):

        # TODO: technically we should take the max of attributed values across reports in a query.

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
        # TODO: update logs to handle vector outputs

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

            if MLFLOW in self.config.logs.logging_keys:

                if self.config.user.bias_detection_knob:
                    # The output is a vector of size 2

                    kappa = self.config.user.bias_detection_knob

                    mlflow.log_metrics(
                        {
                            f"true_output": true_output[1],
                            f"aggregation_output": aggregation_output[1],
                            f"aggregation_noisy_output": aggregation_noisy_output[1],
                            f"true_bias": aggregation_output[1] - true_output[1],
                            "true_empty_epochs": true_output[0] / kappa,
                            "noisy_empty_epochs": aggregation_noisy_output[0] / kappa,
                            "noisy_bias_bound": (aggregation_noisy_output[0] + batch.noise_scale * np.log(1/0.05) / np.sqrt(2)  ) * (batch.global_sensitivity / kappa),
                        }
                    )

                elif isinstance(true_output, np.ndarray):
                    # You probably don't want to log this, but who knows

                    for i in range(len(true_output)):
                        mlflow.log_metrics(
                            {
                                f"true_output_{i}": true_output[i],
                                f"aggregation_output_{i}": aggregation_output[i],
                                f"aggregation_noisy_output_{i}": aggregation_noisy_output[
                                    i
                                ],
                            }
                        )

                else:
                    mlflow.log_metrics(
                        {
                            "true_output": true_output,
                            "aggregation_output": aggregation_output,
                            "aggregation_noisy_output": aggregation_noisy_output,
                        }
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

    config = OmegaConf.load(omegaconf)
    return Evaluation(config).run()


if __name__ == "__main__":
    app()
