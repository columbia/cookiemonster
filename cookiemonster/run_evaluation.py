import os
from typing import Any, Dict, List

import mlflow
import numpy as np
import typer
from coolname import generate_slug
from loguru import logger
from omegaconf import OmegaConf
from termcolor import colored

from cookiemonster.aggregation_policy import AggregationPolicy
from cookiemonster.aggregation_service import AggregationService
from cookiemonster.bias import (aggregate_bias_prediction_metrics,
                                compute_base_bias_metrics,
                                compute_bias_metrics,
                                compute_bias_prediction_metrics,
                                predict_rmsre_naive)
from cookiemonster.budget_accountant import BudgetAccountant
from cookiemonster.dataset import Dataset
from cookiemonster.event_logger import EventLogger
from cookiemonster.query_batch import QueryBatch
from cookiemonster.user import ConversionResult, User
from cookiemonster.utils import (BIAS, BUDGET, FILTERS_STATE, IPA, LOGS_PATH,
                                 MLFLOW, GlobalStatistics,
                                 maybe_initialize_filters, save_logs)

app = typer.Typer()


class Evaluation:
    def __init__(self, config: Dict[str, Any]):
        self.config = OmegaConf.create(config)
        self.dataset = Dataset.create(self.config.dataset)
        self.users: Dict[str, User] = {}

        self.logger = EventLogger()
        self.global_statistics = GlobalStatistics(self.config.user.baseline)

        self.num_queries_answered = 0

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

        for k, v in OmegaConf.to_object(self.config).items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    mlflow.log_param(f"{k}.{k2}", v2)
            else:
                mlflow.log_param(k, v)

    def run(self):
        """Reads events from a dataset and asks users to process them"""

        if MLFLOW in self.config.logs.logging_keys:
            self.setup_mlfow()

        for i, res in enumerate(self.dataset.event_reader()):
            (user_id, event) = res

            if self.num_queries_answered > self.dataset.workload_size:
                logger.info(
                    f"Reached workload size {self.dataset.workload_size} at event {i}"
                )
                break

            if i % 100_000 == 0:
                logger.info(f"Event {i}: {event}")

            if user_id not in self.users:
                self.users[user_id] = User(user_id, self.config)

            # Potentially generate report and spend individual budget
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

                query_id = report.get_query_id()
                value = report.get_value()
                unbiased_value = unbiased_report.get_value()
                global_sensitivity = (
                    report.global_sensitivity
                )  # Computed by the attribution function
                
                # logger.info(f"Event. Value: {value}, Unbiased Value: {unbiased_value}")

                # Create at most one query per histogram (we don't support multi-query histograms yet)
                if query_id not in per_query_batch:
                    per_query_batch[query_id] = QueryBatch(
                        query_id=query_id,
                        noise_scale=event.noise_scale,
                        biggest_id=event.id,
                    )
                else:
                    assert (
                        per_query_batch[query_id].noise_scale == event.noise_scale
                    ), f"Different noise scales. {per_query_batch[query_id].noise_scale} != {event.noise_scale}. We can handle it by requiring >=, or by creating a new query batch every time we find a different noise scale."

                per_query_batch[query_id].add_report(
                    value,
                    unbiased_value,
                    global_sensitivity,
                    event.epochs_window,
                    biggest_id=event.id,
                )

                # Check if the new report triggers scheduling / aggregation
                batch = per_query_batch[query_id]

                if i % 100_000 == 0:
                    logger.info(
                        f"Batch at event {i}: query {query_id}, sum {sum(batch.values)}, true sum {sum(batch.unbiased_values)}"
                    )

                if self.aggregation_policy.should_calculate_summary_reports(batch):
                    query_result = self._calculate_summary_reports(
                        query_id=query_id,
                        batch=batch,
                        destination=event.destination,
                    )
                    self._log_query_result(
                        query_id, batch, event.destination, query_result
                    )

                    # Reset the batch
                    del per_query_batch[query_id]

        # Handle the tail for those queries that have enough events for DP, but are not the preferred batch size
        for (
            destination,
            per_query_batch,
        ) in self.per_destination_per_query_batch.items():
            for query_id, batch in per_query_batch.items():

                # Use the min_batch_size thanks to `tail=True`
                if self.aggregation_policy.should_calculate_summary_reports(
                    batch, tail=True
                ):
                    query_result = self._calculate_summary_reports(
                        query_id=query_id,
                        batch=batch,
                        destination=destination,
                    )
                    self._log_query_result(query_id, batch, destination, query_result)

        logs = self._finalize_logs()

        print(logs["global_statistics"])

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
            batch_window = batch.epochs_window.get_epochs()
            global_epsilon = (
                batch.get_global_epsilon()
            )  # Uses the max global sensitivity over report values
            logger.info(f"IPA budget: {global_epsilon} for window {batch_window}")

            # TODO: is this the tightest parallel composition we can do? Can we only spend from certain epochs?
            filter_result = origin_filters.pay_all_or_nothing(
                batch_window, global_epsilon
            )
            if not filter_result.succeeded():
                logger.info(colored(f"IPA can't run query", "red"))
                return False
        return True

    def _calculate_summary_reports(
        self, *, query_id: str, batch: QueryBatch, destination: str
    ) -> Any:

        # On-device queries always succeed, but IPA can throw an OOB error
        query_succeeded = (
            self._try_consume_budget_for_ipa(destination, batch)
            if self.config.user.baseline == IPA
            else True
        )
        if query_succeeded:
            # Schedule the batch
            aggregation_result = self.aggregation_service.create_summary_report(batch)
            return aggregation_result
        else:
            return None

    def _log_query_result(
        self,
        query_id: str,
        batch: QueryBatch,
        destination: str,
        aggregation_result: Any,
    ) -> None:

        # Keep track of the number of queries answered (potentially with answer=None) to stop when workload_size is reached
        self.num_queries_answered += 1

        if aggregation_result is None:
            if self.config.user.bias_detection_knob is not None:
                true_output = aggregation_output = aggregation_noisy_output = np.nan
            else:
                true_output = aggregation_output = aggregation_noisy_output = None
        else:
            true_output = aggregation_result.true_output
            aggregation_output = aggregation_result.aggregation_output
            aggregation_noisy_output = aggregation_result.aggregation_noisy_output
            logger.info(
                f"Scheduling query #{self.num_queries_answered}, id {query_id}, true_output: {true_output}, aggregation_output: {aggregation_output}, noisy_output: {aggregation_noisy_output}"
            )

        # Log budgeting metrics and accuracy related data
        if BUDGET in self.config.logs.logging_keys:
            budget_metrics = {"max_max": 0, "sum_max": 0, "max_sum": 0, "sum_sum": 0}

            def update_budget_metrics(filters_per_origin):
                # Modifies `budget_metrics` in-place
                if destination in filters_per_origin:
                    budget_accountant = filters_per_origin[destination]
                    max_ = budget_accountant.get_max_consumption_across_blocks()
                    sum_ = budget_accountant.get_sum_consumption_across_blocks()

                    budget_metrics["max_max"] = max(budget_metrics["max_max"], max_)
                    budget_metrics["max_sum"] = max(budget_metrics["max_sum"], sum_)
                    budget_metrics["sum_max"] += max_
                    budget_metrics["sum_sum"] += sum_

            if self.global_filters_per_origin:
                update_budget_metrics(self.global_filters_per_origin)
            else:
                for user in self.users.values():
                    update_budget_metrics(user.filters_per_origin)

            self.logger.log(BUDGET, destination, batch.biggest_id, budget_metrics)

            if MLFLOW in self.config.logs.logging_keys:
                mlflow.log_metrics(
                    metrics={
                        "global_budget_sum_sum": budget_metrics["sum_sum"],
                        "global_budget_max_max": budget_metrics["max_max"],
                    },
                    step=self.num_queries_answered,
                )
                self.logger.store("latest_budget_sum_sum", budget_metrics["sum_sum"])

        if BIAS in self.config.logs.logging_keys:
            self.logger.log(
                BIAS,
                batch.biggest_id,
                destination,
                query_id,
                batch.get_global_epsilon(),
                batch.global_sensitivity,
                {
                    "true_output": true_output,
                    "aggregation_output": aggregation_output,
                    "aggregation_noisy_output": aggregation_noisy_output,
                },
            )

            if MLFLOW in self.config.logs.logging_keys:

                if self.config.user.bias_detection_knob:
                    # TODO(bias): add global bound
                                        
                    bias_metrics = compute_bias_metrics(
                        true_output,
                        aggregation_output,
                        aggregation_noisy_output,
                        kappa=self.config.user.bias_detection_knob,
                        max_report_global_sensitivity=batch.global_sensitivity,
                        laplace_noise_scale=batch.noise_scale,
                        batch_size=len(batch.values),
                        is_monotonic_scalar_query=self.config.user.is_monotonic_scalar_query,
                    )

                    logger.info(
                        f"Bias metrics for query {self.num_queries_answered}: {bias_metrics}"
                    )

                    # TODO: we could use the prior too, but the noisy_output is probably closer? Could also use the p95 bounds on both sides
                    # This is a heuristic, so scaling factors are fair game too
                    predicted_rmsre = predict_rmsre_naive(bias_metrics, batch)
                    true_rmsre = bias_metrics["rmsre"]
                    target_rmsre = self.config.user.target_rmsre

                    # Some extra logs to get workload-level metrics
                    aggregatable_metrics = compute_bias_prediction_metrics(
                        predicted_rmsre,
                        true_rmsre,
                        target_rmsre,
                    )

                    bias_metrics.update(
                        {
                            "rmsre_prediction": predicted_rmsre,
                            "truly_meets_rmsre_target": aggregatable_metrics[
                                "truly_meets_rmsre_target"
                            ],
                            "probably_meets_rmsre_target": aggregatable_metrics[
                                "probably_meets_rmsre_target"
                            ],
                        }
                    )
                    self.logger.log_one(MLFLOW, aggregatable_metrics)

                elif isinstance(true_output, np.ndarray):
                    # You probably don't want to log this, but who knows
                    bias_metrics = {}

                    for i in range(len(true_output)):
                        bias_metrics.update(
                            {
                                f"true_output_{i}": true_output[i],
                                f"aggregation_output_{i}": aggregation_output[i],
                                f"aggregation_noisy_output_{i}": aggregation_noisy_output[
                                    i
                                ],
                            }
                        )

                else:
                    bias_metrics = compute_base_bias_metrics(
                        true_output,
                        aggregation_output,
                        aggregation_noisy_output,
                        laplace_noise_scale=batch.noise_scale,
                    )

                mlflow.log_metrics(
                    bias_metrics,
                    step=self.num_queries_answered,
                )
                
        if MLFLOW in self.config.logs.logging_keys:
            # Other logs
            epoch_start, epoch_end = batch.epochs_window.get_epochs()
            mlflow.log_metrics(
                    metrics={
                        "epoch_start": epoch_start,
                        "epoch_end": epoch_end,
                    },
                    step=self.num_queries_answered,
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

    def _finalize_logs(self):
        self._log_all_filters_state()

        logs = {}
        logs["global_statistics"] = self.global_statistics.dump()

        if MLFLOW in self.config.logs.logging_keys:
            aggregatable_metrics = self.logger.logs.pop(MLFLOW, None)
            aggregated_metrics = aggregate_bias_prediction_metrics(aggregatable_metrics)

            # Also add the average budget here
            hardcoded_destination = "1"
            stats = logs["global_statistics"][hardcoded_destination]
            n_device_epochs = (
                stats["num_unique_device_filters_touched"] * stats["num_epochs_touched"]
            )
            budget_sum = self.logger.get("latest_budget_sum_sum")
            aggregated_metrics["avg_budget"] = budget_sum / n_device_epochs

            mlflow.log_metrics(
                metrics=aggregated_metrics, step=self.num_queries_answered
            )

            # TODO: plot some CDFs and log them as artifacts?
            # mlflow.log_figure()

        logs["event_logs"] = self.logger.logs
        logs["config"] = OmegaConf.to_object(self.config)
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

    config = OmegaConf.load(omegaconf)
    return Evaluation(config).run()


if __name__ == "__main__":
    app()
