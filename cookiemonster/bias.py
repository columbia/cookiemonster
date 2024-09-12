from typing import Dict

import numpy as np
import pandas as pd
from loguru import logger


def compute_base_bias_metrics(
    true_output: float,
    aggregation_output: float,
    aggregation_noisy_output: float,
    laplace_noise_scale: float,
) -> Dict[str, float]:
    true_bias = true_output - aggregation_output

    # E[(M(D) - Q(D))^2] = E[(Q'(D) + X - Q(D))^2] = (Q(D) - Q'(D))^2 + E[X^2]
    # With X ~ Lap(b) with variance 2b^2
    mse = true_bias**2 + 2 * laplace_noise_scale**2
    if true_output != 0:
        # √E[(M(D) - Q(D))^2 / Q(D)^2]
        rmsre = np.sqrt(mse / true_output**2)
    else:
        rmsre = np.nan
    return {
        "output_true": true_output,
        "aggregation_output": aggregation_output,
        "aggregation_output_noisy": aggregation_noisy_output,
        "bias_true": true_bias,
        "rmsre": rmsre,
    }


def compute_bias_metrics(
    true_output: np.ndarray,
    aggregation_output: np.ndarray,
    aggregation_noisy_output: np.ndarray,
    kappa: float,
    max_report_global_sensitivity: float,
    laplace_noise_scale: float,
    batch_size: int,
    is_monotonic_scalar_query: bool = False,
) -> Dict[str, float]:
    """
    Outputs are vectors with 2 dimensions (first one for the DP count, second for the scalar metric)
    """

    m = compute_base_bias_metrics(
        true_output[1],
        aggregation_output[1],
        aggregation_noisy_output[1],
        laplace_noise_scale,
    )
    
    # Formula needs the sensitivity of the original query, without the count query
    delta = max_report_global_sensitivity - kappa
    
    empty_epochs_true = aggregation_output[0] / kappa
    empty_epochs_noisy = aggregation_noisy_output[0] / kappa
    bias_noisy = empty_epochs_noisy * delta
    bias_noisy_p95_confidence = bias_noisy + (
        delta / kappa
    ) * laplace_noise_scale * np.log(1 / 0.05) / np.sqrt(2)

    if is_monotonic_scalar_query:
        logger.info(
            f"Monotonic scalar query, batch size {batch_size}, sensitivity {delta}"
        )
        monotonic_bias_bound = (
            batch_size * delta - aggregation_noisy_output[1]
        )
        bias_best_bound = min(bias_noisy_p95_confidence, monotonic_bias_bound)
    else:
        monotonic_bias_bound = np.nan
        bias_best_bound = bias_noisy_p95_confidence

    m.update(
        {
            "empty_epochs_true": empty_epochs_true,
            "empty_epochs_noisy": empty_epochs_noisy,
            "bias_noisy": bias_noisy,
            "bias_noisy_p95_confidence": bias_noisy_p95_confidence,
            "bias_bound_from_true_empty_epochs": empty_epochs_true
            * delta,
            "bias_monotonic_bound": monotonic_bias_bound,
            "bias_best_bound": bias_best_bound,
        }
    )
    return m


def predict_rmsre_naive(bias_metrics, batch):
    return np.sqrt(
        (bias_metrics["bias_best_bound"] ** 2 + 2 * batch.noise_scale**2)
        / bias_metrics["aggregation_output_noisy"] ** 2
    )


def compute_bias_prediction_metrics(
    predicted_rmsre,
    true_rmsre,
    target_rmsre,
):

    metrics = {
        "predicted_rmse": predicted_rmsre,
        "truly_meets_rmsre_target": 1 if true_rmsre <= target_rmsre else 0,
        "probably_meets_rmsre_target": (1 if predicted_rmsre <= target_rmsre else 0),
        "accurately_predicted_rmsre_target": (
            1
            if ((true_rmsre <= target_rmsre) == (predicted_rmsre <= target_rmsre))
            else 0
        ),
        "tp": (
            1 if true_rmsre > target_rmsre and predicted_rmsre > target_rmsre else 0
        ),
        "fp": (
            1 if true_rmsre <= target_rmsre and predicted_rmsre > target_rmsre else 0
        ),
        "tn": (
            1 if true_rmsre <= target_rmsre and predicted_rmsre <= target_rmsre else 0
        ),
        "fn": (
            1 if true_rmsre > target_rmsre and predicted_rmsre <= target_rmsre else 0
        ),
    }

    return metrics


def aggregate_bias_prediction_metrics(aggregatable_metrics):

    if aggregatable_metrics is None:
        return {}

    df = pd.DataFrame(aggregatable_metrics)
    aggregated_metrics = {
        "prediction_accuracy": df["accurately_predicted_rmsre_target"].mean(),
        "true_positives": df["tp"].mean(),
        "false_positives": df["fp"].mean(),
        "true_negatives": df["tn"].mean(),
        "false_negatives": df["fn"].mean(),
    }
    return aggregated_metrics
