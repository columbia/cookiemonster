from typing import Dict

import numpy as np
import pandas as pd


def compute_bias_metrics(
    true_output: np.ndarray,
    aggregation_output: np.ndarray,
    aggregation_noisy_output: np.ndarray,
    kappa: float,
    max_report_global_sensitivity: float,
    laplace_noise_scale: float,
) -> Dict[str, float]:
    """
    Outputs are vectors with 2 dimensions (first one for the DP count, second for the scalar metric)
    """

    m = {
        "true_output": true_output[1],
        "aggregation_output": aggregation_output[1],
        "aggregation_noisy_output": aggregation_noisy_output[1],
        "true_bias": true_output[1] - aggregation_output[1],
        "true_empty_epochs": true_output[0] / kappa,  # Should be 0 for us
        "true_empty_or_exhausted_epochs": aggregation_output[0] / kappa,
        "noisy_empty_epochs": aggregation_noisy_output[0] / kappa,
        "noisy_bias": aggregation_noisy_output[0]
        * (max_report_global_sensitivity / kappa),
        "noisy_bias_p95_confidence": (
            aggregation_noisy_output[0]
            + laplace_noise_scale * np.log(1 / 0.05) / np.sqrt(2)
        )
        * (max_report_global_sensitivity / kappa),
    }

    # E[(M(D) - Q(D))^2] = E[(Q'(D) + X - Q(D))^2] = (Q(D) - Q'(D))^2 + E[X^2]
    # With X ~ Lap(b) with variance 2b^2
    mse = m["true_bias"] ** 2 + 2 * laplace_noise_scale**2
    if m["true_output"] != 0:
        # √E[(M(D) - Q(D))^2 / Q(D)^2]
        rmsre = np.sqrt(mse / m["true_output"] ** 2)
    else:
        rmsre = np.nan

    m["rmsre"] = rmsre
    return m


def predict_rmsre_naive(bias_metrics, batch):
    return np.sqrt(
        (bias_metrics["noisy_bias"] ** 2 + 2 * batch.noise_scale**2)
        / bias_metrics["aggregation_noisy_output"] ** 2
    )


def compute_bias_prediction_metrics(
    predicted_rmsre,
    true_rmsre,
    target_rmsre,
):

    metrics = {
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
