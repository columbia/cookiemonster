import math


def get_epsilon_from_accuracy_for_counts(
    n: int, cap_value: int, *, a: float = 0.05, b: float = 0.01
) -> float:
    """
    By default, allows an absolute error of 0.05n whp for a sum of n values, each in [0, cap_value].
    - That's very accurate (too accurate and expensive) if the true sum is close to n*cap_value for large cap_value.
    - When cap_value is small (e.g. cap_value = 5) and queries are close to 1 on average then it's good.
    - When cap_value is very small and queries are close to 0 on average then it's not accurate at all.
    """

    # Original (identical) statement: return cap_value * math.log(1 / b) / (n * a)

    # Turbo had cap_value = 1 and sensitivity = 1/n:
    # return get_epsilon_for_high_probability_absolute_error(sensitivity=cap_value / n, absolute_error=a, failure_probability=b)
    return get_epsilon_for_high_probability_absolute_error(
        sensitivity=cap_value, absolute_error=a * n, failure_probability=b
    )


def get_epsilon_for_high_probability_relative_error_wrt_prior(
    sensitivity: float,
    expected_result: float,
    relative_error: float,
    failure_probability: float,
):
    return (
        sensitivity
        * math.log(1 / failure_probability)
        / (relative_error * expected_result)
    )


def get_epsilon_for_high_probability_relative_error_wrt_avg_prior(
    sensitivity: float,
    batch_size: int,
    expected_average_result: float,
    relative_error: float,
    failure_probability: float,
):
    return get_epsilon_for_high_probability_relative_error_wrt_prior(
        sensitivity=sensitivity,
        expected_result=batch_size * expected_average_result,
        relative_error=relative_error,
        failure_probability=failure_probability,
    )


def get_epsilon_for_high_probability_absolute_error(
    sensitivity: float, absolute_error: float, failure_probability: float
):
    """Returns smallest epsilon so that Pr[|M(D) - Q(D)| > absolute_error] <= failure_probability
    where Q has sensitivity `senstivity` and M(D) = Q(D) + X
    with X ~ Lap(sensitivity / epsilon)
    (so M is epsilon-DP)

    Uses a tail bound: Pr[|X| > absolute_error] = exp(-absolute_error * epsilon / sensitivity)
    """
    return sensitivity * math.log(1 / failure_probability) / absolute_error


def get_epsilon_for_rmsre_wrt_avg_prior(
    sensitivity: float,
    batch_size: int,
    expected_average_result: float,
    relative_error: float,
):
    expected_result = expected_average_result * batch_size
    return get_epsilon_for_rmse(sensitivity, relative_error * expected_result)


def get_epsilon_for_rmsre_wrt_prior(
    sensitivity: float, expected_result: float, relative_error: float
):
    return get_epsilon_for_rmse(sensitivity, relative_error * expected_result)


def get_epsilon_for_rmse(sensitivity: float, rmse: float):
    """Returns smallest epsilon so that sqrt(E[(M(D) - Q(D))^2]) < rmse
    where Q has sensitivity `senstivity` and M(D) = Q(D) + X
    with X ~ Lap(sensitivity / epsilon)
    (so M is epsilon-DP)

    Uses a variance: V[|X|] = 2 * (sensitivity / epsilon)^2
    """
    return math.sqrt(2) * sensitivity / rmse
