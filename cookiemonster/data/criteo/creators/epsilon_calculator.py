import math

def get_epsilon_from_accuracy_for_counts(n: int, cap_value: int, *, a: float = 0.05, b: float = 0.01) -> float:
    s = cap_value
    a = 0.05
    b = 0.01
    return s * math.log(1 / b) / (n * a)
