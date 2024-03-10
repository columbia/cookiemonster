from typing import List, Tuple


def attribution_window_to_list(attribution_window: Tuple[int, int]) -> List[int]:
    return list(range(attribution_window[0], attribution_window[1] + 1))
