from typing import Dict, List, Any, Union, Tuple


def attribution_window_to_list(attribution_window: Tuple[int, int]) -> List[int]:
    return list(range(self.attribution_window[0], self.attribution_window[1] + 1))
