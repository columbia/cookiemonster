import json
from pathlib import Path
from typing import Union

import pandas as pd


def load_ray_experiment(logs: Union[Path, str]) -> pd.DataFrame:
    results = []
    print(logs.glob("**/result.json"))
    for run_result in logs.glob("**/result.json"):
        print(run_result)
        try:
            with open(run_result, "r") as f:
                d = json.load(f)
            results.append(d)
        except Exception:
            pass
    df = pd.DataFrame(results)
    return df
