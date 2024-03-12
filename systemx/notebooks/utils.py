from experiments.ray.analysis import load_ray_experiment
from systemx.utils import LOGS_PATH



def get_df(path):
    logs = LOGS_PATH.joinpath(path)
    df = load_ray_experiment(logs)
    return df