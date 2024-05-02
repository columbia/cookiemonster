# CookieMonster: Effective On-Device DP Budgeting for Private Attribution Measurement Systems

## Repo Structure

- `chromium-prototype`: Contains the following submodules:
    - `chromium`: On-device DP budgeting prototyped on Chromium based on Google's Attribution Reporting API.
    - `devtools-frontend`: frontend changes for extending Attribution Reporting API's Source and Trigger registration parameters. 
    - `depot_tools`: tweaked script to help fetch our Chromium version
    
    These submodules are not meant to be initialized upon cloning this repo. We keep them here only for pointers. See instructions below on how to setup the Chromium prototype.
- `config`: Custom configurations to be used in run_evaluation.py
- `cookiemonster`: A lightweight implementation in Python of the on-device DP budgeting that we use to run the experiments.
- `data`: Contains datasets that we used for the evaluation. Read their corresponding instructions on how to create them.
- `demo`: creating servers for adtech, advertiser, publisher using the extended API. 
- `experiments`: Contains scripts that use Ray.tune to run many experiments in parallel.
- `notebooks`: Contains notebooks that we use to analyze the results of the experiments



## Install dependencies

Install the package management system, poetry: `pip install poetry`.

We use Python 3.10.11. If that is not your system version, we advise using [pyenv](https://github.com/pyenv/pyenv).
Once pyenv is installed, run: `poetry env use 3.10.11`.

Then, install the dependencies: `poetry install`.

Finally, to run the poetry interactive shell, run `poetry shell`.


## Create datasets

Criteo:
```bash
cd cookiemonster/data/criteo
wget http://go.criteo.net/criteo-research-search-conversion.tar.gz
tar -xzf criteo-research-search-conversion.tar.gz
python3 create_dataset.py
```

Microbenchmark:
```bash
cd cookiemonster/data/microbenchmark

python3 create_dataset.py --user-participation-rate-per-query 0.001 --per-day-user-impressions-rate 0.1 &&
python3 create_dataset.py --user-participation-rate-per-query 0.01 --per-day-user-impressions-rate 0.1 &&
python3 create_dataset.py --user-participation-rate-per-query 0.1 --per-day-user-impressions-rate 0.1 &&
python3 create_dataset.py --user-participation-rate-per-query 1.0 --per-day-user-impressions-rate 0.1 &&
python3 create_dataset.py --user-participation-rate-per-query 0.1 --per-day-user-impressions-rate 0.001 &&
python3 create_dataset.py --user-participation-rate-per-query 0.1 --per-day-user-impressions-rate 0.01 &&
python3 create_dataset.py --user-participation-rate-per-query 0.1 --per-day-user-impressions-rate 1.0
```

PATCG Synthetic Dataset:
```bash
cd cookiemonster/data/patcg

# todo(automate generating them)
```

## Run experiments

### Run one experiment at a time
```bash
python3 cookiemonster/run_evaluation.py --omegaconf config/config.json
```

### Run many experiments in parallel
```bash
python3 experiments/runner.cli.py
```

The results will be stored inside the `logs` directory.
Use the notebooks in `notebooks` to check how to analyze the results.
