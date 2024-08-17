# Cookie Monster

Cookie Monster is an on-device budgeting component that can be integrated into differentially private ad-measurement systems.
Powered by a robust theoretical framework known as Individual Differential Privacy (IDP), a variant of traditional differential privacy, Cookie Monster allows advertisers to conserve significantly more privacy budget compared to existing alternatives.
A description of this project can be found on our paper, titled Cookie Monster: Effective On-Device DP Budgeting for Private Attribution Measurement Systems and published as SOSP '24.

## Repo Structure

- `config`: Custom configurations to be used in run_evaluation.py
- `cookiemonster`: A lightweight implementation in Python of the on-device DP budgeting that we use to run the experiments.
- `data`: Contains datasets that we used for the evaluation.
- `experiments`: Contains scripts that use Ray.tune to run many experiments in parallel.
- `notebooks`: Contains notebooks that we use to analyze the results of the experiments
- `demo`: creating servers for adtech, advertiser, publisher using the extended API.
- `chromium-prototype`: Contains the following submodules:
  - `chromium`: On-device DP budgeting prototyped on Chromium based on Google's Attribution Reporting API.
  - `devtools-frontend`: frontend changes for extending Attribution Reporting API's Source and Trigger registration parameters.
  - `depot_tools`: tweaked script to help fetch our Chromium version

    These submodules are not meant to be initialized upon cloning this repo. We keep them here only for pointers. See instructions below on how to setup the Chromium prototype.


## 1. Requirements

Make sure you have a working installation of [`docker`](https://docs.docker.com/get-docker/).

## 2. Install Cookie Monster
### Download the code

Clone this repository on your machine:
```bash
git clone https://github.com/columbia/cookiemonster.git
```

Enter the repository:
```bash
cd cookiemonster
```

### Build the Turbo docker

Build the docker image for CookieMonster. This will automatically install all dependencies required for the CookieMonster system as well as the datasets, queries and workloads used in the evaluation section of the paper. This step takes several minutes to finish (~20') due to the processing and generation of the datasets.
``` bash 
sudo docker build -t cookiemonster -f Dockerfile .


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
https://docs.google.com/document/d/1Vxq4LrMe3A2WIlu-7IYP1Hycr_nz3_qTpPAICX9fLcw/edit#heading=h.5viiz0en8hkz
cd cookiemonster/data/microbenchmark
```bash
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
