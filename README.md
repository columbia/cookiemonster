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

Spec requirements: at least 128 GB memory, and 100 GB disk space.


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

### Build the Cookie Monster docker

Build the docker image for CookieMonster. This will automatically install all dependencies required for the CookieMonster system as well as the datasets used in the evaluation section of the paper. This step takes several minutes to finish (~30') due to the processing and generation of the datasets.
``` bash 
sudo docker build -t cookiemonster -f Dockerfile .
```

## 3. Reproduce experiments

The [experiments/runner.cli.py](https://github.com/columbia/cookiemonster/blob/artifact-sosp/experiments/runner.cli.py) script automates the execution of multiple experiments concurrently using [Ray](https://www.ray.io/). You can find the configuration for each experiment hardcoded inside the script.

The script [experiments/run_all.sh](https://github.com/columbia/cookiemonster/blob/artifact-sosp/experiments/run_all.sh) contains a complete list of all the commands that generate the experiments presented in the paper.

### 3.1. Run all experiments

Reproduce all Cookie Monster experiments by running the cookiemonster docker with the following command:

``` bash
sudo docker run -v $PWD/logs:/cookiemonster/logs $PWD/figures:/cookiemonster/figures -v $PWD/cookiemonster/config:/cookiemonster/cookiemonster/config -v $PWD/temp:/tmp --network=host --name cookiemonster --shm-size=204.89gb --rm cookiemonster experiments/run_all.sh
```

This step takes around 12 hours to finish.

Make sure that the cookiemonster container has enough disk space at `/tmp` which Ray uses to store checkpoints and other internal data. If that's not the case then mount the `/tmp` directory on a directory that has enough space.

For example, use an additional -v flag followed by the directory of your choice:

`-v $PWD/temp:/tmp`

With the `-v` flag we mount directories `cookiemonster/logs`, `cookiemonster/figures` and `cookiemonster/config` from the host into the container so that we can access the logs/figures from the host even after the container stops and also allow for the container to access user-defined configurations stored in the host.

### 3.2. Analyze results

The [experiments/runner.cli.py](https://github.com/columbia/cookiemonster/blob/artifact-sosp/experiments/runner.cli.py) script will automatically analyze the execution logs and create plots corresponding to the figures presented in the paper.

Check the `figures` directory for all the outputs.