# adsprivate

# Effective On-Device DP Budgeting for Private Attribution Measurement Systems

## Repo Structure

- `chromium-prototype`: Contains the following submodules:
    - `chromium`: On-device DP budgeting prototyped on Chromium based on Google's Attribution Reporting API.
    - `devtools-frontend`: frontend changes for extending Attribution Reporting API's Source and Trigger registration parameters. 
    - `depot_tools`: tweaked script to help fetch our Chromium version
    
    These submodules are not meant to be initialized upon cloning this repo. We keep them here only for pointers. See instructions below on how to setup the Chromium prototype.

- `demo`: creating servers for adtech, advertiser, publisher using the extended API. 

- `cookiemonster`: A lightweight implementation of the on-device DP budgeting that we use to run experiments.
    - `data`: Contains datasets that we used for the evaluation. Run the corresponding `create_dataset.py` scripts to create each dataset.
    - `experiments`: Contains scripts that use Ray.tune to run many experiments in parallel.
    - `notebooks`: Contains notebooks that we use to analyze the results of the experiments
     - `cookiemonster`: Contains the main functionality of the on-device  DP budgeting

## Install dependencies

```bash
pip install poetry
poetry install
poetry shell
```

## Create datasets

Criteo:
```bash
cd adsprivate/cookiemonster/data/criteo
wget http://go.criteo.net/criteo-research-search-conversion.tar.gz
tar -xzf criteo-research-search-conversion.tar.gz
python3 create_dataset.py
```

## Run `cookiemonster` experiments

Enter repo:
```bash
cd adsprivate
```

### Run many experiments in parallel
```bash
PYTHONPATH="cookiemonster" python3 cookiemonster/experiments/runner.cli.py
```

The results will be stored inside the `adsprivate/cookiemonster/logs` directory.
Use the notebooks in `adsprivate/cookiemonster/notebooks` to check how to analyze the results.

### Run one experiment at a time
```bash
python3 cookiemonster/cookiemonster/run_evaluation.py --config cookiemonster/config/config.json
```

The results will be stored inside the `adsprivate/cookiemonster/logs` directory.


## Setup `chromium`

Don't initialize the chromium submodule. We keep it here just for quick access to the chromuim repo. Use the following instructions to download/install our chromium instead.

#### Clone depot_tools:
```bash
git clone https://github.com/columbia/depot_tools.git
```

#### Add depot_tools to PATH
```bash
export PATH="/path/to/depot_tools:$PATH"
```

#### Create a directory to download chromium
```bash
mkdir chromium && cd chromium
```

#### Fetch chromium
```bash
fetch --nohooks chromium
```

Press `n` to the following:
`OK to update it to https://chromium.googlesource.com/chromium/tools/depot_tools.git ? [Y/n] n`


#### Install dependencies
```bash
cd src
./build/install-build-deps.sh
```

#### Run the hooks
```bash
gclient runhooks
```

Press `n` to the following:
`OK to update it to https://chromium.googlesource.com/chromium/tools/depot_tools.git ? [Y/n] n`

#### Setup the build
```bash
gn gen out/Default
```


#### Pull changes from our devtools-frontend 
Hacky way to pull changes from our devtools-frontend. 
Previous command fetched the google repos instead. TODO: fix that.
```bash
cd src/third_party/devtools-frontend/src/
git remote add cu https://github.com/columbia/devtools-frontend.git
git fetch cu
git checkout cu/cu-ara
```


#### Build chromium
```bash
autoninja -C out/Default chrome
```

#### Setup Exporting Display
```bash
sudo apt-get install xorg
sudo apt-get install xvfb
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
```


#### Run Chromium:
``` bash
out/Default/chrome --disable-gpu --user-data-dir=/mydata/chromium-data --remote-debugging-port=8888 --flag-switches-begin --disable-field-trial-config  --start-maximized --enable-privacy-sandbox-ads-apis --privacy-sandbox-enrollment-overrides=http://arapi-adtech.localhost:8085 --show-overdraw-feedback --flag-switches-end --restore-last-session  http://arapi-publisher.localhost:8087/
```

