# adsprivate

# CookieMonster: Effective On-Device DP Budgeting for Private Attribution Measurement Systems

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