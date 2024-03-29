## CookieMonster Python imlementation


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

Synthetic:
```bash
cd adsprivate/cookiemonster/data/synthetic

python3 create_dataset.py --user-conversions-rate 0.1 &&
python3 create_dataset.py --user-conversions-rate 0.25 &&
python3 create_dataset.py --user-conversions-rate 0.5 &&
python3 create_dataset.py --user-conversions-rate 0.75 &&
python3 create_dataset.py --user-conversions-rate 1.0

python3 create_dataset.py --per-day-user-impressions-rate 0.25 &&
python3 create_dataset.py --per-day-user-impressions-rate 0.5 &&
python3 create_dataset.py --per-day-user-impressions-rate 0.75 &&
python3 create_dataset.py --per-day-user-impressions-rate 1.0
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
python3 cookiemonster/cookiemonster/run_evaluation.py --omegaconf cookiemonster/config/config.json
```

The results will be stored inside the `adsprivate/cookiemonster/logs` directory.
