# Create microbenchmark datasets
cd data/microbenchmark

python3 create_dataset.py --user-participation-rate-per-query 0.001 --per-day-user-impressions-rate 0.1 &&
python3 create_dataset.py --user-participation-rate-per-query 0.01 --per-day-user-impressions-rate 0.1 &&
python3 create_dataset.py --user-participation-rate-per-query 0.1 --per-day-user-impressions-rate 0.1 &&
python3 create_dataset.py --user-participation-rate-per-query 1.0 --per-day-user-impressions-rate 0.1 &&
python3 create_dataset.py --user-participation-rate-per-query 0.1 --per-day-user-impressions-rate 0.001 &&
python3 create_dataset.py --user-participation-rate-per-query 0.1 --per-day-user-impressions-rate 0.01 &&
python3 create_dataset.py --user-participation-rate-per-query 0.1 --per-day-user-impressions-rate 1.0

# Create criteo datasets
cd ../criteo

wget http://go.criteo.net/criteo-research-search-conversion.tar.gz
tar -xzf criteo-research-search-conversion.tar.gz
python3 create_dataset.py

# Create patcg datasets
cd ../patcg
python3 download_data.py
python3 create_dataset.py