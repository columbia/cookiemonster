import uuid
import pandas as pd
import numpy as np
import random
import datetime
import typer
from omegaconf import OmegaConf
from typing import Dict, Any, List


app = typer.Typer()

def generate_uuid() :
  return uuid.uuid4().hex.upper()

def right_skewed_probability(levels, p_start = 2.0/3, p_end = 1.0/3) :
  prob = np.linspace(p_start, p_end, levels)
  prob /= prob.sum()
  return prob

def generate_column_uniform(levels, n) :
  values = [val for val in range(levels)]
  return np.random.choice(values, size=n)

def generate_column_right_skewed(levels, n) :
  values = [val for val in range(levels)]
  return np.random.choice(values, size=n, p=right_skewed_probability(levels))

def generate_log_normal_distribution(n, mean_lognormal=0.5, sigma_lognormal = 1.0) :
  return np.ceil(np.random.lognormal(mean=mean_lognormal, sigma=sigma_lognormal, size=n)).astype(int)

def generate_poisson_distribution(n) :
  distribution = np.ceil(np.random.poisson(lam=0.5, size=n)).astype(int)
  distribution_2 = [val + 1 for val in distribution]
  return distribution_2

def generate_random_date(start_date, num_days) :
  start_seconds = int((start_date - datetime.datetime(1970, 1, 1)).total_seconds())
  # end_seconds = int((end_date - datetime.datetime(1970, 1, 1)).total_seconds())
  end_seconds = start_seconds + (num_days * 24 * 60 * 60)
  random_seconds = random.randint(start_seconds, end_seconds)
  return datetime.datetime.utcfromtimestamp(random_seconds)
  
def generate_publisher_user_profile(config) :
  id_attribute = 'device_id'
  attribute_columns = ['pub_profile_1', 'pub_profile_2', 'pub_profile_3', 'pub_profile_4', 'pub_profile_5', 'pub_profile_6', 'pub_profile_7', 'pub_profile_8']
  segment = 'pub_segment'

  data = {}
  data[id_attribute] = [generate_uuid() for _ in range(config.num_users)]
  data[attribute_columns[0]] = generate_column_uniform(2, config.num_users)
  data[attribute_columns[1]] = generate_column_uniform(10, config.num_users)
  data[attribute_columns[2]] = generate_column_uniform(1000, config.num_users)
  data[attribute_columns[3]] = generate_column_uniform(10000, config.num_users)
  data[attribute_columns[4]] = generate_column_right_skewed(2, config.num_users)
  data[attribute_columns[5]] = generate_column_right_skewed(10, config.num_users)
  data[attribute_columns[6]] = generate_column_right_skewed(1000, config.num_users)
  data[attribute_columns[7]] = generate_column_right_skewed(10000, config.num_users)
  data[segment] = [val//1000 + 1 for val in range(config.num_users)]

  return pd.DataFrame(data)

def generate_ad_exposure_records(config, publisher_user_profile) :
  # Timestamp of the exposure exp_timestamp is uniformly distributed over 30 days
  # We have one exposure (impression) attribute e.g. publick id - 100 distinct product ids.

  device_id = 'device_id'
  exp_record_id = 'exp_record_id'
  exp_timestamp = 'exp_timestamp'
  exp_ad_interaction = 'exp_ad_interaction'
  attribute_columns = ['exp_attribute_1', 'exp_attribute_2', 'exp_attribute_3', 'exp_attribute_4', 'exp_attribute_5', 'exp_attribute_6', 'exp_attribute_7', 'exp_attribute_8', 'exp_attribute_9']

  # Number of exposures per exposed users ~ ceil(Lognormal(meanlog = 0.5, sdlog = 1)), i.e., around 3.22 exposures per exposed user
  lognormal_distribution = generate_log_normal_distribution(config.num_users)
  records_size = sum(lognormal_distribution)
  start_date = datetime.datetime(2024, 1, 1)
  # end_date = end_date = datetime.datetime(2024, 1, 31)

  data = {}
  data[exp_record_id] = [generate_uuid() for _ in range(records_size)]
  data[device_id] =  np.repeat(publisher_user_profile[device_id], lognormal_distribution)
  data[exp_timestamp] = [generate_random_date(start_date, config.num_days) for _ in range(records_size)]
  data[exp_ad_interaction] = np.random.choice(['view', 'click'], size=records_size, p=[0.99, 0.01])

  data[attribute_columns[0]] = generate_column_uniform(2, records_size)
  data[attribute_columns[1]] = generate_column_uniform(10, records_size)
  data[attribute_columns[1]] = generate_column_uniform(10000, records_size)
  data[attribute_columns[3]] = generate_column_uniform(10000, records_size)
  data[attribute_columns[4]] = generate_column_right_skewed(2, records_size)
  data[attribute_columns[5]] = generate_column_right_skewed(10, records_size)
  data[attribute_columns[6]] = generate_column_right_skewed(1000, records_size)
  data[attribute_columns[7]] = generate_column_right_skewed(10000, records_size)
  data[attribute_columns[8]] = generate_column_uniform(config.num_disjoint_queries, records_size)
  return pd.DataFrame(data)

def generate_advertiser_user_profile(config, publisher_user_profile, ad_exposure_records):
  id_attribute = 'device_id'
  attribute_columns = ['conv_profile_1', 'conv_profile_2', 'conv_profile_3', 'conv_profile_4', 'conv_profile_5', 'conv_profile_6', 'conv_profile_7', 'conv_profile_8']
  segment = 'conv_segment'
  probabilities = [0.01 for _ in range(config.num_users)]

  for i in range(config.num_users):
    scaleup = 0
    probability = 0.01
    if publisher_user_profile.loc[i]['pub_profile_1'] == 1 :
      scaleup +=  probability*0.02
    if publisher_user_profile.loc[i]['pub_profile_2'] > 0 :
      scaleup += probability*0.02*publisher_user_profile.loc[i]['pub_profile_2']/9
    if  publisher_user_profile.loc[i]['pub_profile_3'] > 0 :
      scaleup += probability*0.02*publisher_user_profile.loc[i]['pub_profile_3']/999
    if  publisher_user_profile.loc[i]['pub_profile_4'] > 0 :
      scaleup += probability*0.02*publisher_user_profile.loc[i]['pub_profile_4']/9999
    probabilities += scaleup

  for _,record in ad_exposure_records.iterrows() :
    id = record[id_attribute]
    probability = 0.01
    scaleup = 0
    if record['exp_attribute_5'] == 1 :
      scaleup +=  probability*0.02
    if record['exp_attribute_6'] > 0 :
      scaleup += probability*0.02*record['exp_attribute_6']/9
    if  record['exp_attribute_7'] > 0 :
      scaleup += probability*0.02*record['exp_attribute_7']/999
    if  record['exp_attribute_8'] > 0 :
      scaleup += probability*0.02*record['exp_attribute_8']/9999
    device_index = publisher_user_profile.loc[publisher_user_profile[id_attribute] == id].index[0]
    probabilities[device_index] += scaleup

  publisher_user_profile['probability'] = probabilities
  converted_users_count = (config.num_users * config.conversion_rate)//100
  users = publisher_user_profile[id_attribute].sample(n=converted_users_count, weights=publisher_user_profile['probability'])

  data = {}
  data[id_attribute] = users.tolist()
  data[attribute_columns[0]] = generate_column_uniform(2, converted_users_count)
  data[attribute_columns[1]] = generate_column_uniform(10, converted_users_count)
  data[attribute_columns[2]] = generate_column_uniform(1000, converted_users_count)
  data[attribute_columns[3]] = generate_column_uniform(10000, converted_users_count)
  data[attribute_columns[4]] = generate_column_right_skewed(2, converted_users_count)
  data[attribute_columns[5]] = generate_column_right_skewed(10, converted_users_count)
  data[attribute_columns[6]] = generate_column_right_skewed(1000, converted_users_count)
  data[attribute_columns[7]] = generate_column_right_skewed(10000, converted_users_count)
  data[segment] = [val//1000 + 1 for val in range(converted_users_count)]

  return pd.DataFrame(data)

def generate_conversion_records(config, publisher_user_profile, ad_exposure_records, advertiser_user_profile) :
  device_id = 'device_id'
  conv_record_id = 'conv_record_id'
  conv_timestamp = 'conv_timestamp'
  attribute_columns = ['conv_attribute_1', 'conv_attribute_2', 'conv_attribute_3', 'conv_attribute_4', 'conv_attribute_5', 'conv_attribute_6', 'conv_attribute_7', 'conv_attribute_8', 'conv_attribute_9']
  conv_amount = 'conv_amount'

  userCount = publisher_user_profile.shape[0]
  converted_user_count = advertiser_user_profile.shape[0]

  amount_means = [2.0 for _ in range(converted_user_count)]

  for i in range(converted_user_count):
    scaleup = 0
    mean = 2.0
    device_index = publisher_user_profile.loc[publisher_user_profile[device_id] == advertiser_user_profile.loc[i][device_id]].index[0]
    if publisher_user_profile.loc[device_index]['pub_profile_1'] == 1 :
      scaleup +=  mean*0.04
    if publisher_user_profile.loc[device_index]['pub_profile_2'] > 0 :
      scaleup += mean*0.04*publisher_user_profile.loc[device_index]['pub_profile_2']/9
    if  publisher_user_profile.loc[device_index]['pub_profile_3'] > 0 :
      scaleup += mean*0.04*publisher_user_profile.loc[device_index]['pub_profile_3']/999
    if  publisher_user_profile.loc[device_index]['pub_profile_4'] > 0 :
      scaleup += mean*0.04*publisher_user_profile.loc[device_index]['pub_profile_4']/9999
    amount_means[i] += scaleup

  for _,record in ad_exposure_records.iterrows() :
    id = record[device_id]
    device = advertiser_user_profile.loc[advertiser_user_profile[device_id] == id]
    if device.empty :
      continue

    mean = 2.0
    scaleup = 0
    if record['exp_attribute_5'] == 1 :
      scaleup +=  mean*0.04
    if record['exp_attribute_6'] > 0 :
      scaleup += mean*0.04*record['exp_attribute_6']/9
    if  record['exp_attribute_7'] > 0 :
      scaleup += mean*0.04*record['exp_attribute_7']/999
    if  record['exp_attribute_8'] > 0 :
      scaleup += mean*0.04*record['exp_attribute_8']/9999

    amount_means[device.index[0]] += scaleup

  advertiser_user_profile['amount_mean'] = amount_means

  poisson_distribution = generate_poisson_distribution(converted_user_count)
  records_size = sum(poisson_distribution)
  start_date = datetime.datetime(2024, 1, 1)
  # end_date = end_date = datetime.datetime(2024, 1, 31)

  data = {}
  data[conv_record_id] = [generate_uuid() for _ in range(records_size)]
  data[device_id] =  np.repeat(advertiser_user_profile[device_id], poisson_distribution)
  mean_values = np.repeat(advertiser_user_profile['amount_mean'], poisson_distribution)
  data[conv_timestamp] = [generate_random_date(start_date, config.num_days) for _ in range(records_size)]

  data[attribute_columns[0]] = generate_column_uniform(2, records_size)
  data[attribute_columns[1]] = generate_column_uniform(10, records_size)
  data[attribute_columns[2]] = generate_column_uniform(1000, records_size)
  data[attribute_columns[3]] = generate_column_uniform(10000, records_size)
  data[attribute_columns[4]] = generate_column_right_skewed(2, records_size)
  data[attribute_columns[5]] = generate_column_right_skewed(10, records_size)
  data[attribute_columns[6]] = generate_column_right_skewed(1000, records_size)
  data[attribute_columns[7]] = generate_column_right_skewed(10000, records_size)
  data[attribute_columns[8]] = generate_column_uniform(config.num_disjoint_queries, records_size)

  data[conv_amount] = [np.random.lognormal(mean=value, sigma=1.0) for value in mean_values]

  return pd.DataFrame(data)

def create_synthetic_dataset(config: Dict[str, Any]) :
  '''
      Dataset constraints:
      a) Each user must contribute at most with one report per query
      b) We need  Q disjoint queries (say Q different product ids or some other feature) for an advertiser
      c) we have one advertiser
      d) the total conversion reports for each query is K (this means we probably need many many users)
      e) each user produces impressions for the product ids with a frequency R
      f) say a total of D days

      some possible values:
      Q = 100,
      K=20K,
      R= [value that yields no impressions at all, value that yields at least one impression per day]
      D = 90
  '''
  config = OmegaConf.create(config)

  publisher_user_profile = generate_publisher_user_profile(config)
  ad_exposure_records = generate_ad_exposure_records(config, publisher_user_profile)
  
  advertiser_user_profile = generate_advertiser_user_profile(config, publisher_user_profile, ad_exposure_records)
  conversion_records = generate_conversion_records(config, publisher_user_profile, ad_exposure_records, advertiser_user_profile)

  df_pubs = publisher_user_profile.merge(ad_exposure_records, how='inner', on='device_id',suffixes=('_pub_profile', '_exposure_record'))
  df_convs = advertiser_user_profile.merge(conversion_records, how='inner', on='device_id',suffixes=('_ad_profile', '_conv_record'))

  partner_id = generate_uuid()
  product_id = generate_uuid()
  impressions = df_pubs[["device_id","exp_timestamp"]]
  impressions = impressions.rename(columns={"device_id" : "user_id", "exp_timestamp" : "click_timestamp"})
  impressions["click_day"] = impressions["click_timestamp"].apply(
      lambda x: (7 * (x.isocalendar().week - 1)) + x.isocalendar().weekday
  )
  impressions["key"] = "purchaseCount"
  impressions["partner_id"] = partner_id
  impressions["product_id"] = product_id

  conversions = df_convs[["device_id", 'conv_timestamp', 'conv_amount']]
  conversions = conversions.rename(columns={"device_id" : "user_id", 'conv_timestamp' : 'conversion_timestamp', 'conv_amount' : 'SalesAmountInEuro'})
  conversions["SalesAmountInEuro"] = conversions["SalesAmountInEuro"].round(decimals=0)
  conversions = conversions.sort_values(by=["conversion_timestamp"])
  conversions["conversion_day"] = conversions["conversion_timestamp"].apply(
      lambda x: (7 * (x.isocalendar().week - 1)) + x.isocalendar().weekday
  )
  conversions["partner_id"] = partner_id
  conversions["product_id"] = product_id

  max_values = conversions.groupby(["partner_id"])["SalesAmountInEuro"].max()
  max_values = max_values.reset_index(name="aggregatable_cap_value")
  max_values["aggregatable_cap_value"] = max_values["aggregatable_cap_value"]
  conversions = conversions.merge(max_values, on=["partner_id"], how="left")
  conversions["count"] = "1"

  # timestamp to unix
  conversions["conversion_timestamp"] = [int(time.mktime(date_time.timetuple())) for date_time in conversions["conversion_timestamp"]]
  impressions["click_timestamp"] = [int(time.mktime(date_time.timetuple())) for date_time in impressions["click_timestamp"]]

  impressions["product_id_group"] = impressions["product_id"].apply(hash_to_buckets)
  impressions["filter"] = "product_group_id=" + impressions["product_id_group"].astype(str)

  conversions["product_id_group"] = conversions["product_id"].apply(hash_to_buckets)
  conversions["filter"] = "product_group_id=" + conversions["product_id_group"].astype(str)
  
  x = (
    conversions.groupby(["partner_id", "product_id_group"])
    .size()
    .reset_index(name="count")
)
  x["epsilon"] = x["count"].apply(get_epsilon_from_accuracy)
  x = x.drop(columns=["count"])
  conversions = conversions.merge(x, on=["partner_id", "product_id_group"], how="left")

  conversions["key"] = "product_group_id=" + conversions["product_id_group"].astype(str)

  impressions = impressions.sort_values(by=["click_timestamp"])
  conversions = conversions.sort_values(by=["conversion_timestamp"])
  
  impressions.to_csv("synthetic_impressions.csv", header=True, index=False)
  conversions.to_csv("synthetic_conversions.csv", header=True, index=False)


@app.command()
def create_dataset(
    omegaconf: str = "config.json",
): 
    omegaconf = OmegaConf.load(omegaconf)
    return create_synthetic_dataset(omegaconf)

if __name__ == "__main__":
    app()