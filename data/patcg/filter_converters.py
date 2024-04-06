import pandas as pd

converters = pd.read_csv("advertisers/conversions.csv")
converters = converters[["device_id"]]
converters = converters.drop_duplicates('device_id', keep='last')

chunksize = 10000000
filtered_chunks = []

for i, chunk in enumerate(pd.read_csv("publishers/impressions.csv", chunksize=chunksize)):
    print(i*chunksize)
    filtered_chunk = chunk[chunk['device_id'].isin(converters['device_id'])]
    filtered_chunks.append(filtered_chunk)

filtered_df = pd.concat(filtered_chunks)
filtered_df.to_csv("advertisers/converter_impressions.csv", index=False)