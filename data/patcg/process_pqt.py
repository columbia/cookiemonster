import os
import pandas as pd

publishers = os.path.join(os.path.dirname(__file__), "publishers")
advertisers = os.path.join(os.path.dirname(__file__), "advertisers")


def convert_impressions_to_csv():
    parquet_dir = os.path.join(publishers, "publisher_exposures")
    dfs = []
    for i, file in enumerate(os.listdir(parquet_dir)):
        if file.endswith(".pqt"):
            df = pd.read_parquet(os.path.join(parquet_dir, file))
            df = df[["exp_timestamp", "device_id"]]
            dfs.append(df)
        print(i, file)
    return pd.concat(dfs)


def convert_conversions_to_csv():
    parquet_dir = os.path.join(advertisers, "advertiser_conversions")
    dfs = []
    for i, file in enumerate(os.listdir(parquet_dir)):
        if file.endswith(".pqt"):
            df = pd.read_parquet(os.path.join(parquet_dir, file))
            df = df[["conv_timestamp", "device_id", "conv_attribute_2", "conv_amount"]]
            dfs.append(df)
        print(i, file)
    return pd.concat(dfs)


def convert_to_csv():
    print("writing impressions...")
    impressions = convert_impressions_to_csv()
    impressions.to_csv(os.path.join(publishers, "impressions.csv"), index=False)

    print("writing conversions...")
    conversions = convert_conversions_to_csv()
    conversions.to_csv(os.path.join(advertisers, "conversions.csv"), index=False)


def filter_converters():
    conversions = pd.read_csv(os.path.join(advertisers, "conversions.csv"))
    converters = conversions[["device_id"]]

    print("Dropping duplicate converters...")
    converters = converters.drop_duplicates("device_id", keep="last")

    # Filter out users who never converted from impressions
    chunksize = 10000000
    filtered_chunks = []

    print("Filtering out non-converters...")

    for i, chunk in enumerate(
        pd.read_csv(os.path.join(publishers, "impressions.csv"), chunksize=chunksize)
    ):
        print(i * chunksize)
        filtered_chunk = chunk[chunk["device_id"].isin(converters["device_id"])]
        filtered_chunks.append(filtered_chunk)
    filtered_impressions = pd.concat(filtered_chunks)

    print("Renaming users...")
    renamed_converters = converters.reset_index()
    renamed_converters["i"] = renamed_converters.index

    print("renamed_converters", renamed_converters)

    print("Renaming converters...")
    conversions = renamed_converters.merge(conversions, how="inner", on="device_id")
    conversions["device_id"] = conversions["i"]
    conversions = conversions.drop(columns=["i", "index"])

    conversions.to_csv(
        os.path.join(advertisers, "renamed_conversions.csv"), index=False
    )

    print("Renaming exposed users...")
    filtered_impressions = renamed_converters.merge(
        filtered_impressions, how="inner", on="device_id"
    )
    filtered_impressions["device_id"] = filtered_impressions["i"]
    filtered_impressions = filtered_impressions.drop(columns=["i", "index"])
    filtered_impressions.to_csv(
        os.path.join(publishers, "renamed_filtered_impressions.csv"), index=False
    )


def main():

    convert_to_csv()
    filter_converters()


if __name__ == "__main__":
    main()
