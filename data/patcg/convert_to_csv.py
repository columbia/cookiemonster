import os
import pandas as pd


def main():
    # Path to the directory containing Parquet files
    # parquet_dir = '/mydata/cookiemonster/data/patcg/publishers/publisher_exposures/'
    parquet_dir = '/mydata/cookiemonster/data/patcg/advertisers/advertiser_conversions/'

    # List to store DataFrame objects
    dfs = []

    # Iterate over each file in the directory
    for i, file in enumerate(os.listdir(parquet_dir)):
        if file.endswith('.pqt'):
            # Read Parquet file into a DataFrame
            df = pd.read_parquet(os.path.join(parquet_dir, file))
            # print(df)
            # df = df.drop(['pub_profile_1', 'pub_profile_2', 'pub_profile_3',
            #         'pub_profile_4', 'pub_profile_5', 'pub_profile_6', 'pub_profile_7',
            #         'pub_profile_8', 'exp_record_id', 'pub_segment'], axis=1)
            df = df[["conv_timestamp", "device_id", "conv_amount"]]
            dfs.append(df)

        print(i, file)
    # Concatenate all DataFrames into a single DataFrame
    merged_df = pd.concat(dfs)

    # Path to save the merged DataFrame as CSV
    # csv_path = '/mydata/cookiemonster/data/patcg/publishers/impressions.csv'
    csv_path = '/mydata/cookiemonster/data/patcg/advertisers/conversions.csv'

    # Save the DataFrame as CSV
    merged_df.to_csv(csv_path, index=False)

    print(f"Merged DataFrame saved as CSV: {csv_path}")

if __name__ == '__main__':
    main()