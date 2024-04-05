import os
import pandas as pd


def main():
    # Path to the directory containing Parquet files
    parquet_dir = '/mydata/cookiemonster/data/patcg/publisher_exposures/'

    # List to store DataFrame objects
    dfs = []

    # Iterate over each file in the directory
    for file in os.listdir(parquet_dir):
        if file.endswith('.pqt'):
            # Read Parquet file into a DataFrame
            df = pd.read_parquet(os.path.join(parquet_dir, file))
            dfs.append(df)

    # Concatenate all DataFrames into a single DataFrame
    merged_df = pd.concat(dfs)

    # Path to save the merged DataFrame as CSV
    csv_path = '/mydata/cookiemonster/data/patcg/impressions.csv'

    # Save the DataFrame as CSV
    merged_df.to_csv(csv_path, index=False)

    print(f"Merged DataFrame saved as CSV: {csv_path}")

if __name__ == '__main__':
    main()