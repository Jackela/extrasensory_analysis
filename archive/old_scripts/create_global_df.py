import pandas as pd
import os
import glob

# Path to the directory containing user data
data_path = 'ExtraSensory.per_uuid_features_labels'
all_files = glob.glob(os.path.join(data_path, '*', '*.csv'))

# Check if there are any files to process
if not all_files:
    print(f"No CSV files found in the directory: {data_path}")
else:
    # Load and concatenate all csv files
    df_list = [pd.read_csv(file) for file in all_files]
    global_mvp_df = pd.concat(df_list, ignore_index=True)

    # Save the combined dataframe to a pickle file
    global_mvp_df.to_pickle('database.pkl')

    print(f"Successfully created and saved global_mvp_df to database.pkl")
    print(f"Number of rows in global_mvp_df: {len(global_mvp_df)}")
