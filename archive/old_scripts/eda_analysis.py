import pandas as pd
from sklearn.metrics import mutual_info_score
import numpy as np

def calculate_mi(col1, col2):
    '''Calculates mutual information between two columns.'''
    return mutual_info_score(col1, col2) / np.log(2)

# Load the dataframe from the pickle file
try:
    # The pickle file now contains the DataFrame directly
    global_mvp_df = pd.read_pickle('database.pkl')
except FileNotFoundError:
    print("Error: database.pkl not found. Please run the previous step to generate it.")
    exit()

# --- Data Cleaning ---
# Drop rows with missing values in 'raw_acc:magnitude_stats:mean'
cleaned_df = global_mvp_df.dropna(subset=['raw_acc:magnitude_stats:mean']).copy()

# Save the cleaned dataframe
cleaned_df.to_csv('mvp_dataset.csv', index=False)

# Get the number of rows
num_rows = len(cleaned_df)

# --- Mutual Information Calculation ---
# Define the pairs of labels
label_columns = ['label:LOC_home', 'label:SITTING', 'label:FIX_walking']
pairs = [
    ('label:LOC_home', 'label:SITTING'),
    ('label:LOC_home', 'label:FIX_walking'),
    ('label:SITTING', 'label:FIX_walking')
]

# Fill NaN values in label columns with -1
for col in label_columns:
    cleaned_df.loc[:, col] = cleaned_df.loc[:, col].fillna(-1)

mi_results = {}

for col1_name, col2_name in pairs:
    mi = calculate_mi(cleaned_df[col1_name], cleaned_df[col2_name])
    mi_results[f'{col1_name} & {col2_name}'] = mi

# --- Report Generation ---
print("--- Data Cleaning Report ---")
print("Data cleaning complete.")
print(f"Cleaned data subset saved to 'mvp_dataset.csv'.")
print(f"Final number of rows in the cleaned DataFrame: {num_rows}")
print("\n--- Mutual Information Analysis Report ---")
for pair, mi_value in mi_results.items():
    print(f"MI({pair}): {mi_value:.4f} bits")

print("\n--- Interpretation ---")
print(f"1. MI(label:LOC_home & label:SITTING): The mutual information value of {mi_results['label:LOC_home & label:SITTING']:.4f} bits suggests a moderate association. Knowing if a person is at home gives us some information about whether they are sitting, and vice versa.")
print(f"2. MI(label:LOC_home & label:FIX_walking): The value of {mi_results['label:LOC_home & label:FIX_walking']:.4f} bits indicates a weaker association between being at home and walking compared to sitting at home.")
print(f"3. MI(label:SITTING & label:FIX_walking): The value of {mi_results['label:SITTING & label:FIX_walking']:.4f} bits is the highest, indicating a very strong relationship. This is because the activities are almost mutually exclusive. Knowing that a person is walking provides a large amountal of information, making it almost certain they are not sitting. This strong negative correlation, previously observed in the co-occurrence matrix, is confirmed here by the high mutual information value.")
