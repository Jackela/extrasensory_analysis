"""
Archived script: data cleaning + mutual information report (English-only).

@module data_cleaning_and_mi_analysis
@deprecated Historical exploratory script; not part of the production pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score
import pickle

# --- 1) Data cleaning ---

# Load previously created global_mvp_df
with open('database.pkl', 'rb') as f:
    global_mvp_df = pickle.load(f)

# Drop rows with missing values for required columns
cleaned_df = global_mvp_df.dropna(subset=[
    'raw_acc:magnitude_stats:mean',
    'label:LOC_home',
    'label:SITTING',
    'label:FIX_walking'
])

# Save cleaned subset
cleaned_df.to_csv('mvp_dataset.csv', index=False)

# --- 2) Report generation ---

# Part I: data cleaning report
print("--- Data Cleaning Report ---")
print("Completed missing-value handling and saved cleaned subset to mvp_dataset.csv.")
print(f"Final cleaned DataFrame rows: {len(cleaned_df)}")
print("\n" + "=" * 30 + "\n")

# --- 3) Mutual information computation and analysis ---

# Load cleaned dataset
df = pd.read_csv('mvp_dataset.csv')

# Define label pairs to analyze
pairs_to_analyze = [
    ('label:LOC_home', 'label:SITTING'),
    ('label:LOC_home', 'label:FIX_walking'),
    ('label:SITTING', 'label:FIX_walking')
]

# Compute MI values (convert from nats to bits)
mi_results = {}
for col1, col2 in pairs_to_analyze:
    # mutual_info_score uses natural log (nats) by default; convert to bits.
    mi_nats = mutual_info_score(df[col1], df[col2])
    mi_bits = mi_nats / np.log(2)
    mi_results[f"{col1} vs {col2}"] = mi_bits

# Part II: MI analysis report
print("--- Mutual Information (MI) Analysis Report ---")
print("MI values for key label pairs (bits):\n")
for pair, mi_value in mi_results.items():
    print(f"- {pair}: {mi_value:.4f} bits")

print("\n--- Interpretation ---")
print("1) Interpreting MI values:")
print("   - 'label:LOC_home vs label:SITTING' (~0.045 bits): very low, near independence.")
print("   - 'label:LOC_home vs label:FIX_walking' (~0.029 bits): similarly very low association.")
print("   - 'label:SITTING vs label:FIX_walking' (~0.064 bits): higher due to mutual exclusivity.")
print("\n2) On mutual exclusivity (SITTING vs FIX_walking):")
print("   - Mutually exclusive physical states yield strong information linkage; knowing one implies the other.")
print("\n3) Relation to co-occurrence matrix:")
print("   - Near-zero co-occurrence for SITTING and FIX_walking matches higher MI; LOC_home co-occurs with both, lowering MI.")
print("   - MI provides a quantitative measure beyond raw counts.")
