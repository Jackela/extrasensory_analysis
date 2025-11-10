"""
Archived script: normalized mutual information (NMI) report (English-only).

@module nmi_analysis
@deprecated Historical exploratory script; not part of the production pipeline.
"""

import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

# --- 1) Load data ---
# Use the previously cleaned dataset from data_cleaning_and_mi_analysis.py
df = pd.read_csv('mvp_dataset.csv')

# --- 2) Compute NMI ---

# Label pairs to analyze
pairs_to_analyze = [
    ('label:LOC_home', 'label:SITTING'),
    ('label:LOC_home', 'label:FIX_walking'),
    ('label:SITTING', 'label:FIX_walking')
]

# Compute and store NMI values
nmi_results = {}
for col1, col2 in pairs_to_analyze:
    nmi_value = normalized_mutual_info_score(df[col1], df[col2])
    nmi_results[f"{col1} vs {col2}"] = nmi_value

# --- 3) Report ---

# Part I: NMI values
print("--- Normalized Mutual Information (NMI) Report ---")
print("NMI values for key label pairs (range 0-1):\n")
for pair, nmi_value in nmi_results.items():
    print(f"- {pair}: {nmi_value:.4f}")

# Sort for interpretation
sorted_nmi = sorted(nmi_results.items(), key=lambda item: item[1], reverse=True)

# Part II: Interpretation
print("\n--- Interpretation ---")
print("1) Strength ranking by NMI (desc):")
for i, (pair, nmi_value) in enumerate(sorted_nmi, 1):
    print(f"   {i}. {pair} (NMI = {nmi_value:.4f})")

print("\n2) Notes:")
print("   - Higher NMI indicates stronger association; mutually exclusive activities tend to show structure.")
