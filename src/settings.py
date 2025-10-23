# src/settings.py
# This file contains global constants for the project.

import os

# --- CRITICAL PATHS (USER MUST VERIFY) ---

# Set the path to the JIDT .jar file.
# Located within the jidt/ folder for clarity.
JIDT_JAR_PATH = "jidt/infodynamics.jar" 

# Set the path to the directory containing the combined '*.features_labels.csv' files.
DATA_PATH = "data/ExtraSensory.per_uuid_features_labels"

# --- OUTPUT PATH ---
# This is where the final aggregated results CSV will be saved.
RESULTS_FILE = "results/extrasensory_te_results.csv"

# --- ANALYSIS PARAMETERS (from Proposal) ---

# Number of surrogates for permutation testing. (Proposal Req: 24)
NUM_SURROGATES = 1000

# Maximum history length (k) to search for when optimizing AIS. (Proposal Req: 22)
MAX_K_AIS = 4

# Number of bins to discretize the hour-of-day variable for conditional TE.
NUM_HOUR_BINS = 4

# --- DATA COLUMN NAMES ---
# Explicitly define column names based on data inspection.
COL_TIMESTAMP = 'timestamp'
COL_ACTIVITY_INPUT = 'raw_acc:magnitude_stats:mean' # Confirmed - Deviation from proposal, based on available data
COL_SITTING = 'label:SITTING' # Confirmed
