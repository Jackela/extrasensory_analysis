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

# DEPRECATED: Moved to YAML config['surrogates']
# NUM_SURROGATES = 1000

# DEPRECATED: Moved to YAML config['k_selection']['k_max']
# MAX_K_AIS = 4

# Minimum samples required per bin to avoid low-n warnings
# NOTE: hour_bins is now configured in YAML presets/configs (typically 6 or 24)
MIN_SAMPLES_PER_BIN = 30

# DEPRECATED: Robustness check never called - would need config if re-enabled
# EMBEDDING_K_GRID = [-1, 0, 1]
# EMBEDDING_TAU_VALUES = [1, 2]

# DEPRECATED: STE parameters - should move to config if STE analysis expanded
# STE_EMBEDDING_DIM = 3
# STE_DELAY = 1

# --- DATA COLUMN NAMES ---
# Explicitly define column names based on data inspection.
COL_TIMESTAMP = 'timestamp'
COL_ACTIVITY_INPUT = 'raw_acc:magnitude_stats:mean' # Confirmed - Deviation from proposal, based on available data
COL_SITTING = 'label:SITTING' # Confirmed

# Tri-axis accelerometer columns for composite features
COL_ACC_X = 'raw_acc:3d:mean_x'
COL_ACC_Y = 'raw_acc:3d:mean_y'
COL_ACC_Z = 'raw_acc:3d:mean_z'
COL_ACC_STD_X = 'raw_acc:3d:std_x'
COL_ACC_STD_Y = 'raw_acc:3d:std_y'
COL_ACC_STD_Z = 'raw_acc:3d:std_z'

# DEPRECATED: Moved to YAML config['feature_modes']
# FEATURE_MODES = ['composite', 'sma_only', 'variance_only', 'magnitude_only']
