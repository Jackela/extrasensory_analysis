# src/settings.py
# Infrastructure paths and ExtraSensory dataset schema constants.
# All configuration parameters are in YAML config files (config/presets/).

import os

# --- INFRASTRUCTURE PATHS ---

# JIDT library path (external dependency)
JIDT_JAR_PATH = "jidt/infodynamics.jar"

# ExtraSensory dataset location
DATA_PATH = "data/ExtraSensory.per_uuid_features_labels"

# Legacy output path (use config['out_dir'] instead)
RESULTS_FILE = "results/extrasensory_te_results.csv"

# --- DATASET SCHEMA ---

# Minimum samples per hour bin threshold (dataset-specific)
MIN_SAMPLES_PER_BIN = 30

# ExtraSensory dataset column names (dataset constants)
COL_TIMESTAMP = 'timestamp'
COL_ACTIVITY_INPUT = 'raw_acc:magnitude_stats:mean'
COL_SITTING = 'label:SITTING'

# Tri-axis accelerometer columns for composite features
COL_ACC_X = 'raw_acc:3d:mean_x'
COL_ACC_Y = 'raw_acc:3d:mean_y'
COL_ACC_Z = 'raw_acc:3d:mean_z'
COL_ACC_STD_X = 'raw_acc:3d:std_x'
COL_ACC_STD_Y = 'raw_acc:3d:std_y'
COL_ACC_STD_Z = 'raw_acc:3d:std_z'
