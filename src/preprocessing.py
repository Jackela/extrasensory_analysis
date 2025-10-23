# src/preprocessing.py
# Functions for loading and preprocessing ExtraSensory data
# using the combined '.features_labels.csv' files and revised variable definitions.

import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import KBinsDiscretizer
import os
import warnings
import src.settings as settings # Import settings to use defined column names


def load_subject_data(uuid: str) -> pd.DataFrame:
    """
    Loads the combined features_labels CSV for a single subject.
    Uses DATA_PATH from settings.
    """
    file_path = os.path.join(settings.DATA_PATH, f"{uuid}.features_labels.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found for UUID {uuid} at {file_path}")

    # Explicitly set the timestamp column as index during loading
    data = pd.read_csv(file_path, index_col=settings.COL_TIMESTAMP)
    return data


def create_variables(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates the three critical time series variables from the loaded dataframe:
    - A: Activity Level (Based on COL_ACTIVITY_INPUT, z-scored, 5-bin quantile discretized)
    - S: Sitting State (Based on COL_SITTING, binary)
    - H_binned: Binned hour-of-day indicator using NUM_HOUR_BINS

    Returns aligned, non-NaN integer arrays for A, S, and H_binned.
    """

    # --- Input Validation ---
    required_cols = [settings.COL_ACTIVITY_INPUT, settings.COL_SITTING]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    # 1. Variable S (Sitting): Binary state from COL_SITTING
    series_S = df[settings.COL_SITTING].copy()
    # Handle potential NaNs in label column (fill with 0, assuming NaN means not sitting)
    series_S = series_S.fillna(0)

    # 2. Variable A (Activity): Based on COL_ACTIVITY_INPUT (Revised definition)
    continuous_A = df[settings.COL_ACTIVITY_INPUT].copy()

    # 3. Variable H (Hour of Day): From timestamp index
    timestamps = pd.to_datetime(df.index, unit='s')
    series_H = timestamps.hour.astype(int)

    # 4. Align, Clean, and Package
    aligned_df = pd.DataFrame({
        'S': series_S,
        'A_cont': continuous_A,
        'H': series_H
    })

    # Drop rows where the *continuous activity measure* is missing,
    # as this is essential before standardization/discretization.
    aligned_df = aligned_df.dropna(subset=['A_cont'])

    # Ensure sufficient data after dropping NaNs
    if len(aligned_df) < 200:
        raise ValueError(f"Insufficient data (N={len(aligned_df)}) after cleaning NaNs from activity column.")

    # 5. Final Preprocessing for Variable A (Revised Method)

    # Step 1: Z-score normalization *within subject* (Proposal Req: 15)
    # Apply to the cleaned continuous data
    zscored_A = zscore(aligned_df['A_cont'])

    # Step 2: Quantile Discretization (Proposal Req: 16, 17)
    # Use KBinsDiscretizer for 5 equal-frequency bins (quintiles).
    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')

    # Reshape for the discretizer
    reshaped_A = zscored_A.reshape(-1, 1)

    # Apply discretization, suppressing potential warnings about bin edges
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Ensure output is integer and flattened
        final_A = discretizer.fit_transform(reshaped_A).astype(int).flatten()

    # Final check on discretization output
    if np.max(final_A) == 0 and len(np.unique(final_A)) == 1:
        # Check if the original data had zero variance before concluding failure
        if aligned_df['A_cont'].nunique() > 1:
            raise ValueError("Discretization failed; resulted in only one bin despite variance in input.")
        else:
            # If input truly had no variance, discretization to one bin is expected but maybe not useful
            raise ValueError("Input data for activity has zero variance, cannot discretize meaningfully.")

    # 6. Get final, aligned, integer arrays
    final_S = aligned_df['S'].values.astype(int)
    hour_bins = settings.NUM_HOUR_BINS
    if hour_bins < 1:
        raise ValueError("NUM_HOUR_BINS in settings.py must be >= 1.")

    bin_edges = np.linspace(0, 24, hour_bins + 1)
    series_H_binned = pd.cut(
        aligned_df['H'],
        bins=bin_edges,
        right=False,
        labels=False,
        include_lowest=True
    ).astype(int)

    final_H_binned = series_H_binned.values.astype(int)

    # Ensure all arrays have the same length after processing
    assert len(final_A) == len(final_S) == len(final_H_binned), "Array lengths do not match after processing!"

    return final_A, final_S, final_H_binned
