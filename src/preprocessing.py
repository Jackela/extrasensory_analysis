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


def compute_sma(df: pd.DataFrame) -> np.ndarray:
    """
    Computes Signal Magnitude Area (SMA) from tri-axis accelerometer data.
    SMA = (|ax| + |ay| + |az|) / 3
    
    Returns continuous SMA values.
    """
    required_cols = [settings.COL_ACC_X, settings.COL_ACC_Y, settings.COL_ACC_Z]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing tri-axis columns for SMA: {missing}")
    
    sma = (np.abs(df[settings.COL_ACC_X]) + 
           np.abs(df[settings.COL_ACC_Y]) + 
           np.abs(df[settings.COL_ACC_Z])) / 3.0
    return sma.values


def compute_triaxis_variance(df: pd.DataFrame) -> np.ndarray:
    """
    Computes tri-axis variance metric from std columns.
    Variance = sqrt(std_x^2 + std_y^2 + std_z^2)
    
    Returns continuous variance values.
    """
    required_cols = [settings.COL_ACC_STD_X, settings.COL_ACC_STD_Y, settings.COL_ACC_STD_Z]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing std columns for variance: {missing}")
    
    variance = np.sqrt(df[settings.COL_ACC_STD_X]**2 + 
                       df[settings.COL_ACC_STD_Y]**2 + 
                       df[settings.COL_ACC_STD_Z]**2)
    return variance.values


def create_composite_feature(df: pd.DataFrame, mode: str = 'composite') -> np.ndarray:
    """
    Creates activity feature based on specified mode:
    - 'composite': 0.6*SMA + 0.4*variance (weighted blend)
    - 'sma_only': SMA only
    - 'variance_only': Variance only
    - 'magnitude_only': Original magnitude mean (baseline)
    
    Returns continuous feature values.
    """
    if mode == 'magnitude_only':
        if settings.COL_ACTIVITY_INPUT not in df.columns:
            raise ValueError(f"Missing column: {settings.COL_ACTIVITY_INPUT}")
        return df[settings.COL_ACTIVITY_INPUT].values
    
    elif mode == 'sma_only':
        return compute_sma(df)
    
    elif mode == 'variance_only':
        return compute_triaxis_variance(df)
    
    elif mode == 'composite':
        sma = compute_sma(df)
        variance = compute_triaxis_variance(df)
        # Weighted blend: 60% SMA, 40% variance
        return 0.6 * sma + 0.4 * variance
    
    else:
        raise ValueError(f"Unknown feature mode: {mode}. Must be one of {settings.FEATURE_MODES}")


def create_variables(df: pd.DataFrame, feature_mode: str = 'composite', hour_bins: int = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates the core time series variables from the loaded dataframe:
    - A: Activity Level (composite/SMA/variance/magnitude feature, z-scored, 5-bin quantile discretized)
    - S: Sitting State (Based on COL_SITTING, binary)
    - H_raw: Hour-of-day indicator in 24 bins (0-23)
    - H_binned: Hour-of-day indicator using hour_bins parameter

    Args:
        df: Input dataframe with ExtraSensory features
        feature_mode: Feature engineering mode ('composite', 'sma_only', 'variance_only', 'magnitude_only')
        hour_bins: Number of bins for H_binned (required, must be passed from config)

    Returns aligned, non-NaN integer arrays for A, S, H_raw, and H_binned.
    """
    
    if hour_bins is None:
        raise ValueError("hour_bins is required and must be passed from config file")

    # --- Input Validation ---
    required_cols = [settings.COL_SITTING]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    # 1. Variable S (Sitting): Binary state from COL_SITTING
    series_S = df[settings.COL_SITTING].copy()
    # Handle potential NaNs in label column (fill with 0, assuming NaN means not sitting)
    series_S = series_S.fillna(0)

    # 2. Variable A (Activity): Use composite feature based on mode
    continuous_A = create_composite_feature(df, mode=feature_mode)

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
    final_H_raw = aligned_df['H'].values.astype(int)
    if hour_bins < 1:
        raise ValueError(f"hour_bins must be >= 1, got {hour_bins}")

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
    assert len(final_A) == len(final_S) == len(final_H_raw) == len(final_H_binned), (
        "Array lengths do not match after processing!"
    )

    return final_A, final_S, final_H_raw, final_H_binned
