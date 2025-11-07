"""
Loading and preprocessing for ExtraSensory data (English-only).

Defines feature engineering and variable construction used by TE/CTE.
Contracts use JSDoc-style with DbC elements for clarity.

@module preprocessing
"""

import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import KBinsDiscretizer
import os
import warnings
import src.settings as settings # Import settings to use defined column names


def load_subject_data(uuid: str) -> pd.DataFrame:
    """
    Load the combined features_labels CSV for a single subject.

    @param {str} uuid - Subject UUID (filename stem).
    @returns {pd.DataFrame} DataFrame indexed by timestamp.
    @throws {FileNotFoundError} If the subject file does not exist.
    @pre `settings.DATA_PATH` points to ExtraSensory data root.
    @post DataFrame index is timestamps (seconds since epoch).
    """
    file_path = os.path.join(settings.DATA_PATH, f"{uuid}.features_labels.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found for UUID {uuid} at {file_path}")

    # Explicitly set the timestamp column as index during loading
    data = pd.read_csv(file_path, index_col=settings.COL_TIMESTAMP)
    return data


def compute_sma(df: pd.DataFrame) -> np.ndarray:
    """
    Compute Signal Magnitude Area (SMA) from tri-axis accelerometer.

    SMA = (|ax| + |ay| + |az|) / 3.

    @param {pd.DataFrame} df - Data containing tri-axis accelerometer columns.
    @returns {np.ndarray} Continuous SMA values aligned to df rows.
    @throws {ValueError} If required columns are missing.
    @pre Columns present: settings.COL_ACC_X/Y/Z.
    @post Returns 1-D float array of length len(df).
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
    Compute tri-axis variance metric from per-axis standard deviations.

    Variance = sqrt(std_x^2 + std_y^2 + std_z^2).

    @param {pd.DataFrame} df - Data with std columns.
    @returns {np.ndarray} Continuous variance values aligned to df rows.
    @throws {ValueError} If required std columns are missing.
    @pre Columns present: settings.COL_ACC_STD_X/Y/Z.
    @post Returns 1-D float array of length len(df).
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
    Create activity feature per mode.

    Modes:
    - 'composite': 0.6*SMA + 0.4*variance
    - 'sma_only': SMA only
    - 'variance_only': Variance only
    - 'magnitude_only': Raw magnitude mean

    @param {pd.DataFrame} df - Input data.
    @param {str} mode - Feature mode.
    @returns {np.ndarray} Continuous feature values.
    @throws {ValueError} If required columns are missing or mode unknown.
    @pre df contains mode-specific columns; mode is valid.
    @post Returns 1-D float array.
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
        raise ValueError(f"Unknown feature mode: {mode}. Must be one of ['composite', 'sma_only', 'variance_only', 'magnitude_only']")


def create_variables(
    df: pd.DataFrame,
    feature_mode: str = 'composite',
    hour_bins: int = None,
    a_bins: int = 5,
    s_mode: str = 'binary'
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct aligned variables A, S, H_raw, H_binned from input dataframe.

    - A: z-scored activity feature → quantile discretized into `a_bins` bins (0..a_bins-1)
    - S: sitting label (binary). Other modes may be added in future.
    - H_raw: hour-of-day (0..23)
    - H_binned: hour-of-day binned into `hour_bins`

    @param {pd.DataFrame} df - Input dataframe.
    @param {str} feature_mode - Feature engineering mode.
    @param {int} hour_bins - Number of bins for H_binned (required).
    @param {int} a_bins - Number of quantile bins for A (default: 5).
    @param {str} s_mode - 'binary' (default). Placeholder for future S discretizations.
    @returns {(np.ndarray,np.ndarray,np.ndarray,np.ndarray)} A,S,H_raw,H_binned integer arrays.
    @throws {ValueError} If required columns missing or insufficient data.
    @pre hour_bins >= 1 and df contains COL_SITTING and feature columns.
    @post All returned arrays are aligned and equal length.
    """
    
    if hour_bins is None:
        raise ValueError("hour_bins is required and must be passed from config file")
    if not isinstance(a_bins, int) or a_bins < 1:
        raise ValueError(f"a_bins must be int >= 1, got {a_bins}")
    if s_mode not in ('binary', 'quantile3'):
        # Supported modes: 'binary' (raw label) and 'quantile3' (rolling-mean proxy discretized into 3 quantiles)
        raise NotImplementedError(f"s_mode '{s_mode}' is not implemented; supported: 'binary', 'quantile3'")

    # --- Input Validation ---
    required_cols = [settings.COL_SITTING]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    # 1. Variable S (Sitting): Binary label or 3-quantile discretized proxy
    series_S = df[settings.COL_SITTING].copy()
    # Handle potential NaNs in label column (fill with 0, assuming NaN means not sitting)
    series_S = series_S.fillna(0)
    
    if s_mode == 'quantile3':
        # Derive a continuous-like proxy via centered 5-sample rolling mean
        # (Used as recent fraction-of-time sitting), then 3-quantile discretize
        s_proxy = series_S.rolling(window=5, center=True, min_periods=1).mean()
        # Temporarily place in dataframe for alignment with A/H later on
        series_S_quant = s_proxy

    # 2. Variable A (Activity): Use composite feature based on mode
    continuous_A = create_composite_feature(df, mode=feature_mode)

    # 3. Variable H (Hour of Day): From timestamp index
    timestamps = pd.to_datetime(df.index, unit='s')
    series_H = timestamps.hour.astype(int)

    # 4. Align, Clean, and Package
    aligned_df = pd.DataFrame({
        'S': series_S if s_mode == 'binary' else series_S_quant,
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
    # Use KBinsDiscretizer for equal-frequency bins (quantiles)
    discretizer = KBinsDiscretizer(n_bins=a_bins, encode='ordinal', strategy='quantile')

    # Reshape for the discretizer
    reshaped_A = zscored_A.reshape(-1, 1)

    # Apply discretization, suppressing potential warnings about bin edges
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Ensure output is integer and flattened
        final_A = discretizer.fit_transform(reshaped_A).astype(int).flatten()
    # Defensive clamp to [0, a_bins-1] to avoid boundary rounding issues
    final_A = np.clip(final_A, 0, a_bins - 1)

    # Final check on discretization output
    if np.max(final_A) == 0 and len(np.unique(final_A)) == 1:
        # Check if the original data had zero variance before concluding failure
        if aligned_df['A_cont'].nunique() > 1:
            raise ValueError("Discretization failed; resulted in only one bin despite variance in input.")
        else:
            # If input truly had no variance, discretization to one bin is expected but maybe not useful
            raise ValueError("Input data for activity has zero variance, cannot discretize meaningfully.")

    # 6. Get final, aligned, integer arrays
    if s_mode == 'binary':
        final_S = aligned_df['S'].values.astype(int)
    else:
        # Discretize the proxy into 3 equal-frequency bins (quantiles)
        s_discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
        s_vals = aligned_df['S'].astype(float).to_numpy().reshape(-1, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final_S = s_discretizer.fit_transform(s_vals).astype(int).flatten()
        # Defensive clamp to [0, 2] for 3-quantile discretization
        final_S = np.clip(final_S, 0, 2)
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
