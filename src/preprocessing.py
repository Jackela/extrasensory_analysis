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
    hour_bins: int = 6,
    a_bins: int = 5,
    s_mode: str = 'binary'
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct aligned variables A, S, H_raw, H_binned from input dataframe (Discrete path).

    - A: z-scored composite feature → 5-quantile discretization → int in {0,1,2,3,4}
    - S: binary sitting label → int in {0,1}
    - H_raw: hour-of-day (0..23)
    - H_binned: 6-bin hour-of-day (4-hour chunks) → int in {0,1,2,3,4,5}
    """
    if hour_bins is None:
        hour_bins = 6
    if hour_bins != 6:
        # Enforce the plan's 6-bin requirement
        hour_bins = 6
    if a_bins != 5:
        a_bins = 5
    if s_mode != 'binary':
        raise NotImplementedError("Only binary S is supported in the constrained discrete pipeline")

    # Validate required columns
    if settings.COL_SITTING not in df.columns:
        raise ValueError(f"Missing required column: {settings.COL_SITTING}")

    # S (binary)
    series_S = df[settings.COL_SITTING].fillna(0).astype(int)

    # A (composite -> z-score -> 5-quantile discretization)
    A_cont = create_composite_feature(df, mode=feature_mode).astype(float)
    zA = zscore(A_cont, nan_policy='omit')

    # H (hour of day) → 6-bin
    timestamps = pd.to_datetime(df.index, unit='s')
    H_raw = timestamps.hour.astype(int)
    # Build preliminary frame to drop NaNs in zA before discretization
    prelim = pd.DataFrame({'A_z': zA, 'S': series_S.values, 'H_raw': H_raw.values})
    prelim = prelim.dropna(subset=['A_z'])

    # Discretize A on cleaned rows only
    reshaped = prelim['A_z'].to_numpy().reshape(-1, 1)
    discretizer = KBinsDiscretizer(n_bins=a_bins, encode='ordinal', strategy='quantile')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a_disc = discretizer.fit_transform(reshaped).astype(int).flatten()
    a_disc = np.clip(a_disc, 0, a_bins - 1)

    # H 6-bin on cleaned rows
    bin_edges = np.linspace(0, 24, hour_bins + 1)
    h_binned = pd.cut(prelim['H_raw'], bins=bin_edges, right=False, labels=False, include_lowest=True).astype(int)

    aligned = pd.DataFrame({
        'A': a_disc,
        'S': prelim['S'].astype(int).to_numpy(),
        'H_raw': prelim['H_raw'].astype(int).to_numpy(),
        'H_bin': h_binned.to_numpy().astype(int)
    })
    aligned = aligned.dropna()
    if len(aligned) < 200:
        raise ValueError(f"Insufficient data (N={len(aligned)}) after preprocessing.")

    A_final = aligned['A'].to_numpy().astype(int)
    S_final = aligned['S'].to_numpy().astype(int)
    H_raw_final = aligned['H_raw'].to_numpy().astype(int)
    H_binned_final = aligned['H_bin'].to_numpy().astype(int)
    assert len(A_final) == len(S_final) == len(H_raw_final) == len(H_binned_final)
    return A_final, S_final, H_raw_final, H_binned_final
