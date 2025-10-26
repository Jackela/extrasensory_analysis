"""Unified FDR correction utilities.

Implements per-(family, tau) Benjamini-Hochberg FDR correction
with comprehensive p→q column generation.
"""
import pandas as pd
import numpy as np
from statsmodels.stats.multitest import fdrcorrection
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


def apply_fdr_per_family_tau(
    df: pd.DataFrame,
    p_cols: List[str],
    q_cols: List[str],
    family: str,
    tau_col: str = 'tau',
    tau_values: List[int] = [1, 2],
    alpha: float = 0.05
) -> pd.DataFrame:
    """Apply BH-FDR correction per (family, tau) with full p→q mapping.
    
    Args:
        df: DataFrame with p-values
        p_cols: List of p-value column names (e.g., ['p_A2S', 'p_S2A', 'p_Delta'])
        q_cols: List of q-value column names (e.g., ['q_A2S', 'q_S2A', 'q_Delta'])
        family: Method family name (TE, CTE, STE, GC)
        tau_col: Column name for tau
        tau_values: List of tau values to process
        alpha: FDR significance level
    
    Returns:
        DataFrame with q-value columns added
    """
    if len(df) == 0:
        return df
    
    # Initialize q columns with NaN
    for q_col in q_cols:
        df[q_col] = np.nan
    
    # Apply FDR separately for each tau
    for tau_val in tau_values:
        if tau_col not in df.columns:
            # No tau column (e.g., GC), treat all as one group
            mask_all = df.index
            n_tests_total = 0
            n_sig_total = 0
            
            for p_col, q_col in zip(p_cols, q_cols):
                if p_col not in df.columns:
                    continue
                
                mask = df[p_col].notna()
                if mask.sum() == 0:
                    continue
                
                _, q_vals = fdrcorrection(df.loc[mask, p_col].values, alpha=alpha)
                df.loc[mask, q_col] = q_vals
                
                n_tests_total += mask.sum()
                n_sig_total += (q_vals < alpha).sum()
            
            logger.info(f"FDR {family}: {n_tests_total} tests, {n_sig_total} significant @ alpha={alpha}")
            break
        
        else:
            # Per-tau FDR
            mask_tau = (df[tau_col] == tau_val)
            if mask_tau.sum() == 0:
                continue
            
            n_tests_total = 0
            n_sig_total = 0
            
            for p_col, q_col in zip(p_cols, q_cols):
                if p_col not in df.columns:
                    continue
                
                mask = mask_tau & df[p_col].notna()
                if mask.sum() == 0:
                    continue
                
                _, q_vals = fdrcorrection(df.loc[mask, p_col].values, alpha=alpha)
                df.loc[mask, q_col] = q_vals
                
                n_tests_total += mask.sum()
                n_sig_total += (q_vals < alpha).sum()
            
            logger.info(f"FDR {family} tau={tau_val}: {n_tests_total} tests, {n_sig_total} significant @ alpha={alpha}")
    
    return df


def compute_delta_pvalue(
    df: pd.DataFrame,
    delta_col: str,
    p_delta_col: str,
    q_delta_col: str = None,
    method: str = 'wilcoxon',
    alpha: float = 0.05
) -> Tuple[float, float]:
    """Compute group-level p-value for Delta via Wilcoxon signed-rank test.
    
    Args:
        df: DataFrame with Delta values
        delta_col: Column name for Delta (e.g., 'Delta_TE')
        p_delta_col: Column name to store p-value
        q_delta_col: Column name to store q-value (optional)
        method: Test method ('wilcoxon' or 'ttest')
        alpha: Significance level
    
    Returns:
        (group_median_delta, p_value)
    """
    from scipy.stats import wilcoxon, ttest_1samp
    
    if delta_col not in df.columns or len(df) == 0:
        return (np.nan, np.nan)
    
    delta_vals = df[delta_col].dropna()
    if len(delta_vals) < 3:
        return (float(delta_vals.median()) if len(delta_vals) > 0 else np.nan, np.nan)
    
    median_delta = float(delta_vals.median())
    
    if method == 'wilcoxon':
        try:
            stat, p_val = wilcoxon(delta_vals, alternative='two-sided')
            return (median_delta, float(p_val))
        except:
            return (median_delta, np.nan)
    
    elif method == 'ttest':
        try:
            stat, p_val = ttest_1samp(delta_vals, 0, alternative='two-sided')
            return (median_delta, float(p_val))
        except:
            return (median_delta, np.nan)
    
    return (median_delta, np.nan)
