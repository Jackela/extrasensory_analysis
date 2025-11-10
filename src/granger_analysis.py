"""
Granger causality baseline analysis using statsmodels VAR.

Provides a symmetric A↔S Granger test with basic lag selection
and a delta based on -log10(p) differences.

@module granger_analysis
"""

import logging
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
import warnings

logger = logging.getLogger(__name__)


def run_granger_causality(series_A: np.ndarray, series_S: np.ndarray, max_lag: int = 4) -> dict:
    """
    Compute Granger causality tests for A→S and S→A.

    Uses VAR with lag order selection and reports p-values and a delta.

    @param {np.ndarray} series_A - Activity series (discrete or continuous).
    @param {np.ndarray} series_S - Sitting series (binary or discrete).
    @param {int} max_lag - Maximum lag to test.
    @returns {dict} {'gc_A_to_S_pval','gc_S_to_A_pval','gc_optimal_lag','gc_Delta'}
    @pre len(series_A) == len(series_S) and sufficiently large for VAR.
    @post Returns NaN entries on failure; logs warnings for insufficient data.
    """
    results = {}
    
    try:
        # Combine series into 2D array for VAR: [A, S]
        data = np.column_stack([series_A, series_S])
        
        # Ensure sufficient data
        if len(data) < max_lag + 50:
            logger.warning(f"Insufficient data for Granger test: N={len(data)}, max_lag={max_lag}")
            return {
                'gc_A_to_S_pval': np.nan,
                'gc_S_to_A_pval': np.nan,
                'gc_optimal_lag': np.nan,
                'gc_Delta': np.nan
            }
        
        # Fit VAR model to select optimal lag
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = VAR(data)
            lag_order_results = model.select_order(maxlags=max_lag)
            optimal_lag = lag_order_results.aic
            
            if optimal_lag < 1:
                optimal_lag = 1
            if optimal_lag > max_lag:
                optimal_lag = max_lag
                
            results['gc_optimal_lag'] = int(optimal_lag)
        
        # Test A -> S (does A Granger-cause S?)
        # grangercausalitytests expects [effect, cause] ordering
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gc_A_to_S = grangercausalitytests(
                    data[:, [1, 0]],  # [S, A] - test if A causes S
                    maxlag=optimal_lag,
                    verbose=False
                )
                # Extract p-value from F-test at optimal lag
                p_A_to_S = gc_A_to_S[optimal_lag][0]['ssr_ftest'][1]
                results['gc_A_to_S_pval'] = float(p_A_to_S)
        except Exception as e:
            logger.debug(f"Granger A->S test failed: {e}")
            results['gc_A_to_S_pval'] = np.nan
        
        # Test S -> A (does S Granger-cause A?)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gc_S_to_A = grangercausalitytests(
                    data[:, [0, 1]],  # [A, S] - test if S causes A
                    maxlag=optimal_lag,
                    verbose=False
                )
                p_S_to_A = gc_S_to_A[optimal_lag][0]['ssr_ftest'][1]
                results['gc_S_to_A_pval'] = float(p_S_to_A)
        except Exception as e:
            logger.debug(f"Granger S->A test failed: {e}")
            results['gc_S_to_A_pval'] = np.nan
        
        # Compute Delta as -log10(p) difference (positive means A->S is stronger)
        if np.isfinite(results.get('gc_A_to_S_pval', np.nan)) and \
           np.isfinite(results.get('gc_S_to_A_pval', np.nan)):
            p_AS = max(results['gc_A_to_S_pval'], 1e-16)  # Avoid log(0)
            p_SA = max(results['gc_S_to_A_pval'], 1e-16)
            # Negative Delta means S->A is stronger (matching TE convention)
            results['gc_Delta'] = -np.log10(p_AS) - (-np.log10(p_SA))
        else:
            results['gc_Delta'] = np.nan
            
    except Exception as e:
        logger.error(f"Granger causality analysis failed: {e}")
        results = {
            'gc_A_to_S_pval': np.nan,
            'gc_S_to_A_pval': np.nan,
            'gc_optimal_lag': np.nan,
            'gc_Delta': np.nan
        }
    
    return results
