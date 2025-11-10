"""
Symbolic Transfer Entropy using ordinal pattern encoding.

Encodes continuous series via Bandt–Pompe ordinal patterns, then
computes discrete TE using JIDT.

@module symbolic_te
"""

import logging
import numpy as np
from itertools import permutations

logger = logging.getLogger(__name__)


def ordinal_pattern_encode(series: np.ndarray, dim: int = 3, delay: int = 1) -> np.ndarray:
    """
    Encode a time series into ordinal patterns (Bandt–Pompe).

    @param {np.ndarray} series - Input continuous time series.
    @param {int} dim - Embedding dimension (pattern length).
    @param {int} delay - Time delay between elements.
    @returns {np.ndarray} Array of pattern indices in [0, dim!-1].
    @throws {ValueError} If the series is too short for encoding.
    @pre n - (dim-1)*delay >= 10.
    @post Returns int array of length n_patterns.
    """
    n = len(series)
    n_patterns = n - (dim - 1) * delay
    
    if n_patterns < 10:
        raise ValueError(f"Series too short for ordinal encoding: N={n}, dim={dim}, delay={delay}")
    
    # Create all possible permutations and map to indices
    all_perms = list(permutations(range(dim)))
    perm_to_idx = {perm: i for i, perm in enumerate(all_perms)}
    
    patterns = np.zeros(n_patterns, dtype=int)
    
    for i in range(n_patterns):
        # Extract embedded vector
        indices = [i + j * delay for j in range(dim)]
        vec = series[indices]
        
        # Get ordinal pattern (rank ordering)
        rank = np.argsort(np.argsort(vec))
        pattern = tuple(rank)
        
        patterns[i] = perm_to_idx.get(pattern, 0)
    
    return patterns


def run_symbolic_te_analysis(series_A: np.ndarray, series_S: np.ndarray,
                              k_A: int, k_S: int, tau: int = 1, num_surrogates: int = 1000) -> dict:
    """
    Compute Symbolic TE using JIDT after ordinal encoding.

    @param {np.ndarray} series_A - Activity series (continuous).
    @param {np.ndarray} series_S - Sitting series (discrete/continuous).
    @param {int} k_A - History for A (symbolic).
    @param {int} k_S - History for S (symbolic).
    @param {int} tau - Delay parameter.
    @param {int} num_surrogates - Surrogates for significance.
    @returns {dict} {'STE(A->S)','p_ste(A->S)','STE(S->A)','p_ste(S->A)','Delta_STE'}
    @pre Series are long enough for ordinal encoding; k symbolic parameters are small.
    @post Returns NaN values on failure; logs errors.
    """
    from src.jidt_adapter import SymbolicTE
    from src.params import STEParams
    import gc
    
    results = {}
    
    try:
        # Use smaller k for symbolic sequences (already embedded)
        k_symbolic = min(2, k_A, k_S)
        
        # --- Compute STE(A -> S) ---
        # STE algorithm constants (ordinal pattern parameters)
        STE_ORDINAL_DIM = 3  # Ordinal pattern dimension
        STE_ORDINAL_DELAY = 1  # Ordinal pattern delay
        
        params_A2S = STEParams(
            ordinal_dim=STE_ORDINAL_DIM,
            ordinal_delay=STE_ORDINAL_DELAY,
            k_source=k_symbolic,
            k_dest=k_symbolic,
            tau=tau,
            num_surrogates=num_surrogates,
            seed=42
        )
        calc_A2S = SymbolicTE(params_A2S)
        ste_A_to_S, p_A_to_S = calc_A2S.compute(series_A, series_S)
        
        results['STE(A->S)'] = ste_A_to_S
        results['p_ste(A->S)'] = p_A_to_S
        
        # --- Compute STE(S -> A) ---
        params_S2A = STEParams(
            ordinal_dim=STE_ORDINAL_DIM,
            ordinal_delay=STE_ORDINAL_DELAY,
            k_source=k_symbolic,
            k_dest=k_symbolic,
            tau=tau,
            num_surrogates=num_surrogates,
            seed=42
        )
        calc_S2A = SymbolicTE(params_S2A)
        ste_S_to_A, p_S_to_A = calc_S2A.compute(series_S, series_A)
        
        results['STE(S->A)'] = ste_S_to_A
        results['p_ste(S->A)'] = p_S_to_A
        
        # --- Compute Delta_STE = A→S − S→A ---
        if np.isfinite(ste_A_to_S) and np.isfinite(ste_S_to_A):
            results['Delta_STE'] = ste_A_to_S - ste_S_to_A
        else:
            results['Delta_STE'] = np.nan
    
    except Exception as e:
        logger.error(f"Symbolic TE failed: {e}")
        results = {
            'STE(A->S)': np.nan,
            'p_ste(A->S)': np.nan,
            'STE(S->A)': np.nan,
            'p_ste(S->A)': np.nan,
            'Delta_STE': np.nan
        }
    
    gc.collect()
    return results
