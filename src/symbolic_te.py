# src/symbolic_te.py
# Symbolic Transfer Entropy using ordinal pattern encoding

import logging
import numpy as np
from itertools import permutations
import src.settings as settings

logger = logging.getLogger(__name__)


def ordinal_pattern_encode(series: np.ndarray, dim: int = 3, delay: int = 1) -> np.ndarray:
    """
    Encodes time series into ordinal patterns (Bandt-Pompe symbolization).
    
    Args:
        series: Input time series (continuous)
        dim: Embedding dimension (pattern length)
        delay: Time delay between elements
        
    Returns:
        Array of ordinal pattern indices (0 to dim!-1)
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


def compute_ste_with_jidt(patterns_source: np.ndarray, patterns_dest: np.ndarray,
                           k: int, base: int, tau: int = 1) -> tuple[float, float]:
    """
    Computes Symbolic Transfer Entropy using JIDT on ordinal patterns.
    
    Args:
        patterns_source: Ordinal pattern sequence (source)
        patterns_dest: Ordinal pattern sequence (destination)
        k: History length
        base: Alphabet size (dim! for ordinal patterns)
        tau: Time delay parameter (default=1)
        
    Returns:
        (STE value, p-value)
    """
    try:
        # Import JIDT classes
        import src.analysis as analysis
        analysis.start_jvm()
        classes = analysis.get_jidt_classes()
        TECalculator = classes["TE"]
        
        # Convert to Java arrays
        java_source = analysis.java_array_int(patterns_source)
        java_dest = analysis.java_array_int(patterns_dest)
        
        # Calculate TE on symbolic sequences with tau parameter
        calc = TECalculator(base, k, k, 1, tau)  # (base, k_dest, k_source, delay_dest, delay_source)
        calc.initialise()
        calc.addObservations(java_source, java_dest)
        ste = calc.computeAverageLocalOfObservations()
        
        # Compute significance
        measure_dist = calc.computeSignificance(settings.NUM_SURROGATES)
        p_val = measure_dist.pValue
        
        return (float(ste) if np.isfinite(ste) else np.nan,
                float(p_val) if np.isfinite(p_val) else np.nan)
        
    except Exception as e:
        logger.error(f"STE calculation failed: {e}")
        return (np.nan, np.nan)


def run_symbolic_te_analysis(series_A: np.ndarray, series_S: np.ndarray,
                              k_A: int, k_S: int, tau: int = 1, num_surrogates: int = 1000) -> dict:
    """
    Computes Symbolic Transfer Entropy using JIDT adapter.
    Returns harmonized Delta=A→S−S→A.
    
    Args:
        series_A: Activity time series (continuous, will be symbolized)
        series_S: Sitting time series (discrete, will be symbolized)
        k_A: History length for A
        k_S: History length for S
        tau: Time delay parameter (default=1)
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
