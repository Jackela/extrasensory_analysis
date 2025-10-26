"""K selection via Active Information Storage (AIS).

Implements AIS-based selection of embedding dimension k per subject.
AIS measures self-prediction: AIS_k = I(X_t; X_{t-1:t-k})
"""
import numpy as np
import logging
from typing import Tuple, Dict
import jpype
from jpype.types import JArray, JInt

logger = logging.getLogger(__name__)


def compute_ais(series: np.ndarray, k: int, base: int, num_surrogates: int = 100) -> Tuple[float, float]:
    """Compute Active Information Storage for given k.
    
    Args:
        series: Discretized time series (int32)
        k: History length
        base: Alphabet size
        num_surrogates: Number of surrogates for significance testing
    
    Returns:
        (ais_value, p_value)
    """
    try:
        # JIDT ActiveInformationCalculatorDiscrete
        AISClass = jpype.JClass("infodynamics.measures.discrete.ActiveInformationCalculatorDiscrete")
        calc = AISClass(base, k)
        calc.initialise()
        
        # Convert to Java array
        ja = JArray(JInt)(series.astype(np.int32))
        calc.addObservations(ja)
        
        # Compute AIS
        ais = float(calc.computeAverageLocalOfObservations())
        
        # Significance test
        measure_dist = calc.computeSignificance(num_surrogates)
        p_value = float(measure_dist.pValue)
        
        return (ais, p_value)
    except Exception as e:
        logger.error(f"AIS computation failed for k={k}: {e}")
        return (np.nan, np.nan)


def select_k_via_ais(
    series: np.ndarray,
    base: int,
    k_range: list,
    num_surrogates: int = 100,
    criterion: str = 'max_ais',
    k_max: int = None,
    min_samples: int = None
) -> Dict[str, any]:
    """Select optimal k via AIS across k_range with optional constraints.
    
    Args:
        series: Discretized time series
        base: Alphabet size
        k_range: List of k values to test (e.g., [1,2,3,4,5,6])
        num_surrogates: Surrogates for AIS significance
        criterion: Selection criterion ('max_ais' or 'first_plateau')
        k_max: Optional hard cap on k (for computational feasibility)
        min_samples: Optional sample count for undersampling guard
    
    Returns:
        {
            'k_selected': int,
            'k_original': int,  # Before constraints applied
            'ais_values': {k: ais},
            'p_values': {k: p},
            'criterion': str,
            'capped': bool
        }
    """
    ais_values = {}
    p_values = {}
    
    for k in k_range:
        ais, p = compute_ais(series, k, base, num_surrogates)
        ais_values[k] = ais
        p_values[k] = p
        logger.debug(f"k={k}: AIS={ais:.6f}, p={p:.4f}")
    
    # Select k based on criterion
    if criterion == 'max_ais':
        # Choose k with maximum AIS
        valid_ais = {k: v for k, v in ais_values.items() if np.isfinite(v)}
        if not valid_ais:
            k_selected = k_range[0]  # Fallback to k=1
        else:
            k_selected = max(valid_ais, key=valid_ais.get)
    
    elif criterion == 'first_plateau':
        # Choose k where AIS stops increasing significantly
        k_selected = k_range[0]
        for i in range(len(k_range) - 1):
            k_curr, k_next = k_range[i], k_range[i + 1]
            ais_curr, ais_next = ais_values.get(k_curr, 0), ais_values.get(k_next, 0)
            if np.isfinite(ais_curr) and np.isfinite(ais_next):
                improvement = (ais_next - ais_curr) / (ais_curr + 1e-10)
                if improvement < 0.1:  # Less than 10% improvement
                    k_selected = k_curr
                    break
                k_selected = k_next
    else:
        k_selected = k_range[0]
    
    k_original = k_selected
    capped = False
    
    # Apply undersampling guard if min_samples provided
    if min_samples is not None:
        for k in sorted([kk for kk in k_range if kk <= k_selected], reverse=True):
            state_space = (base ** k) * (5 ** k)  # Assuming 5-bin activity
            samples_per_state = min_samples / max(state_space, 1)
            if samples_per_state >= 25:  # Minimum 25 samples per state
                if k < k_selected:
                    k_selected = k
                    capped = True
                    logger.warning(f"Undersampling guard: reduced k from {k_original} to {k}")
                break
    
    # Apply hard cap if k_max provided
    if k_max is not None and k_selected > k_max:
        k_selected = k_max
        capped = True
        logger.warning(f"Hard cap: reduced k from {k_original} to {k_max}")
    
    return {
        'k_selected': k_selected,
        'k_original': k_original,
        'ais_values': ais_values,
        'p_values': p_values,
        'criterion': criterion,
        'capped': capped
    }
