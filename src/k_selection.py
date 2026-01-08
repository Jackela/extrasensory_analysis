"""
K selection utilities.

Provides:
- AIS-based selection for discrete series (legacy)
- Ragwitz criterion selection for continuous series

@module k_selection
"""
import numpy as np
import logging
from typing import Tuple, Dict
from sklearn.neighbors import NearestNeighbors
import jpype
from jpype.types import JArray, JInt

logger = logging.getLogger(__name__)


def compute_ais(series: np.ndarray, k: int, base: int, num_surrogates: int = 100) -> Tuple[float, float]:
    """
    Compute Active Information Storage (AIS) for a given k.

    @param {np.ndarray} series - Discretized time series (int32).
    @param {int} k - History length.
    @param {int} base - Alphabet size.
    @param {int} num_surrogates - Surrogates for significance.
    @returns {[float,float]} (ais_value, p_value) or (NaN, NaN) on failure.
    @pre len(series) > 0 and k >= 1.
    @post Returns finite values or NaN on errors.
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
    """
    Select optimal k via AIS across k_range with optional constraints.

    @param {np.ndarray} series - Discretized series.
    @param {int} base - Alphabet size.
    @param {list} k_range - k values to evaluate.
    @param {int} num_surrogates - Surrogates for AIS significance.
    @param {str} criterion - 'max_ais' or 'first_plateau'.
    @param {int|null} k_max - Hard cap on k.
    @param {int|null} min_samples - For undersampling guard (samples availability).
    @returns {dict} {'k_selected','k_original','ais_values','p_values','criterion','capped'}
    @pre k_range non-empty; base >= 2.
    @post Returns a consistent selection record; may set 'capped' when adjusted.
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


def select_k_via_ragwitz(
    series: np.ndarray,
    k_range: list,
    tau: int = 1,
    n_neighbors: int = 1,
    max_samples: int = 20000,
) -> Dict[str, any]:
    """
    Select embedding dimension k for a continuous series using a Ragwitz-style criterion.

    Heuristic implementation: choose k minimizing 1-step prediction MSE via nearest-neighbor prediction
    in the delay-embedded space (delay=tau).

    @param series {np.ndarray} Continuous series (float64)
    @param k_range {list[int]} Candidate k values
    @param tau {int} Delay between lags (default 1)
    @param n_neighbors {int} Number of neighbors (default 1 for local-constant)
    @param max_samples {int} Optional cap for computational cost
    @returns {dict} {'k_selected','mse':{k:mse}}
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n < (max(k_range) + 2) * tau + 1:
        # Too short; fallback
        return {'k_selected': k_range[0], 'mse': {}}

    # Optionally subsample to limit cost
    if n > max_samples:
        idx = np.linspace(0, n - 1, max_samples).astype(int)
        x = x[idx]
        n = len(x)

    mse_map: Dict[int, float] = {}
    for k in k_range:
        # Build embedded vectors X_t = [x_t, x_{t-tau}, ..., x_{t-(k-1)tau}]
        T = n - (k * tau + tau)  # ensure future at t+tau exists
        if T <= 1:
            mse_map[k] = np.inf
            continue
        embed = np.stack([x[(k-1-i)*tau : (k-1-i)*tau + T] for i in range(k)], axis=1)
        target_future = x[k*tau + tau : k*tau + tau + T]

        # Nearest neighbor in embedded space (exclude self)
        nn = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(embed)), algorithm='auto')
        nn.fit(embed)
        dists, idxs = nn.kneighbors(embed, return_distance=True)
        # Use first non-self neighbor
        nn_idx = idxs[:, 1] if idxs.shape[1] > 1 else idxs[:, 0]
        pred = target_future[nn_idx]
        mse = float(np.mean((pred - target_future) ** 2))
        mse_map[k] = mse

    # Select k with minimal MSE (ties -> smallest k)
    k_selected = sorted(mse_map.items(), key=lambda kv: (kv[1], kv[0]))[0][0]
    return {'k_selected': int(k_selected), 'mse': mse_map}
