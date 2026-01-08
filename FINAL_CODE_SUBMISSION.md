## File: src/jidt_adapter.py
```python
"""
JIDT adapters for discrete TE/CTE and symbolic TE.

This module provides thin wrappers around JIDT calculators with explicit
contract-style docstrings and English-only documentation. It implements:
- Discrete TE with delay (tau) via 6-arg initialise
- True (conditional) TE using ConditionalTransferEntropyCalculatorDiscrete
- Symbolic TE via ordinal pattern encoding

@module jidt_adapter
"""
import logging
import gc
import numpy as np
import jpype
from jpype.types import JArray, JInt
from typing import Tuple, Optional
from src.params import TEParams, CTEParams, STEParams, CTEKraskovParams

logger = logging.getLogger(__name__)


def validate_series(series: np.ndarray, base: int, name: str) -> np.ndarray:
    """
    Validate a discrete time series and convert it to int32 within [0, base-1].

    @param {np.ndarray|Sequence[int]} series - Discrete series to validate.
    @param {int} base - Alphabet size; symbols must be in [0, base-1].
    @param {str} name - Human-readable series name for error messages.
    @returns {np.ndarray} int32 array with validated symbols.
    @throws {ValueError} If series is empty, contains NaN/Inf, or values out of range.
    @pre base >= 2 and name is non-empty.
    @post Returned array has dtype int32 and length > 0.
    """
    if not isinstance(series, np.ndarray):
        series = np.array(series)
    
    if len(series) == 0:
        raise ValueError(f"{name}: empty series")
    
    if np.any(np.isnan(series)) or np.any(np.isinf(series)):
        raise ValueError(f"{name}: contains NaN/Inf")
    
    series = series.astype(np.int32)
    
    if np.any(series < 0) or np.any(series >= base):
        raise ValueError(f"{name}: symbols must be in [0, {base-1}], got range [{series.min()}, {series.max()}]")
    
    return series


def java_array_int(py_array: np.ndarray) -> JArray:
    """
    Convert a numpy array to a Java int[] for JPype/JIDT interop.

    @param {np.ndarray|Sequence[int]} py_array - Source array.
    @returns {JArray} Java int[] with the same values.
    @pre Input is convertible to a 1-D int32 numpy array.
    @post Result length equals input length.
    """
    if not isinstance(py_array, np.ndarray):
        py_array = np.array(py_array, dtype=np.int32)
    if py_array.dtype != np.int32:
        py_array = py_array.astype(np.int32)
    return JArray(JInt)(py_array)


def _compute_cte_value_only(source: np.ndarray, dest: np.ndarray, cond: np.ndarray, params: CTEParams) -> float:
    """
    Compute CTE value once (no internal significance), with safe truncation.
    """
    # Validate inputs against their native bases
    source = validate_series(source, params.base_source, "source")
    dest = validate_series(dest, params.base_dest, "dest")
    cond = validate_series(cond, params.base_cond, "cond")

    if not (len(source) == len(dest) == len(cond)):
        raise ValueError(f"Length mismatch: source={len(source)}, dest={len(dest)}, cond={len(cond)}")

    # Apply data-level lag for tau>1
    tau = int(params.tau)
    if tau > 1:
        source = source[:-tau]
        dest = dest[tau:]
        cond = cond[tau:]

    # Safe truncation for block permutation compatibility
    k_hist = int(params.k_dest)
    block = 24
    n_full = int(len(source))
    n_eff = max(0, n_full - k_hist)
    num_safe_blocks = n_eff // block
    n_eff_safe = num_safe_blocks * block
    n_total_safe = n_eff_safe + k_hist
    if num_safe_blocks >= 1 and n_total_safe <= n_full:
        safe_source = source[:n_total_safe]
        safe_dest = dest[:n_total_safe]
        safe_cond = cond[:n_total_safe]
    else:
        safe_source, safe_dest, safe_cond = source, dest, cond

    # Initialise CTE calculator (discrete) and compute value
    CTECalc = jpype.JClass("infodynamics.measures.discrete.ConditionalTransferEntropyCalculatorDiscrete")
    calc = CTECalc()
    base_main = int(max(params.base_source, params.base_dest))
    calc.initialise(base_main, int(params.k_dest), 1, int(params.base_cond))

    j_source = java_array_int(safe_source)
    j_dest = java_array_int(safe_dest)
    j_cond = java_array_int(safe_cond)
    calc.addObservations(j_source, j_dest, j_cond)

    try:
        return float(calc.computeAverageLocalOfObservations())
    except Exception as e:
        logger.error(f"CTE value computation failed: {e}")
        return float('nan')


def _block_permute_source(source: np.ndarray, k_hist: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """
    Block-permute the source array after the first k_hist samples, with block size 'block'.
    Keeps the first k_hist elements in place to preserve initial history alignment.
    Assumes the caller has already truncated length so that (len(source)-k_hist) is a multiple of block.
    """
    n = len(source)
    if n <= k_hist:
        return source.copy()
    tail = source[k_hist:]
    # Split tail into contiguous blocks of equal size
    blocks = [tail[i:i+block] for i in range(0, len(tail), block)]
    order = rng.permutation(len(blocks))
    shuffled_tail = np.concatenate([blocks[i] for i in order]) if blocks else tail
    return np.concatenate([source[:k_hist], shuffled_tail])


class DiscreteTE:
    """
    Wrapper for JIDT TransferEntropyCalculatorDiscrete using 6-arg initialise.

    @param {TEParams} params - TE configuration including bases, histories, and delay (tau).
    @pre JPype JVM is started and JIDT classes are available on the classpath.
    @invariant self.calc is either None or a live JIDT calculator instance.
    """
    
    def __init__(self, params: TEParams):
        self.params = params
        self.calc = None
        
        # Use 0-arg constructor, then 6-arg initialise
        common_base = max(params.base_source, params.base_dest)
        TECalculator = jpype.JClass("infodynamics.measures.discrete.TransferEntropyCalculatorDiscrete")
        self.calc = TECalculator()
        
        # JIDT v1.5: initialise(base, k_dest, k_dest_tau, k_source, k_source_tau, delay)
        # k_tau=1 means use consecutive history, delay=tau for time lag
        self.calc.initialise(
            common_base,
            params.k_dest,
            1,  # k_dest_tau (consecutive)
            params.k_source,
            1,  # k_source_tau (consecutive)
            params.tau  # delay parameter
        )
        
        logger.debug(f"DiscreteTE initialized: base={common_base}, k_dest={params.k_dest}, k_src={params.k_source}, delay={params.tau}")
    
    def compute(self, source: np.ndarray, dest: np.ndarray) -> Tuple[float, float]:
        """
        Compute discrete TE and an associated p-value via surrogate testing.

        @param {np.ndarray} source - Source series with alphabet size params.base_source.
        @param {np.ndarray} dest - Destination series with alphabet size params.base_dest.
        @returns {[float, float]} A tuple (te_value, p_value); NaN on failure.
        @throws {ValueError} If validation fails (length mismatch, invalid symbols).
        @pre len(source) == len(dest) > 0.
        @post Returned values are finite or NaN when errors occur; internal GC invoked.
        """
        try:
            # Validate
            source = validate_series(source, self.params.base_source, "source")
            dest = validate_series(dest, self.params.base_dest, "dest")
            
            if len(source) != len(dest):
                raise ValueError(f"Length mismatch: source={len(source)}, dest={len(dest)}")
            
            # Add observations (JIDT uses simple addObservations without start/finalise)
            self.calc.addObservations(java_array_int(source), java_array_int(dest))
            
            # Compute
            te_value = self.calc.computeAverageLocalOfObservations()
            
            # Significance (adaptive or fixed)
            p_value = np.nan
            if self.params.adaptive_stages and len(self.params.adaptive_stages) > 0:
                last_p = np.nan
                for n_surr in self.params.adaptive_stages:
                    measure_dist = self.calc.computeSignificance(int(n_surr))
                    last_p = float(measure_dist.pValue)
                    # Early stop if decisively significant or non-significant
                    if self.params.early_stop_sig is not None and last_p <= self.params.early_stop_sig:
                        break
                    if self.params.early_stop_nonsig is not None and last_p >= self.params.early_stop_nonsig:
                        break
                p_value = last_p
            else:
                measure_dist = self.calc.computeSignificance(self.params.num_surrogates)
                p_value = measure_dist.pValue
            
            return (float(te_value) if np.isfinite(te_value) else np.nan,
                    float(p_value) if np.isfinite(p_value) else np.nan)
        except Exception as e:
            logger.error(f"DiscreteTE.compute failed: {e}")
            return (np.nan, np.nan)
        finally:
            gc.collect()
    
    def dispose(self):
        """
        Dispose the underlying calculator and trigger GC.

        @post self.calc is None.
        """
        self.calc = None
        gc.collect()


class StratifiedCTE:
    """
    Deprecated stratified-CTE (Fisher-merged) implementation.

    The method computes TE within hour-of-day bins and merges p-values using Fisher's
    method. It was found to be methodologically unreliable at k=4 during validation
    (opposite conclusions to True CTE) and is kept only for historical reference.

    @deprecated Use compute_true_cte() instead.
    """
    
    def __init__(self, params: CTEParams):
        self.params = params
    
    def compute(self, source: np.ndarray, dest: np.ndarray, cond: np.ndarray) -> Tuple[float, float]:
        """
        Compute stratified CTE and aggregate p-value via Fisher's method.

        @param {np.ndarray} source - Source series (A) with base params.base_source.
        @param {np.ndarray} dest - Destination series (S) with base params.base_dest.
        @param {np.ndarray} cond - Conditioning series (hour bin) with base params.base_cond.
        @returns {[float, float]} Weighted-average CTE and merged p-value; NaN on failure.
        @pre len(source) == len(dest) == len(cond) > 0; symbols within their bases.
        @post Returns NaN,NaN if no strata produced valid TE.
        @note For tau>1, applies data-level lag BEFORE stratification.
        """
        try:
            # Validate
            source = validate_series(source, self.params.base_source, "source")
            dest = validate_series(dest, self.params.base_dest, "dest")
            cond = validate_series(cond, self.params.base_cond, "cond")
            
            if not (len(source) == len(dest) == len(cond)):
                raise ValueError(f"Length mismatch: source={len(source)}, dest={len(dest)}, cond={len(cond)}")
            
            # Data-level lag for tau>1: shift source, align dest/cond BEFORE stratification
            if self.params.tau > 1:
                tau = self.params.tau
                source = source[:-tau]  # Drop last tau values from source
                dest = dest[tau:]       # Align dest (skip first tau)
                cond = cond[tau:]       # Align cond (skip first tau)
                logger.debug(f"StratifiedCTE: Applied data-level lag tau={tau}, N_after={len(source)}")
            
            # Stratify by conditioning variable
            unique_cond = np.unique(cond)
            te_values = []
            p_values = []
            weights = []
            
            for h in unique_cond:
                mask = (cond == h)
                n_h = mask.sum()
                
                # Compute TE for this stratum (bins already filtered upstream)
                source_h = source[mask]
                dest_h = dest[mask]
                
                # Use tau=1 for stratum TE since data is already globally lagged
                te_params = TEParams(
                    base_source=self.params.base_source,
                    base_dest=self.params.base_dest,
                    k_source=self.params.k_source,
                    k_dest=self.params.k_dest,
                    tau=1,  # Already lagged at global level
                    num_surrogates=self.params.num_surrogates,
                    adaptive_stages=self.params.adaptive_stages,
                    early_stop_sig=self.params.early_stop_sig,
                    early_stop_nonsig=self.params.early_stop_nonsig,
                    seed=self.params.seed
                )
                
                te_calc = DiscreteTE(te_params)
                te_h, p_h = te_calc.compute(source_h, dest_h)
                te_calc.dispose()
                
                if np.isfinite(te_h):
                    te_values.append(te_h)
                    p_values.append(p_h)
                    weights.append(n_h / len(source))
            
            if not te_values:
                return (np.nan, np.nan)
            
            # Weighted average CTE
            cte_value = np.average(te_values, weights=weights)
            
            # Aggregate p-value (Fisher's method)
            from scipy.stats import combine_pvalues
            if len(p_values) > 1:
                _, p_combined = combine_pvalues(p_values, method='fisher')
            else:
                p_combined = p_values[0] if p_values else np.nan
            
            return (float(cte_value) if np.isfinite(cte_value) else np.nan,
                    float(p_combined) if np.isfinite(p_combined) else np.nan)
            
        except Exception as e:
            logger.error(f"StratifiedCTE.compute failed: {e}")
            return (np.nan, np.nan)
        finally:
            gc.collect()


def compute_true_cte(source: np.ndarray, dest: np.ndarray, cond: np.ndarray, params: CTEParams) -> Tuple[float, float]:
    """
    Compute True Conditional Transfer Entropy using JIDT's discrete conditional TE.

    @param {np.ndarray} source - Source series (A) with base params.base_source.
    @param {np.ndarray} dest - Destination series (S) with base params.base_dest.
    @param {np.ndarray} cond - Conditioning series (hour bin) with base params.base_cond.
    @param {CTEParams} params - CTE configuration (k_dest, tau, surrogates, adaptive stages).
    @returns {[float, float]} A tuple (cte_value, p_value); NaN on failure.
    @throws {ValueError} If input validation fails or lengths mismatch.
    @pre len(source) == len(dest) == len(cond) > 0; symbols within their bases.
    @post Returns finite values or NaN on failure; arrays are garbage-collected.
    @note For tau>1, data-level lag is applied to align series before adding observations.
    @note JIDT ConditionalTransferEntropyCalculatorDiscrete initialise MUST use the 4-arg signature:
          initialise(int base, int history, int numOtherInfoContributors, int base_others).
          Here we set:
            - base = max(base_A, base_S)
            - history = k_S (destination history)
            - numOtherInfoContributors = 1
            - base_others = base_H (hour bins)
          JIDT API for this calculator does not expose separate setters for k_A (source history),
          hence k_A is effectively tied to k_S. Callers should ensure k_A == k_S for consistency.
    """
    try:
        # Manual non-causal offset per expert plan (tau = -1 semantics):
        # source_A_noncausal = source_A[2:]
        # dest_S_noncausal = dest_S[:-2]
        # condition_H_noncausal = condition_H[:-2]
        k_hist = int(params.k_dest)
        s_full = validate_series(source, params.base_source, "source")
        d_full = validate_series(dest, params.base_dest, "dest")
        c_full = validate_series(cond, params.base_cond, "cond")
        s_nc = s_full[k_hist:]
        d_nc = d_full[:-k_hist]
        c_nc = c_full[:-k_hist]

        # Compute TE_actual on non-causal arrays
        te_actual = _compute_cte_value_only(s_nc, d_nc, c_nc, params)

        # Prepare non-causal arrays for surrogate generation and safe truncation
        s = s_nc
        d = d_nc
        c = c_nc
        block = 24
        block = 24
        n_full = int(len(s))
        n_eff = max(0, n_full - k_hist)
        num_safe_blocks = n_eff // block
        n_eff_safe = num_safe_blocks * block
        n_total_safe = n_eff_safe + k_hist
        if num_safe_blocks >= 1 and n_total_safe <= n_full:
            safe_s = s[:n_total_safe]
            safe_d = d[:n_total_safe]
            safe_c = c[:n_total_safe]
        else:
            safe_s, safe_d, safe_c = s, d, c

        # Surrogate loop: block-permute source only
        # Fixed number of surrogates per expert plan
        M = 1000
        rng = np.random.default_rng(42)
        te_surrogates = np.empty(M, dtype=float)
        for i in range(M):
            shuf_s = _block_permute_source(safe_s, k_hist=k_hist, block=block, rng=rng)
            te_i = _compute_cte_value_only(shuf_s, safe_d, safe_c, params)
            te_surrogates[i] = te_i

        # Right-tailed p-value
        if np.isnan(te_actual):
            p_value = float('nan')
        else:
            ge = int(np.sum(te_surrogates >= te_actual))
            p_value = float((ge + 1) / (M + 1))

        return (te_actual if np.isfinite(te_actual) else np.nan,
                p_value if np.isfinite(p_value) else np.nan)
    except Exception as e:
        logger.error(f"TrueCTE.compute failed: {e}")
        return (np.nan, np.nan)
    finally:
        gc.collect()


def compute_true_cte_kraskov(source: np.ndarray, dest: np.ndarray, cond: np.ndarray, params: CTEKraskovParams) -> Tuple[float, float]:
    """
    Compute Conditional Transfer Entropy using JIDT Kraskov continuous estimator.

    This requires JIDT class: infodynamics.measures.continuous.kraskov.ConditionalTransferEntropyCalculatorKraskov

    @param source {np.ndarray} float64
    @param dest {np.ndarray} float64
    @param cond {np.ndarray} float64 (e.g., hour bin as numeric)
    @returns (cte_value, p_value)
    """
    try:
        # Validate numeric arrays
        s = np.asarray(source, dtype=float)
        d = np.asarray(dest, dtype=float)
        c = np.asarray(cond, dtype=float)
        if not (len(s) == len(d) == len(c)):
            raise ValueError(f"Length mismatch: source={len(s)}, dest={len(d)}, cond={len(c)}")

        # Data-level lag for tau>1
        tau = int(params.tau)
        if tau > 1:
            s = s[:-tau]
            d = d[tau:]
            c = c[tau:]

        CTEClass = jpype.JClass("infodynamics.measures.continuous.kraskov.ConditionalTransferEntropyCalculatorKraskov")
        calc = CTEClass()
        # Set neighbors if supported
        try:
            calc.setProperty("k", str(int(params.k_nn)))
        except Exception:
            pass

        # Initialise with histories and delay
        calc.initialise(int(params.k_dest), int(params.k_source), tau)

        # Add observations (continuous)
        # Some JIDT versions expect cond as 1-D double[]; others support multiple via 2D
        try:
            calc.addObservations(s.tolist(), d.tolist(), c.tolist())
        except Exception:
            # Fallback: pass cond as a 2D array with one column
            calc.addObservations(s.tolist(), d.tolist(), [c.tolist()])

        # Compute value
        cte_value = float('nan')
        try:
            cte_value = float(calc.computeAverageLocalOfObservations())
        except Exception as e:
            logger.error(f"Kraskov TrueCTE value failed: {e}")

        # Significance if available
        p_value = float('nan')
        try:
            md = calc.computeSignificance(int(params.num_surrogates))
            p_value = float(md.pValue)
        except Exception as e:
            logger.warning(f"Kraskov TrueCTE significance not available/failed: {e}")

        return (cte_value if np.isfinite(cte_value) else np.nan,
                p_value if np.isfinite(p_value) else np.nan)
    except Exception as e:
        logger.error(f"TrueCTE(Kraskov).compute failed: {e}")
        return (np.nan, np.nan)
    finally:
        gc.collect()

class SymbolicTE:
    """
    Wrapper for Symbolic Transfer Entropy using ordinal patterns + DiscreteTE.

    @param {STEParams} params - Symbolic TE configuration including ordinal dimension and delay.
    @pre Ordinal dimension >= 2 and sufficient series length for encoding.
    """
    
    def __init__(self, params: STEParams):
        self.params = params
    
    def ordinal_pattern_encode(self, series: np.ndarray) -> np.ndarray:
        """
        Encode a numeric series as ordinal patterns.

        @param {np.ndarray} series - Numeric time series.
        @returns {np.ndarray} Discrete sequence of ordinal pattern indices (int32).
        @throws {ValueError} If series is too short for the given ordinal parameters.
        @pre n_patterns = n - (dim-1)*delay >= 10.
        @post Returns length n_patterns array with values in [0, dim!-1].
        """
        from itertools import permutations
        
        n = len(series)
        n_patterns = n - (self.params.ordinal_dim - 1) * self.params.ordinal_delay
        
        if n_patterns < 10:
            raise ValueError(f"Series too short for ordinal encoding: N={n}")
        
        # Create permutation lookup
        all_perms = list(permutations(range(self.params.ordinal_dim)))
        perm_to_idx = {perm: i for i, perm in enumerate(all_perms)}
        
        patterns = np.zeros(n_patterns, dtype=np.int32)
        
        for i in range(n_patterns):
            indices = [i + j * self.params.ordinal_delay for j in range(self.params.ordinal_dim)]
            vec = series[indices]
            rank = tuple(np.argsort(np.argsort(vec)))
            patterns[i] = perm_to_idx.get(rank, 0)
        
        return patterns
    
    def compute(self, source: np.ndarray, dest: np.ndarray) -> Tuple[float, float]:
        """
        Compute Symbolic TE and p-value by encoding both series as ordinal patterns.

        @param {np.ndarray} source - Source numeric series.
        @param {np.ndarray} dest - Destination numeric series.
        @returns {[float, float]} (ste_value, p_value); NaN on failure or insufficient data.
        @pre After encoding, min(len(source_pat), len(dest_pat)) >= 100.
        @post Returns NaN if insufficient data; otherwise values from DiscreteTE.
        """
        try:
            import math
            
            # Convert to float for ordinal encoding
            source_f = source.astype(float)
            dest_f = dest.astype(float)
            
            # Encode as ordinal patterns
            patterns_source = self.ordinal_pattern_encode(source_f)
            patterns_dest = self.ordinal_pattern_encode(dest_f)
            
            # Alphabet size
            base = math.factorial(self.params.ordinal_dim)
            
            # Ensure sufficient data
            min_len = min(len(patterns_source), len(patterns_dest))
            if min_len < 100:
                logger.warning(f"Insufficient symbolic data: N={min_len}")
                return (np.nan, np.nan)
            
            # Truncate
            patterns_source = patterns_source[:min_len]
            patterns_dest = patterns_dest[:min_len]
            
            # Create TE params for symbolic data
            te_params = TEParams(
                base_source=base,
                base_dest=base,
                k_source=self.params.k_source,
                k_dest=self.params.k_dest,
                tau=self.params.tau,
                num_surrogates=self.params.num_surrogates,
                seed=self.params.seed
            )
            
            # Compute TE on symbolic sequences
            te_calc = DiscreteTE(te_params)
            ste_value, p_value = te_calc.compute(patterns_source, patterns_dest)
            te_calc.dispose()
            
            return (ste_value, p_value)
            
        except Exception as e:
            logger.error(f"SymbolicTE.compute failed: {e}")
            return (np.nan, np.nan)
        finally:
            gc.collect()
```

## File: src/preprocessing.py
```python
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
```

## File: src/run_lag_sweep.py
```python
"""
Lag sweep runner (N=60 users) for discrete Conditional TE with manual offsets.

For each user and tau in [-3,-2,-1,0,1,2,3]:
- Manually slice arrays to implement the lag (no DELAY property)
- Apply safe truncation so that (L - k_dest) is divisible by BLOCK_SIZE=24
- Compute TE(A->S|H) and TE(S->A|H) using _compute_cte_value_only
- Compute p-values via manual block-permutation surrogates (source only)

Outputs: per_user_lag_sweep_FINAL.csv in analysis/out/lag_sweep_<STAMP>/
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src import preprocessing
from src.params import CTEParams
from src.jidt_adapter import _compute_cte_value_only
from src.analysis import start_jvm, shutdown_jvm


BLOCK_SIZE = 24
K_DEST = 2  # target history (S)
K_SRC = 1   # source history (A) implicit in the calculator
TAUS = [-3, -2, -1, 0, 1, 2, 3]


def manual_offset_arrays(A: np.ndarray, S: np.ndarray, H: np.ndarray, tau: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply integer lag tau by manual slicing.
    Positive tau means S/H shift forward (A leads), negative tau means A leads.
    Returns aligned arrays of equal length.
    """
    if tau == 0:
        return A.copy(), S.copy(), H.copy()
    if tau > 0:
        # align A[:-tau] with S[tau:], H[tau:]
        return A[:-tau], S[tau:], H[tau:]
    else:
        t = -tau
        # align A[t:] with S[:-t], H[:-t]
        return A[t:], S[:-t], H[:-t]


def safe_truncate(A: np.ndarray, S: np.ndarray, H: np.ndarray, k_dest: int = K_DEST, block: int = BLOCK_SIZE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Truncate so that (L - k_dest) is a multiple of block.
    """
    L = int(len(A))
    if not (len(S) == L and len(H) == L):
        m = min(L, len(S), len(H))
        A, S, H = A[:m], S[:m], H[:m]
        L = m
    n_eff = L - k_dest
    if n_eff < block:
        # not enough data for one safe block; return as-is and let caller decide
        return A[:L], S[:L], H[:L]
    n_eff_safe = (n_eff // block) * block
    L_safe = n_eff_safe + k_dest
    return A[:L_safe], S[:L_safe], H[:L_safe]


def block_permute_source(src: np.ndarray, k_hist: int, block: int, rng: np.random.Generator) -> np.ndarray:
    if len(src) <= k_hist:
        return src.copy()
    head = src[:k_hist]
    tail = src[k_hist:]
    blocks = [tail[i:i+block] for i in range(0, len(tail), block)]
    order = rng.permutation(len(blocks))
    shuf_tail = np.concatenate([blocks[i] for i in order]) if blocks else tail
    return np.concatenate([head, shuf_tail])


def te_with_perm(A: np.ndarray, S: np.ndarray, H: np.ndarray, base_A: int, base_S: int, base_H: int, M: int = 1000) -> tuple[float, float]:
    params = CTEParams(
        base_source=base_A,
        base_dest=base_S,
        base_cond=base_H,
        k_source=K_SRC,
        k_dest=K_DEST,
        num_cond_bins=1,
        tau=1,
        num_surrogates=M,
    )
    te_actual = _compute_cte_value_only(A, S, H, params)
    rng = np.random.default_rng(42)
    te_sur = np.empty(M, dtype=float)
    for i in range(M):
        shuf_A = block_permute_source(A, k_hist=K_DEST, block=BLOCK_SIZE, rng=rng)
        te_sur[i] = _compute_cte_value_only(shuf_A, S, H, params)
    if np.isnan(te_actual):
        p = float('nan')
    else:
        p = float((1 + np.sum(te_sur >= te_actual)) / (M + 1))
    return te_actual, p


def load_user_ids_from_report() -> List[str]:
    # Prefer the stable 60-user list from report/data
    p = Path('report/data/per_user_true_cte.csv')
    if p.exists():
        df = pd.read_csv(p)
        df = df[df.get('tau', 1) == 1]
        ids = df.drop_duplicates(subset=['user_id'])['user_id'].tolist()
        return ids
    # Fallback: scan data directory
    data_root = Path('data/ExtraSensory.per_uuid_features_labels')
    ids = [f.name.split('.features_labels.csv')[0] for f in data_root.glob('*.features_labels.csv')]
    return ids[:60]


def main():
    ap = argparse.ArgumentParser(description='Run lag sweep across N=60 users (discrete CTE, manual offsets).')
    ap.add_argument('--outdir', default=None, help='Output directory (default: analysis/out/lag_sweep_<stamp>)')
    args = ap.parse_args()

    stamp = datetime.now().strftime('%Y%m%d_%H%M')
    out_dir = Path(args.outdir) if args.outdir else Path(f'analysis/out/lag_sweep_{stamp}')
    out_dir.mkdir(parents=True, exist_ok=True)

    user_ids = load_user_ids_from_report()
    if len(user_ids) != 60:
        print(f"WARN: user count = {len(user_ids)} (expected 60)")

    rows = []
    start_jvm()
    try:
        for uid in user_ids:
            try:
                df = preprocessing.load_subject_data(uid)
                A, S, H_raw, H = preprocessing.create_variables(df, feature_mode='composite', hour_bins=6, a_bins=5, s_mode='binary')
                A = A.astype(int); S = S.astype(int); H = H.astype(int)
                for tau in TAUS:
                    # Manual offset
                    A_tau, S_tau, H_tau = manual_offset_arrays(A, S, H, tau)
                    # Safe truncation using (L - k_dest)
                    A_safe, S_safe, H_safe = safe_truncate(A_tau, S_tau, H_tau, k_dest=K_DEST, block=BLOCK_SIZE)
                    # Base sizes
                    base_A = int(np.max(A_safe)) + 1 if len(A_safe) else 0
                    base_S = int(np.max(S_safe)) + 1 if len(S_safe) else 0
                    base_H = int(np.max(H_safe)) + 1 if len(H_safe) else 0
                    if min(len(A_safe), len(S_safe), len(H_safe)) < (K_DEST + BLOCK_SIZE):
                        rows.append({
                            'user_id': uid, 'tau': tau,
                            'TE_A_to_S': np.nan, 'p_A_to_S': np.nan,
                            'TE_S_to_A': np.nan, 'p_S_to_A': np.nan,
                        })
                        continue
                    # A->S
                    te_a2s, p_a2s = te_with_perm(A_safe, S_safe, H_safe, base_A, base_S, base_H)
                    # S->A
                    te_s2a, p_s2a = te_with_perm(S_safe, A_safe, H_safe, base_S, base_A, base_H)
                    rows.append({
                        'user_id': uid, 'tau': tau,
                        'TE_A_to_S': te_a2s, 'p_A_to_S': p_a2s,
                        'TE_S_to_A': te_s2a, 'p_S_to_A': p_s2a,
                    })
            except Exception as e:
                print(f"ERROR user {uid}: {e}")
                for tau in TAUS:
                    rows.append({'user_id': uid, 'tau': tau, 'TE_A_to_S': np.nan, 'p_A_to_S': np.nan, 'TE_S_to_A': np.nan, 'p_S_to_A': np.nan})
    finally:
        shutdown_jvm()

    out_csv = out_dir / 'per_user_lag_sweep_FINAL.csv'
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


if __name__ == '__main__':
    main()

```

## File: src/report_lag_sweep.py
```python
"""
Reporting for lag sweep: aggregate means per tau and plot curves.

Input: per_user_lag_sweep_FINAL.csv (from src/run_lag_sweep.py)
Output: fig_lag_sweep_curves.png in the same directory
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon, t as student_t


def main():
    ap = argparse.ArgumentParser(description='Report lag sweep results (means per tau and curves).')
    ap.add_argument('--csv', required=True, help='Path to per_user_lag_sweep_FINAL.csv')
    args = ap.parse_args()

    p = Path(args.csv)
    df = pd.read_csv(p)
    # Ensure numeric
    df['tau'] = pd.to_numeric(df['tau'], errors='coerce')
    df['TE_A_to_S'] = pd.to_numeric(df['TE_A_to_S'], errors='coerce')
    df['TE_S_to_A'] = pd.to_numeric(df['TE_S_to_A'], errors='coerce')

    # Aggregate means per tau
    means = df.groupby('tau', as_index=False).agg(
        mean_A2S=('TE_A_to_S', 'mean'),
        mean_S2A=('TE_S_to_A', 'mean'),
        n=('user_id', 'count'),
    ).sort_values('tau')

    # Print a text table
    print('Lag sweep means (bits):')
    for _, r in means.iterrows():
        print(f"tau={int(r['tau']):2d}  mean_A2S={r['mean_A2S']:.6f}  mean_S2A={r['mean_S2A']:.6f}  n={int(r['n'])}")

    # Plot curves
    out_png = p.parent / 'fig_lag_sweep_curves.png'
    plt.figure(figsize=(7, 4.2))
    plt.plot(means['tau'], means['mean_A2S'], '-o', color='#4C78A8', label='TE(A→S|H)')
    plt.plot(means['tau'], means['mean_S2A'], '-o', color='#E45756', label='TE(S→A|H)')
    plt.axhline(0.0, color='#777', ls=':', lw=1)
    plt.xlabel('tau (samples)')
    plt.ylabel('TE (bits)')
    plt.title('Lag sweep: mean TE vs tau (N=60)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved: {out_png}")

    # Paired Tests: per-user S→A at tau ∈ {-1, 0, 1}
    print("\nPaired Tests (per-user S→A at tau = 1 vs -1, and 1 vs 0):")
    per_user = (
        df[["user_id", "tau", "TE_S_to_A"]]
        .dropna(subset=["tau"])
        .copy()
    )
    per_user = per_user.groupby(["user_id", "tau"], as_index=False)["TE_S_to_A"].mean()
    pv = per_user.pivot_table(index="user_id", columns="tau", values="TE_S_to_A", aggfunc="mean")
    def paired_arrays(t1, t2):
        sub = pv[[t1, t2]].dropna()
        return sub[t1].to_numpy(), sub[t2].to_numpy()

    try:
        a1, a_1 = paired_arrays(1, -1)
        t_stat_1, p_val_1 = ttest_rel(a1, a_1)
        w_stat_1, w_p_1 = wilcoxon(a1, a_1)
    except Exception:
        t_stat_1 = p_val_1 = w_stat_1 = w_p_1 = float("nan")
    print(f"Test 1 (g(1) vs g(-1)) t={t_stat_1:.6f} p={p_val_1:.6g}")
    print(f"Test 1 (g(1) vs g(-1)) Wilcoxon W={w_stat_1:.6f} p={w_p_1:.6g}")

    try:
        a1, a0 = paired_arrays(1, 0)
        t_stat_2, p_val_2 = ttest_rel(a1, a0)
        w_stat_2, w_p_2 = wilcoxon(a1, a0)
    except Exception:
        t_stat_2 = p_val_2 = w_stat_2 = w_p_2 = float("nan")
    print(f"Test 2 (g(1) vs g(0))  t={t_stat_2:.6f} p={p_val_2:.6g}")
    print(f"Test 2 (g(1) vs g(0))  Wilcoxon W={w_stat_2:.6f} p={w_p_2:.6g}")
    # Bar chart of mean differences with 95% CI
    try:
        sub10 = pv[[1, 0]].dropna()
        sub1m1 = pv[[1, -1]].dropna()
        d2 = (sub10[1] - sub10[0]).to_numpy()
        d1 = (sub1m1[1] - sub1m1[-1]).to_numpy()
        def mean_ci(a):
            n = len(a)
            mu = float(a.mean()) if n>0 else float('nan')
            sd = float(a.std(ddof=1)) if n>1 else float('nan')
            if n > 1:
                tcrit = float(student_t.ppf(0.975, df=n-1))
                se = sd / (n ** 0.5)
                lo, hi = mu - tcrit * se, mu + tcrit * se
            else:
                lo = hi = float('nan')
            return mu, lo, hi
        mu1, lo1, hi1 = mean_ci(d1)
        mu2, lo2, hi2 = mean_ci(d2)
        def p_str(p):
            try:
                return 'p < 1e-7' if p < 1e-7 else f"p = {p:.2e}"
            except Exception:
                return 'p = nan'
        p1s = p_str(p_val_1)
        p2s = p_str(p_val_2)
        out_bar = p.parent / 'fig_paired_tests.png'
        labels = ['g(1)-g(-1)', 'g(1)-g(0)']
        means_bar = [mu1, mu2]
        ci_lows = [mu1 - lo1 if np.isfinite(lo1) else 0.0, mu2 - lo2 if np.isfinite(lo2) else 0.0]
        ci_highs = [hi1 - mu1 if np.isfinite(hi1) else 0.0, hi2 - mu2 if np.isfinite(hi2) else 0.0]
        x = np.arange(len(labels))
        plt.figure(figsize=(5.5, 4))
        plt.bar(x, means_bar, yerr=[ci_lows, ci_highs], capsize=6, color=['#4C78A8', '#E45756'], alpha=0.9)
        for i, (m, ps) in enumerate(zip(means_bar, [p1s, p2s])):
            y_annot = m + (ci_highs[i] if np.isfinite(ci_highs[i]) else 0) + 0.002
            plt.text(i, y_annot, ps, ha='center', va='bottom', fontsize=9)
        plt.xticks(x, labels)
        plt.axhline(0.0, color='#777', ls=':', lw=1)
        plt.ylabel('Mean difference in TE (bits)')
        plt.title('Paired differences with 95% CI')
        plt.tight_layout()
        plt.savefig(out_bar, dpi=300)
        plt.close()
        print(f"Saved: {out_bar}")
    except Exception as e:
        print(f"WARN: Could not generate paired tests plot: {e}")


if __name__ == '__main__':
    main()



```

## File: src/reporting.py
```python
"""
Reporting utilities for the final constrained discrete CTE plan.

Reads per_user_true_cte_discrete_k2l1_blockperm.csv and computes:
- Bonferroni-corrected significance counts (raw p * N_users)
- Robust group statistics over the 60 Delta_TE values (tau==1):
  * Mean, sample SD, and N
  * One-sample t-test vs 0
  * Wilcoxon signed-rank test (median vs 0)
  * Sign test summary (counts of negatives / positives / zeros)
- Saves a histogram of the 60 Delta_TE values as fig_delta_te_distribution.png
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ttest_1samp, wilcoxon
import matplotlib.pyplot as plt


def summarize_results(csv_path: str | Path) -> dict:
    p = Path(csv_path)
    df = pd.read_csv(p)
    # Expect columns: user_id, tau, k, l, hour_bins, CTE_true_A2S, CTE_true_S2A, Delta_TE, p_A2S, p_S2A
    # Filter to tau==1 and keep 1 row per user (N=60)
    df1 = df[df['tau'] == 1].copy()
    df1 = df1.drop_duplicates(subset=['user_id'])
    # Delta values
    delta = pd.to_numeric(df1['Delta_TE'], errors='coerce').dropna().to_numpy()
    n = int(delta.size)
    # Bonferroni: multiply p by N and test against alpha
    alpha = 0.05
    bonf = n if n > 0 else 1
    # Use tau==1 p-values if present; else fall back to zeros for counts
    p_a2s = pd.to_numeric(df1.get('p_A2S'), errors='coerce') if 'p_A2S' in df1.columns else pd.Series([np.nan]*n)
    p_s2a = pd.to_numeric(df1.get('p_S2A'), errors='coerce') if 'p_S2A' in df1.columns else pd.Series([np.nan]*n)
    sig_A2S = int(((p_a2s * bonf) < alpha).fillna(False).sum())
    sig_S2A = int(((p_s2a * bonf) < alpha).fillna(False).sum())

    mu = float(np.mean(delta)) if n else float('nan')
    sd = float(np.std(delta, ddof=1)) if n > 1 else float('nan')
    t_stat, p_val = ttest_1samp(delta, 0.0) if n >= 2 else (float('nan'), float('nan'))
    # Wilcoxon signed-rank (non-parametric)
    try:
        w_stat, w_p = wilcoxon(delta) if n >= 1 else (float('nan'), float('nan'))
    except ValueError:
        w_stat, w_p = (float('nan'), float('nan'))
    # Sign test summary
    neg = int((delta < 0).sum())
    pos = int((delta > 0).sum())
    zeros = int((delta == 0).sum())

    # Plot and save histogram next to the CSV
    fig_path = p.parent / 'fig_delta_te_distribution.png'
    try:
        plt.figure(figsize=(6, 4))
        plt.hist(delta, bins=12, color="#4C78A8", alpha=0.9, edgecolor='white')
        plt.axvline(0.0, color="#777", ls=":", lw=1)
        plt.axvline(mu, color="#A05195", ls="-", lw=1.5, label=f"mean={mu:.4f}")
        plt.xlabel("Delta TE (bits)")
        plt.ylabel("Count")
        plt.title(f"Delta TE distribution (N={n}, tau=1)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()
        fig_saved = True
    except Exception:
        fig_saved = False

    return {
        'N_users': n,
        'Bonferroni_factor': bonf,
        'A2S_sig_count': sig_A2S,
        'S2A_sig_count': sig_S2A,
        'Delta_mean': mu,
        'Delta_sd': sd,
        'ttest_t': float(t_stat) if np.isfinite(t_stat) else float('nan'),
        'ttest_p': float(p_val) if np.isfinite(p_val) else float('nan'),
        'wilcoxon_W': float(w_stat) if np.isfinite(w_stat) else float('nan'),
        'wilcoxon_p': float(w_p) if np.isfinite(w_p) else float('nan'),
        'sign_neg': neg,
        'sign_pos': pos,
        'sign_zero': zeros,
        'figure_path': str(fig_path),
        'figure_saved': fig_saved,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Summarize constrained discrete CTE results')
    ap.add_argument('--csv', default='analysis/out/latest/per_user_true_cte_discrete_k2l1_blockperm.csv', help='Path to per_user CSV')
    args = ap.parse_args()
    res = summarize_results(args.csv)
    print('N_users          :', res['N_users'])
    print('Bonferroni factor:', res['Bonferroni_factor'])
    print('A2S sig count    :', res['A2S_sig_count'])
    print('S2A sig count    :', res['S2A_sig_count'])
    print('Delta mean (bits):', f"{res['Delta_mean']:.6f}")
    print('Delta sd (bits)  :', f"{res['Delta_sd']:.6f}")
    print('t-test t         :', f"{res['ttest_t']:.4f}")
    print('t-test p         :', f"{res['ttest_p']:.6g}")
    print('Wilcoxon W       :', f"{res['wilcoxon_W']:.4f}")
    print('Wilcoxon p       :', f"{res['wilcoxon_p']:.6g}")
    print('Sign test counts :', f"neg={res['sign_neg']} pos={res['sign_pos']} zero={res['sign_zero']}")
    print('Histogram saved  :', res['figure_saved'], res['figure_path'])


if __name__ == '__main__':
    main()
```

## File: run_production.py
```python
#!/usr/bin/env python
"""Production-ready pipeline with tracking, heartbeat, and monitoring.

Features:
- run_info.yaml (JIDT version, JVM params, git commit, seed)
- k_selected_by_user.csv (AIS k-selection tracking)
- hbin_counts.csv (configurable hour bins, typically 6 or 24)
- status.json (continuous heartbeat with ETA)
- pipeline.log (structured logging with proper levels)
- CTE hour_bins from config, low_n_hours preserved
"""
import sys, json, logging, glob, gc, yaml, subprocess, time, traceback
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn

sys.path.insert(0, str(Path.cwd()))

from src import preprocessing, analysis, granger_analysis, symbolic_te
from src.fdr_utils import compute_delta_pvalue
from src.k_selection import select_k_via_ais, select_k_via_ragwitz
from src.quality_control import QualityController, DataQualityError

# Setup logging to both file and console
def setup_logging(out_dir):
    """
    Configure logging to both `pipeline.log` and console.

    @param {Path} out_dir - Output directory where `pipeline.log` is written.
    @returns {logging.Logger} Root logger configured with file and console handlers.
    @pre out_dir exists or is creatable by the caller.
    @post File handler captures DEBUG+; console handler captures INFO+.
    """
    log_file = out_dir / 'pipeline.log'
    
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    
    # File handler (capture DEBUG and above)
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Console handler (INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger

logger = logging.getLogger(__name__)


class ProductionPipeline:
    """
    Production pipeline with comprehensive tracking and monitoring.

    @param {str|Path} config_path - Path to YAML config (preset or custom).
    @param {str|Path|null} resume_dir - Existing output dir to resume from.
    @param {str|null} shard - Shard spec `i/N` for parallelized runs.
    @param {bool} no_progress - Disable progress bar output.
    @invariant self.out_dir is a directory; self.config validated.
    """
    
    def __init__(self, config_path, resume_dir=None, shard: str = None, no_progress: bool = False):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.no_progress = bool(no_progress)
        # Record shard info if provided (format: "i/N")
        self.shard = None
        self.shard_idx = None
        self.shard_total = None
        if shard:
            try:
                parts = str(shard).split('/')
                if len(parts) == 2:
                    self.shard_idx = int(parts[0])
                    self.shard_total = int(parts[1])
                    self.shard = shard
            except Exception:
                self.shard = None
        
        # Validate required config fields
        required_fields = ['hour_bins', 'taus', 'k_selection', 'surrogates', 'feature_modes', 'out_dir']
        missing = [f for f in required_fields if f not in self.config]
        if missing:
            raise ValueError(f"Missing required config fields: {missing}")
        
        # Validate hour_bins
        if not isinstance(self.config['hour_bins'], int) or self.config['hour_bins'] < 1:
            raise ValueError(f"hour_bins must be int >= 1, got {self.config['hour_bins']}")
        
        # Create or resume output directory
        if resume_dir:
            self.out_dir = Path(resume_dir)
            if not self.out_dir.exists():
                raise ValueError(f"Resume directory does not exist: {resume_dir}")
            self.is_resume = True
            logger.info(f"RESUME MODE: Continuing from {self.out_dir}")
        else:
            ts = datetime.now().strftime('%Y%m%d_%H%M')
            out_base = self.config['out_dir'].replace('<STAMP>', ts)
            # If running under a shard, add a shard-specific suffix for isolation
            if self.shard is not None:
                out_base = f"{out_base}_shard{self.shard_idx}of{self.shard_total}"
            self.out_dir = Path(out_base)
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.is_resume = False

        # Replace placeholders in JVM opts (e.g., {OUT_DIR})
        try:
            jvm_cfg = self.config.get('jvm', {})
            if isinstance(jvm_cfg.get('opts'), list):
                replaced = []
                for opt in jvm_cfg['opts']:
                    if isinstance(opt, str):
                        replaced.append(opt.replace('{OUT_DIR}', str(self.out_dir)))
                    else:
                        replaced.append(opt)
                self.config['jvm']['opts'] = replaced
        except Exception:
            pass
        
        # Setup logging
        setup_logging(self.out_dir)

        # Load quality control configuration
        self.quality = self._load_quality_control()

        # Diagnostics flags
        diag = self.config.get('run_diagnostics', {}) if isinstance(self.config, dict) else {}
        self.diag_k_only = bool(diag.get('k_selection_only', False))
        self.diag_qc_only = bool(diag.get('qc_stats_only', False))

        # Tracking
        self.results = {'te': [], 'cte': [], 'true_cte': [], 'ste': [], 'gc': [], 'k_selected': [], 'hbin_counts': []}
        self.errors = []
        self.start_time = None
        self.users_completed = 0
        self.total_users = 0
        self.completed_combinations = set()  # Track (user_id, feature_mode) combinations
        self.uuid_map = {}  # Map full UUID to short ID
        self.current_stage = None  # Track current processing stage
    
    def _load_quality_control(self):
        """
        Load quality control configuration from profile or inline config.

        @returns {QualityController} Initialized controller with thresholds.
        @pre config has `quality_profile` or inline `quality_control`.
        @post Returns a controller ready for validation and reporting.
        """
        quality_profile = self.config.get('quality_profile', 'balanced')
        
        # Try to load quality profile file
        profile_path = Path(f'config/quality/{quality_profile}.yaml')
        if profile_path.exists():
            with open(profile_path) as f:
                quality_config = yaml.safe_load(f)
            logger.info(f"QUALITY: Loaded profile '{quality_profile}' from {profile_path}")
        else:
            # Use inline quality_control config or defaults
            logger.warning(f"QUALITY: Profile '{quality_profile}' not found, using inline config")
            quality_config = self.config
        
        # Allow inline overrides
        if 'quality_control' in self.config:
            logger.info("QUALITY: Applying inline quality_control overrides")
            # Merge inline overrides (simple shallow merge)
            if 'quality_control' not in quality_config:
                quality_config['quality_control'] = {}
            quality_config['quality_control'].update(self.config['quality_control'])
        
        return QualityController(quality_config)
    
    def get_git_info(self):
        """
        Get git commit hash and short working tree status.

        @returns {dict} {'commit','dirty','status'} or fallbacks when git absent.
        """
        try:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
            status = subprocess.check_output(['git', 'status', '--short'], text=True).strip()
            return {
                'commit': commit[:8],
                'dirty': len(status) > 0,
                'status': status if len(status) > 0 else 'clean'
            }
        except:
            return {'commit': 'unknown', 'dirty': False, 'status': 'N/A'}
    
    def get_jidt_version(self):
        """
        Extract JIDT version metadata from the jar location.

        @returns {dict} {'jar','version','detected'} with best-effort detection.
        """
        jar_path = Path('jidt/infodynamics.jar')
        if jar_path.exists():
            return {'jar': str(jar_path), 'version': 'v1.5', 'detected': True}
        return {'jar': 'N/A', 'version': 'unknown', 'detected': False}
    
    def load_checkpoint(self):
        """
        Load existing results from checkpoint CSV files in the output dir.

        @returns {int} Count of user-feature combinations already completed.
        @post Internal result buffers and completed set are populated.
        """
        checkpoint_files = {
            'te': self.out_dir / 'per_user_te.csv',
            'cte': self.out_dir / 'per_user_cte.csv',
            'ste': self.out_dir / 'per_user_ste.csv',
            'gc': self.out_dir / 'per_user_gc.csv',
            'k_selected': self.out_dir / 'k_selected_by_user.csv',
            'hbin_counts': self.out_dir / 'hbin_counts.csv'
        }
        
        for key, fpath in checkpoint_files.items():
            if fpath.exists():
                df = pd.read_csv(fpath)
                logger.info(f"CHECKPOINT: Loaded {len(df)} rows from {fpath.name}")
                
                # Track completed (user_id, feature_mode) combinations
                if 'user_id' in df.columns and 'feature_mode' in df.columns:
                    for _, row in df.iterrows():
                        self.completed_combinations.add((row['user_id'], row['feature_mode']))
                elif 'user_id' in df.columns:
                    # For k_selected and hbin_counts (no feature_mode)
                    for _, row in df.iterrows():
                        # Mark all feature modes as having k selected
                        for mode in self.config.get('feature_modes', ['composite']):
                            self.completed_combinations.add((row['user_id'], mode))
        
        logger.info(f"CHECKPOINT: {len(self.completed_combinations)} combinations already completed")
        return len(self.completed_combinations)
    
    def save_checkpoint(self, method, data):
        """
        Save an incremental checkpoint after a method completes for a user.

        @param {str} method - One of 'te','cte','true_cte','ste','gc','k_selected','hbin_counts'.
        @param {dict|list[dict]} data - Row or rows to append/save.
        @returns {None}
        @post Corresponding CSV exists and contains the new rows.
        """
        file_map = {
            'te': 'per_user_te.csv',
            'cte': 'per_user_cte.csv',
            'ste': 'per_user_ste.csv',
            'gc': 'per_user_gc.csv',
            'k_selected': 'k_selected_by_user.csv',
            'hbin_counts': 'hbin_counts.csv'
        }
        
        if method not in file_map:
            return
        
        fpath = self.out_dir / file_map[method]
        df_new = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
        
        # Append to existing file or create new
        if fpath.exists():
            df_new.to_csv(fpath, mode='a', header=False, index=False)
        else:
            df_new.to_csv(fpath, index=False)
    
    def write_run_info(self, seed=None):
        """
        Write `run_info.yaml` with environment details and configuration.

        @param {int|null} seed - Optional RNG seed to record for reproducibility.
        @returns {None}
        @post `run_info.yaml` exists in `self.out_dir` with metadata.
        """
        git_info = self.get_git_info()
        jidt_info = self.get_jidt_version()
        
        jvm_opts = self.config.get('jvm', {})
        
        run_info = {
            'timestamp': datetime.now().isoformat(),
            'schema_version': self.config.get('schema_version', 'v1.0'),
            'git': git_info,
            'jidt': jidt_info,
            'jvm': {
                'heap_min': jvm_opts.get('xms', '32g'),
                'heap_max': jvm_opts.get('xmx', '48g'),
                'gc': jvm_opts.get('opts', []),
                'classpath': 'jidt/infodynamics.jar'
            },
            'random_seed': seed if seed else 'system_default',
            'config': self.config,
            'implementation': {
                'TE': 'JIDT-v1.5-6arg-initialise(base,k_dest,1,k_src,1,delay)',
                'CTE': 'STRATIFIED-TE-with-data-level-lag-before-stratification',
                'STE': 'JIDT-v1.5-6arg-initialise',
                'GC': 'statsmodels-VAR-AIC',
                'FDR': 'BH-per-family-per-tau'
            }
        }
        
        with open(self.out_dir / 'run_info.yaml', 'w') as f:
            yaml.dump(run_info, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"run_info.yaml written: git={git_info['commit']}, JIDT={jidt_info['version']}")
    
    def update_heartbeat(self, current_user=None, current_stage=None):
        """
        Update `status.json` with current progress, ETA, and latest task info.

        @param {str|null} current_user - Current user ID being processed.
        @param {str|null} current_stage - Human-readable stage description.
        @returns {None}
        @post `status.json` reflects current pipeline progress.
        """
        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        if self.users_completed > 0:
            avg_time_per_user = elapsed / self.users_completed
            remaining_users = self.total_users - self.users_completed
            eta_seconds = avg_time_per_user * remaining_users
            eta_time = datetime.now() + timedelta(seconds=eta_seconds)
            eta_str = eta_time.strftime('%Y-%m-%d %H:%M:%S')
            throughput = (self.users_completed / elapsed) * 3600 if elapsed > 0 else 0
        else:
            eta_str = 'calculating...'
            avg_time_per_user = 0
            throughput = 0
        
        # Shorten current_user UUID
        short_user = None
        if current_user:
            parts = current_user.split('/')
            short_uuid = parts[0][:8] if len(parts[0]) > 8 else parts[0]
            short_user = f"{short_uuid}/{parts[1]}" if len(parts) > 1 else short_uuid
        
        status = {
            'status': 'running',
            'timestamp': datetime.now().isoformat(),
            'pipeline': {
                'total_users': self.total_users,
                'completed_users': self.users_completed,
                'failed_users': len([e for e in self.errors if e.get('method') == 'PROCESS']),
                'progress_percent': round(100 * self.users_completed / self.total_users, 1) if self.total_users > 0 else 0
            },
            'current_task': {
                'user_id': short_user,
                'stage': current_stage or 'processing'
            } if current_user else None,
            'performance': {
                'elapsed_seconds': int(elapsed),
                'elapsed_formatted': str(timedelta(seconds=int(elapsed))),
                'eta_seconds': int(eta_seconds) if self.users_completed > 0 else None,
                'eta_time': eta_str,
                'throughput_users_per_hour': round(throughput, 2),
                'avg_time_per_user': round(avg_time_per_user, 2)
            },
            'errors': {
                'count': len(self.errors),
                'last_error': self.errors[-1] if self.errors else None
            }
        }
        
        with open(self.out_dir / 'status.json', 'w') as f:
            json.dump(status, f, indent=2)
    
    def compute_hbin_counts(self, H_raw, user_id, feature_mode):
        """
        Compute and record hour-bin sample counts for diagnostics/reporting.

        @param {np.ndarray} H_raw - Hour-of-day values (0..23).
        @param {str} user_id - User UUID string.
        @param {str} feature_mode - Feature engineering mode.
        @returns {dict|None} Summary row appended to `self.results['hbin_counts']` or None when empty.
        @post Adds a record to hbin counts; returns that record.
        """
        if len(H_raw) == 0:
            return
        
        # Count samples per hour (0-23)
        bin_counts = {}
        for h in range(24):
            bin_counts[f'hour_{h:02d}'] = int((H_raw == h).sum())
        
        hbin_result = {
            'user_id': user_id,
            'feature_mode': feature_mode,
            'total_samples': len(H_raw),
            **bin_counts
        }
        self.results['hbin_counts'].append(hbin_result)
        self.save_checkpoint('hbin_counts', hbin_result)
    
    def select_k(self, S, base_S, user_id, H_raw=None):
        """
        Select k via AIS or use a fixed k from config.

        AIS strategy supports optional constraints:
        - k_max: Hard cap on k (e.g., 4 for computational feasibility)
        - undersampling_guard: Prevent undersampled states (min 25 samples/state)

        @param {np.ndarray} S - Destination series for AIS-based k-selection.
        @param {int} base_S - Alphabet size for S.
        @param {str} user_id - User identifier for logging.
        @param {np.ndarray|null} H_raw - Optional hour-of-day raw values.
        @returns {int} Selected k value.
        @post Logs selection details; returns a valid k >= 1.
        """
        k_strategy = self.config.get('k_selection', {}).get('strategy', 'fixed')
        
        if k_strategy == 'AIS':
            try:
                k_config = self.config['k_selection']
                k_grid = k_config['k_grid']
                k_max = k_config.get('k_max')  # Can be None for no limit
                undersampling_guard = k_config.get('undersampling_guard', False)
                
                min_samples = None
                if undersampling_guard and H_raw is not None:
                    min_samples = min([(H_raw == h).sum() for h in range(24)])
                
                k_info = select_k_via_ais(
                    S.astype(int), base_S, k_grid,
                    num_surrogates=100, criterion='max_ais',
                    k_max=k_max, min_samples=min_samples
                )
                k_selected = k_info['k_selected']
                
                k_result = {
                    'user_id': user_id,
                    'k_selected': k_selected,
                    'k_original': k_info.get('k_original', k_selected),
                    'capped': k_info.get('capped', False),
                    'ais_values': json.dumps(k_info['ais_values']),
                    'criterion': k_info['criterion']
                }
                self.results['k_selected'].append(k_result)
                self.save_checkpoint('k_selected', k_result)
                
                cap_msg = f" (capped from {k_info['k_original']})" if k_info.get('capped') else ""
                short_id = self.uuid_map.get(user_id, user_id[:8])
                logger.info(f"{short_id} | K_SELECT | k={k_selected}{cap_msg}")
                return k_selected
            except Exception as e:
                short_id = self.uuid_map.get(user_id, user_id[:8])
                logger.error(f"{short_id} | K_SELECT | FAIL | {e}")
                self.errors.append({
                    'user_id': user_id, 
                    'method': 'K_SELECTION', 
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'timestamp': datetime.now().isoformat()
                })
                return 4  # Fallback
        elif k_strategy == 'RAGWITZ':
            try:
                k_config = self.config['k_selection']
                k_grid = k_config.get('k_grid', [1,2,3,4,5,6])
                tau = int(self.config.get('taus', [1])[0])
                info = select_k_via_ragwitz(S.astype(float), k_grid, tau=tau)
                k_selected = info['k_selected']
                k_result = {
                    'user_id': user_id,
                    'k_selected': int(k_selected),
                    'criterion': 'ragwitz',
                    'mse': json.dumps(info.get('mse', {}))
                }
                self.results['k_selected'].append(k_result)
                self.save_checkpoint('k_selected', k_result)
                short_id = self.uuid_map.get(user_id, user_id[:8])
                logger.info(f"{short_id} | K_SELECT | k={k_selected} (ragwitz)")
                return int(k_selected)
            except Exception as e:
                short_id = self.uuid_map.get(user_id, user_id[:8])
                logger.error(f"{short_id} | K_SELECT(RAGWITZ) | FAIL | {e}")
                self.errors.append({
                    'user_id': user_id,
                    'method': 'K_SELECTION_RAGWITZ',
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'timestamp': datetime.now().isoformat()
                })
                return 2
        else:
            # Fixed k from config
            return self.config.get('k_selection', {}).get('k_fixed', 4)
    
    def process_user(self, user_id, feature_mode):
        """
        Process a single (user, feature_mode) combination end-to-end.

        @param {str} user_id - Full user UUID.
        @param {str} feature_mode - Feature engineering mode name.
        @returns {None}
        @post Results and errors are appended to their respective buffers.
        """
        # Create short UUID mapping
        short_id = user_id[:8]
        self.uuid_map[user_id] = short_id
        
        # Check if already completed (resume mode)
        if (user_id, feature_mode) in self.completed_combinations:
            logger.info(f"{short_id} | SKIP | mode={feature_mode} (already completed)")
            return
        
        try:
            start_time = datetime.now()
            logger.info(f"{short_id} | START | mode={feature_mode}")
            
            # Load data
            self.current_stage = 'loading'
            self.update_heartbeat(current_user=f"{user_id}/{feature_mode}", current_stage='loading')
            raw = preprocessing.load_subject_data(user_id)
            hour_bins = self.config['hour_bins']
            A, S, H_raw, H_binned = preprocessing.create_variables(raw, feature_mode=feature_mode, hour_bins=hour_bins)
            logger.info(f"{short_id} | LOADED | mode={feature_mode} samples={len(A)}")
            
            if len(A) == 0:
                logger.warning(f"{short_id} | EMPTY_DATA | mode={feature_mode}")
                self.errors.append({
                    'user_id': user_id, 
                    'feature_mode': feature_mode, 
                    'error': 'empty_data',
                    'timestamp': datetime.now().isoformat()
                })
                return
            
            # Global quality validation
            try:
                self.quality.validate_global(len(A), short_id)
            except DataQualityError as e:
                logger.warning(f"{short_id} | SKIP | {e}")
                self.errors.append({
                    'user_id': user_id,
                    'feature_mode': feature_mode,
                    'error': f'quality_global: {e}',
                    'timestamp': datetime.now().isoformat()
                })
                return
            
            # For discrete path, compute bases; base_A from A (0..4), base_S from S (0..1)
            base_A, base_S = int(np.max(A)) + 1, int(np.max(S)) + 1
            base_H = int(np.max(H_binned)) + 1
            
            # Select k (only once per user, reuse for all modes)
            self.current_stage = 'k_selection'
            self.update_heartbeat(current_user=f"{user_id}/{feature_mode}", current_stage='k_selection')
            # Fixed embeddings per final plan: k (dest=S history)=2, l (source=A history)=1
            k, l = 2, 1
            logger.info(f"{short_id} | K_SELECT | k={k}")

            # Fast diagnostic mode: write k-selection and/or QC stats only, skip TE/CTE
            if self.diag_k_only or self.diag_qc_only:
                merged_dir = Path('analysis/out/production_k6_true_cte_merged')
                merged_dir.mkdir(parents=True, exist_ok=True)
                if self.diag_k_only:
                    out_k = merged_dir / 'k_selected_by_user_ALL.csv'
                    df = pd.DataFrame([{
                        'user_id': user_id,
                        'feature_mode': feature_mode,
                        'k_A': int(k),
                        'k_S': int(l)
                    }])
                    if out_k.exists():
                        df.to_csv(out_k, mode='a', index=False, header=False)
                    else:
                        df.to_csv(out_k, index=False)
                if self.diag_qc_only:
                    # Compute per-bin counts on H_binned
                    n_bins_total = int(self.config.get('hour_bins', 6))
                    counts = [(H_binned == b).sum() for b in range(n_bins_total)]
                    n_bins_filtered = int(np.sum(np.array(counts) < self.quality.cte.min_bin_samples))
                    out_qc = merged_dir / 'qc_stats_ALL.csv'
                    dfq = pd.DataFrame([{
                        'user_id': user_id,
                        'n_total_samples': int(len(A)),
                        'n_bins_total': n_bins_total,
                        'n_bins_filtered': n_bins_filtered
                    }])
                    if out_qc.exists():
                        dfq.to_csv(out_qc, mode='a', index=False, header=False)
                    else:
                        dfq.to_csv(out_qc, index=False)
                # Done for this user in diagnostic mode
                return
            
            # Compute hour bin counts (using H_binned)
            self.compute_hbin_counts(H_binned, user_id, feature_mode)
            
            # Validate and filter CTE hour bins
            num_bins = base_H
            hour_counts = [(H_binned == h).sum() for h in range(num_bins)]
            cte_passed, valid_bins, cte_diagnostics = self.quality.validate_cte(len(A), hour_counts, short_id)
            low_n_hours = cte_diagnostics['low_bins']
            
            # Resolve analysis modes (default: run all existing modes for backward compatibility)
            analysis_modes = self.config.get('analysis_modes', ['global_te', 'stratified_te', 'ste', 'gc'])

            # TE
            num_surrogates = self.config.get('surrogates', 1000)
            # Adaptive surrogate testing configuration
            stat_cfg = self.config.get('statistical', {})
            adapt_cfg = stat_cfg.get('adaptive_surrogates', {}) if isinstance(stat_cfg, dict) else {}
            adaptive_enabled = bool(adapt_cfg.get('enabled', False))
            adaptive_stages = adapt_cfg.get('stages', [1000, 3000, 10000]) if adaptive_enabled else None
            early_stop_sig = adapt_cfg.get('p_sig', 0.01) if adaptive_enabled else None
            early_stop_nonsig = adapt_cfg.get('p_nonsig', 0.20) if adaptive_enabled else None
            if 'global_te' in analysis_modes and not (self.diag_k_only or self.diag_qc_only):
                for idx, tau in enumerate(self.config['taus'], 1):
                    # Validate TE requirements
                    try:
                        te_valid = self.quality.validate_te(len(A), short_id, base_A, base_S, k)
                    except DataQualityError as e:
                        logger.info(f"{short_id} | TE | tau={tau} SKIP | {e}")
                        continue
                    
                    # NOTE: Intentional OOM handling for Global TE at high k
                    # ------------------------------------------------------
                    # Empirically, when k>=5 (especially k=6, which 44/60 users select),
                    # Global TE frequently OOMs with an 8–12GB JVM heap due to state space explosion.
                    # Diagnostics (see analysis/out/production_k6_true_cte_merged/k_selected_by_user_ALL.csv)
                    # show 73% of users select k=6 with AIS. We therefore accept Global TE failures at
                    # high k and record the TE value as NaN by design:
                    #   1) This is intentional and acceptable (overall OOM rate ≈73% matches k=6 selection rate).
                    #   2) The core method (True CTE) successfully runs at k=6 within ~8–12GB, so conclusions are
                    #      based on True CTE and are not impacted by Global TE OOM.
                    # The try/except block below implements this policy: detect OOM at high k, record NaN, continue.
                    try:
                        self.current_stage = f'TE ({idx}/{len(self.config["taus"])})'
                        self.update_heartbeat(current_user=f"{user_id}/{feature_mode}", current_stage=self.current_stage)
                        
                        te_start = datetime.now()
                        logger.info(f"{short_id} | TE | tau={tau} start")
                        te = analysis.run_te_analysis(A.astype(int), S.astype(int), k, l, base_A, base_S, tau=tau, num_surrogates=num_surrogates,
                                                      adaptive_stages=adaptive_stages, early_stop_sig=early_stop_sig, early_stop_nonsig=early_stop_nonsig)
                        te_elapsed = (datetime.now() - te_start).total_seconds()
                        logger.info(f"{short_id} | TE | tau={tau} done elapsed={te_elapsed:.2f}s")
                        
                        te_result = {
                            'user_id': user_id, 'feature_mode': feature_mode, 'k': k, 'l': l, 'tau': tau,
                            'TE_A2S': te.get('TE(A->S)'), 'TE_S2A': te.get('TE(S->A)'),
                            'Delta_TE': te.get('Delta_TE'), 'p_A2S': te.get('p(A->S)'),
                            'p_S2A': te.get('p(S->A)', np.nan), 'n_samples': len(A), 
                            'low_n': len(A) < self.quality.te.min_samples,
                            'quality_passed': te_valid
                        }
                        self.results['te'].append(te_result)
                        
                        # Save checkpoint after each TE completion
                        self.save_checkpoint('te', te_result)
                        gc.collect()
                    except Exception as e:
                        msg = str(e)
                        is_oom = ('Requested memory' in msg and 'too large for the JVM' in msg)
                        if is_oom and (k in (5, 6)):
                            # Graceful failure for high-k OOM: record NaNs and continue
                            logger.warning(f"{short_id} | TE | tau={tau} WARN | Global TE failed at k={k} due to OOM; recording NaN and continuing")
                            te_result = {
                                'user_id': user_id, 'feature_mode': feature_mode, 'k': k, 'l': l, 'tau': tau,
                                'TE_A2S': np.nan, 'TE_S2A': np.nan, 'Delta_TE': np.nan,
                                'p_A2S': np.nan, 'p_S2A': np.nan,
                                'n_samples': len(A), 'low_n': len(A) < self.quality.te.min_samples,
                                'quality_passed': te_valid
                            }
                            self.results['te'].append(te_result)
                            self.save_checkpoint('te', te_result)
                            gc.collect()
                        else:
                            logger.error(f"{short_id} | TE | tau={tau} FAIL | {e}")
                            self.errors.append({
                                'user_id': user_id, 
                                'feature_mode': feature_mode, 
                                'method': 'TE', 
                                'tau': tau, 
                                'error': str(e),
                                'traceback': traceback.format_exc(),
                                'timestamp': datetime.now().isoformat()
                            })
            
            # CTE (Stratified)
            if 'stratified_te' in analysis_modes and not (self.diag_k_only or self.diag_qc_only):
                if not cte_passed:
                    logger.warning(f"{short_id} | CTE | SKIP | quality check failed")
                else:
                    # Filter samples to valid bins determined by quality control
                    if valid_bins:
                        mask_valid = np.isin(H_binned, valid_bins)
                    else:
                        # No valid bins (should not happen if cte_passed), but guard
                        mask_valid = np.zeros_like(H_binned, dtype=bool)

                    A_cte = A[mask_valid]
                    S_cte = S[mask_valid]
                    H_cte = H_binned[mask_valid]

                    for idx, tau in enumerate(self.config['taus'], 1):
                        try:
                            self.current_stage = f'CTE ({idx}/{len(self.config["taus"])})'
                            self.update_heartbeat(current_user=f"{user_id}/{feature_mode}", current_stage=self.current_stage)
                            
                            cte_start = datetime.now()
                            logger.info(f"{short_id} | CTE | tau={tau} start bins={len(valid_bins)}/{len(hour_counts)}")
                            cte = analysis.run_cte_analysis(
                                A_cte.astype(int),
                                S_cte.astype(int),
                                H_cte,
                                k,
                                l,
                                base_A,
                                base_S,
                                self.config['hour_bins'],
                                tau=tau,
                                num_surrogates=num_surrogates,
                                adaptive_stages=adaptive_stages,
                                early_stop_sig=early_stop_sig,
                                early_stop_nonsig=early_stop_nonsig
                            )
                            cte_elapsed = (datetime.now() - cte_start).total_seconds()
                            logger.info(f"{short_id} | CTE | tau={tau} done elapsed={cte_elapsed:.2f}s")
                            
                            cte_result = {
                                'user_id': user_id, 'feature_mode': feature_mode, 'k': k, 'l': l, 'tau': tau,
                                'hour_bins': self.config['hour_bins'],
                                'CTE_A2S': cte.get('CTE(A->S|H_bin)'), 'CTE_S2A': cte.get('CTE(S->A|H_bin)'),
                                'Delta_CTE': cte.get('Delta_CTE_bin'), 'p_A2S': cte.get('p_cte(A->S|H_bin)'),
                                'p_S2A': cte.get('p_cte(S->A|H_bin)', np.nan), 'n_samples': len(A),
                                'n_samples_per_bin_min': min(hour_counts) if hour_counts else np.nan,
                                'low_n': len(A) < self.quality.cte.min_total_samples,
                                'low_n_hours': json.dumps(low_n_hours),
                                'bins_filtered': len(cte_diagnostics['low_bins']),
                                'quality_passed': cte_passed
                            }
                            self.results['cte'].append(cte_result)
                            
                            # Save checkpoint after each CTE completion
                            self.save_checkpoint('cte', cte_result)
                            gc.collect()
                        except Exception as e:
                            logger.error(f"{short_id} | CTE | tau={tau} FAIL | {e}")
                            self.errors.append({
                                'user_id': user_id, 
                                'feature_mode': feature_mode, 
                                'method': 'CTE', 
                                'tau': tau, 
                                'error': str(e),
                                'traceback': traceback.format_exc(),
                                'timestamp': datetime.now().isoformat()
                            })

            # True CTE (discrete, fixed k=2, l=1, tau=1)
            if 'true_cte' in analysis_modes and not (self.diag_k_only or self.diag_qc_only):
                if not cte_passed:
                    logger.warning(f"{short_id} | TRUE_CTE | SKIP | quality check failed")
                else:
                    # Reuse the same filtered arrays for comparability
                    if valid_bins:
                        mask_valid = np.isin(H_binned, valid_bins)
                    else:
                        mask_valid = np.zeros_like(H_binned, dtype=bool)
                    A_true = A[mask_valid]
                    S_true = S[mask_valid]
                    H_true = H_binned[mask_valid]

                    for idx, tau in enumerate(self.config['taus'], 1):
                        try:
                            self.current_stage = f'TRUE_CTE ({idx}/{len(self.config["taus"])})'
                            self.update_heartbeat(current_user=f"{user_id}/{feature_mode}", current_stage=self.current_stage)
                            
                            t_start = datetime.now()
                            logger.info(f"{short_id} | TRUE_CTE | tau={tau} start bins={len(valid_bins)}/{len(hour_counts)}")
                            cte_true = analysis.run_true_cte_analysis(
                                A_true.astype(int),
                                S_true.astype(int),
                                H_true.astype(int),
                                k,
                                l,
                                base_A,
                                base_S,
                                base_H,
                                tau=tau,
                                num_surrogates=num_surrogates,
                                adaptive_stages=adaptive_stages,
                                early_stop_sig=early_stop_sig,
                                early_stop_nonsig=early_stop_nonsig
                            )
                            t_elapsed = (datetime.now() - t_start).total_seconds()
                            logger.info(f"{short_id} | TRUE_CTE | tau={tau} done elapsed={t_elapsed:.2f}s")

                            res = {
                                'user_id': user_id, 'feature_mode': feature_mode, 'k': k, 'l': l, 'tau': tau,
                                'hour_bins': self.config['hour_bins'],
                                'CTE_true_A2S': cte_true.get('CTE_true(A->S|H)'), 'CTE_true_S2A': cte_true.get('CTE_true(S->A|H)'),
                                'Delta_CTE_true': cte_true.get('Delta_CTE_true'), 'p_A2S': cte_true.get('p_true_cte(A->S|H)'),
                                'p_S2A': cte_true.get('p_true_cte(S->A|H)', np.nan), 'n_samples': len(A),
                                'n_samples_per_bin_min': min(hour_counts) if hour_counts else np.nan,
                                'low_n': len(A) < self.quality.cte.min_total_samples,
                                'low_n_hours': json.dumps(low_n_hours),
                                'bins_filtered': len(cte_diagnostics['low_bins']),
                                'quality_passed': cte_passed
                            }
                            self.results['true_cte'].append(res)
                            self.save_checkpoint('true_cte', res)
                            gc.collect()
                        except Exception as e:
                            logger.error(f"{short_id} | TRUE_CTE | tau={tau} FAIL | {e}")
                            self.errors.append({
                                'user_id': user_id, 
                                'feature_mode': feature_mode, 
                                'method': 'TRUE_CTE', 
                                'tau': tau, 
                                'error': str(e),
                                'traceback': traceback.format_exc(),
                                'timestamp': datetime.now().isoformat()
                            })
            
            # STE
            if 'ste' in analysis_modes and not (self.diag_k_only or self.diag_qc_only):
                for idx, tau in enumerate(self.config['taus'], 1):
                    # Validate STE requirements
                    try:
                        ste_valid = self.quality.validate_ste(len(A), short_id)
                    except DataQualityError as e:
                        logger.info(f"{short_id} | STE | tau={tau} SKIP | {e}")
                        continue
                    
                    try:
                        self.current_stage = f'STE ({idx}/{len(self.config["taus"])})'
                        self.update_heartbeat(current_user=f"{user_id}/{feature_mode}", current_stage=self.current_stage)
                        
                        ste_start = datetime.now()
                        logger.info(f"{short_id} | STE | tau={tau} start")
                        ste = symbolic_te.run_symbolic_te_analysis(A, S, k, k, tau=tau, num_surrogates=num_surrogates)
                        ste_elapsed = (datetime.now() - ste_start).total_seconds()
                        logger.info(f"{short_id} | STE | tau={tau} done elapsed={ste_elapsed:.2f}s")
                        
                        ste_result = {
                            'user_id': user_id, 'feature_mode': feature_mode, 'k': k, 'tau': tau,
                            'STE_A2S': ste.get('STE(A->S)'), 'STE_S2A': ste.get('STE(S->A)'),
                            'Delta_STE': ste.get('Delta_STE'), 'p_A2S': ste.get('p_ste(A->S)'),
                            'p_S2A': ste.get('p_ste(S->A)', np.nan),
                            'n_samples': len(A),
                            'low_n': len(A) < self.quality.ste.min_samples,
                            'quality_passed': ste_valid
                        }
                        self.results['ste'].append(ste_result)
                        
                        # Save checkpoint after each STE completion
                        self.save_checkpoint('ste', ste_result)
                        gc.collect()
                    except Exception as e:
                        logger.error(f"{short_id} | STE | tau={tau} FAIL | {e}")
                        self.errors.append({
                            'user_id': user_id, 
                            'feature_mode': feature_mode, 
                            'method': 'STE', 
                            'tau': tau, 
                            'error': str(e),
                            'traceback': traceback.format_exc(),
                            'timestamp': datetime.now().isoformat()
                        })
            
            # GC
            if 'gc' in analysis_modes and not (self.diag_k_only or self.diag_qc_only):
                max_lag = 8
                try:
                    gc_valid = self.quality.validate_gc(len(A), short_id, max_lag)
                except DataQualityError as e:
                    logger.info(f"{short_id} | GC | SKIP | {e}")
                    gc_valid = False
                
                if gc_valid:
                    try:
                        self.current_stage = 'GC'
                        self.update_heartbeat(current_user=f"{user_id}/{feature_mode}", current_stage='GC')
                        
                        gc_start = datetime.now()
                        logger.info(f"{short_id} | GC | start")
                        gc_res = granger_analysis.run_granger_causality(A, S, max_lag=max_lag)
                        gc_elapsed = (datetime.now() - gc_start).total_seconds()
                        logger.info(f"{short_id} | GC | done elapsed={gc_elapsed:.2f}s")
                        p_A2S = gc_res.get('gc_A_to_S_pval', np.nan)
                        p_S2A = gc_res.get('gc_S_to_A_pval', np.nan)
                        
                        gc_result = {
                            'user_id': user_id, 'feature_mode': feature_mode,
                            'gc_optimal_lag': int(gc_res.get('gc_optimal_lag', 0)) if np.isfinite(gc_res.get('gc_optimal_lag', np.nan)) else 0,
                            'GC_A2S_pval': p_A2S, 'GC_S2A_pval': p_S2A,
                            'sign_GC': 'A2S' if (np.isfinite(p_A2S) and np.isfinite(p_S2A) and p_A2S < p_S2A) else 'S2A',
                            'n_samples': len(A),
                            'low_n': len(A) < self.quality.gc.min_samples,
                            'quality_passed': gc_valid
                        }
                        self.results['gc'].append(gc_result)
                        
                        # Save checkpoint after GC completion
                        self.save_checkpoint('gc', gc_result)
                        gc.collect()
                    except Exception as e:
                        logger.error(f"{short_id} | GC | FAIL | {e}")
                        self.errors.append({
                        'user_id': user_id, 
                        'feature_mode': feature_mode, 
                        'method': 'GC', 
                        'error': str(e),
                        'traceback': traceback.format_exc(),
                        'timestamp': datetime.now().isoformat()
                    })
        
            end_time = datetime.now()
            elapsed = (end_time - start_time).total_seconds()
            logger.info(f"{short_id} | DONE | mode={feature_mode} elapsed={elapsed:.1f}s ({elapsed/60:.1f}min)")
            
            # Mark combination as completed
            self.completed_combinations.add((user_id, feature_mode))
        
        except Exception as e:
            end_time = datetime.now()
            elapsed = (end_time - start_time).total_seconds()
            logger.error(f"{short_id} | FAIL | mode={feature_mode} elapsed={elapsed:.1f}s | {e}")
            logger.debug(f"Full traceback:\n{traceback.format_exc()}")
            self.errors.append({
                'user_id': user_id, 
                'feature_mode': feature_mode, 
                'method': 'PROCESS', 
                'error': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.now().isoformat()
            })
    
    def run(self, user_list, feature_modes):
        """
        Run the pipeline for all users and feature modes.

        @param {list[str]} user_list - UUIDs to process.
        @param {list[str]} feature_modes - Modes to analyze.
        @returns {None}
        @post Kicks off JVM, writes run_info, and iterates through users.
        """
        self.start_time = datetime.now()
        self.total_users = len(user_list) * len(feature_modes)
        
        # Load checkpoint if resuming
        if self.is_resume:
            completed_count = self.load_checkpoint()
            logger.info(f"RESUME: Skipping {completed_count} already completed combinations")
        
        # Write initial run_info
        self.write_run_info(seed=42)
        
        # Start JVM with config parameters
        jvm_cfg = self.config.get('jvm', {})
        analysis.start_jvm(
            xms=jvm_cfg.get('xms', '8g'),
            xmx=jvm_cfg.get('xmx', '16g'),
            gc_opts=jvm_cfg.get('opts', None)
        )
        logger.info("PRODUCTION PIPELINE: Schema v1.0 with tracking + checkpointing")
        
        # Process all combinations without progress bars; log per-user completion
        for user_id in user_list:
            for feat_mode in feature_modes:
                short_id = user_id[:8]
                start_u = datetime.now()
                logger.info(f"{short_id} | USER_START | mode={feat_mode}")
                self.process_user(user_id, feat_mode)
                self.users_completed += 1
                self.update_heartbeat(current_user=f"{user_id}/{feat_mode}", current_stage='completed')
                elapsed = (datetime.now() - start_u).total_seconds()
                logger.info(f"{short_id} | USER_DONE  | mode={feat_mode} elapsed={elapsed:.1f}s ({elapsed/60:.1f}min)")
        
        # Apply FDR and save
        self.finalize()
        
        analysis.shutdown_jvm()
    
    def finalize(self):
        """
        Apply FDR, persist all outputs, and generate final quality report.

        @returns {None}
        @post Output CSVs, run_info.yaml, and status.json reflect final state.
        """
        # Convert to DataFrames
        df_te = pd.DataFrame(self.results['te'])
        df_cte = pd.DataFrame(self.results['cte'])
        df_ste = pd.DataFrame(self.results['ste'])
        df_true_cte = pd.DataFrame(self.results['true_cte'])
        df_gc = pd.DataFrame(self.results['gc'])
        df_k = pd.DataFrame(self.results['k_selected'])
        df_hbin = pd.DataFrame(self.results['hbin_counts'])
        
        # Raw p-values only (no FDR; Bonferroni will be applied externally)
        if len(df_te) > 0:
            _, p_delta = compute_delta_pvalue(df_te, 'Delta_TE', 'p_Delta_TE')
            df_te['p_Delta_TE'] = p_delta
        if len(df_cte) > 0:
            _, p_delta = compute_delta_pvalue(df_cte, 'Delta_CTE', 'p_Delta_CTE')
            df_cte['p_Delta_CTE'] = p_delta
        if len(df_true_cte) > 0:
            _, p_delta = compute_delta_pvalue(df_true_cte, 'Delta_CTE_true', 'p_Delta_CTE_true')
            df_true_cte['p_Delta_CTE_true'] = p_delta
        
        # Save with exact schema order (including quality columns)
        te_cols = ['user_id','feature_mode','k','l','tau','TE_A2S','TE_S2A','Delta_TE','p_A2S','p_S2A','q_A2S','q_S2A','p_Delta_TE','q_Delta_TE','n_samples','low_n','quality_passed']
        cte_cols = ['user_id','feature_mode','k','l','tau','hour_bins','CTE_A2S','CTE_S2A','Delta_CTE','p_A2S','p_S2A','q_A2S','q_S2A','p_Delta_CTE','q_Delta_CTE','n_samples','n_samples_per_bin_min','low_n','low_n_hours','bins_filtered','quality_passed']
        true_cte_cols = ['user_id','feature_mode','k','l','tau','hour_bins','CTE_true_A2S','CTE_true_S2A','Delta_CTE_true','p_A2S','p_S2A','p_Delta_CTE_true','n_samples','n_samples_per_bin_min','low_n','low_n_hours','bins_filtered','quality_passed']
        ste_cols = ['user_id','feature_mode','k','tau','STE_A2S','STE_S2A','Delta_STE','p_A2S','p_S2A','q_A2S','q_S2A','p_Delta_STE','q_Delta_STE','n_samples','low_n','quality_passed']
        gc_cols = ['user_id','feature_mode','gc_optimal_lag','GC_A2S_pval','GC_S2A_pval','q_GC_A2S','q_GC_S2A','sign_GC','n_samples','low_n','quality_passed']
        
        # Save with guards for empty frames
        if len(df_te) > 0:
            # Save only columns that exist; add missing as NaN to preserve schema
            missing = [c for c in te_cols if c not in df_te.columns]
            for c in missing:
                df_te[c] = np.nan
            df_te[te_cols].to_csv(self.out_dir / 'per_user_te.csv', index=False)
        else:
            pd.DataFrame(columns=te_cols).to_csv(self.out_dir / 'per_user_te.csv', index=False)

        if len(df_cte) > 0:
            missing = [c for c in cte_cols if c not in df_cte.columns]
            for c in missing:
                df_cte[c] = np.nan
            df_cte[cte_cols].to_csv(self.out_dir / 'per_user_cte.csv', index=False)
        else:
            pd.DataFrame(columns=cte_cols).to_csv(self.out_dir / 'per_user_cte.csv', index=False)

            # Save legacy file and final-plan discrete-k2-l1 (block permutation) file
            if len(df_true_cte) > 0:
                df_true_cte[true_cte_cols].to_csv(self.out_dir / 'per_user_true_cte.csv', index=False)
                # Map to final-plan schema: rename Delta and select columns
                out2 = df_true_cte.copy()
                out2['Delta_TE'] = out2['Delta_CTE_true']
                final_cols = ['user_id','tau','k','l','hour_bins','CTE_true_A2S','CTE_true_S2A','Delta_TE','p_A2S','p_S2A']
                out2[final_cols].to_csv(self.out_dir / 'per_user_true_cte_discrete_k2l1_blockperm_FINAL.csv', index=False)
            else:
                pd.DataFrame(columns=true_cte_cols).to_csv(self.out_dir / 'per_user_true_cte.csv', index=False)
                pd.DataFrame(columns=['user_id','tau','k','l','hour_bins','CTE_true_A2S','CTE_true_S2A','Delta_TE','p_A2S','p_S2A']).to_csv(self.out_dir / 'per_user_true_cte_discrete_k2l1_blockperm_FINAL.csv', index=False)

        if len(df_ste) > 0:
            missing = [c for c in ste_cols if c not in df_ste.columns]
            for c in missing:
                df_ste[c] = np.nan
            df_ste[ste_cols].to_csv(self.out_dir / 'per_user_ste.csv', index=False)
        else:
            pd.DataFrame(columns=ste_cols).to_csv(self.out_dir / 'per_user_ste.csv', index=False)

        if len(df_gc) > 0:
            df_gc[gc_cols].to_csv(self.out_dir / 'per_user_gc.csv', index=False)
        else:
            pd.DataFrame(columns=gc_cols).to_csv(self.out_dir / 'per_user_gc.csv', index=False)
        
        if len(df_k) > 0:
            df_k.to_csv(self.out_dir / 'k_selected_by_user.csv', index=False)
        
        if len(df_hbin) > 0:
            df_hbin.to_csv(self.out_dir / 'hbin_counts.csv', index=False)
        
        if self.errors:
            pd.DataFrame(self.errors).to_csv(self.out_dir / 'error_log.csv', index=False)
        
        # Final heartbeat
        self.update_heartbeat()
        
        # Update run_info with completion time
        with open(self.out_dir / 'run_info.yaml') as f:
            run_info = yaml.safe_load(f)
        run_info['completed_at'] = datetime.now().isoformat()
        run_info['duration_seconds'] = int((datetime.now() - self.start_time).total_seconds())
        run_info['users_processed'] = len(set([r['user_id'] for r in self.results['te']]))
        run_info['errors_count'] = len(self.errors)
        
        with open(self.out_dir / 'run_info.yaml', 'w') as f:
            yaml.dump(run_info, f, default_flow_style=False, sort_keys=False)
        
        # Generate quality report if enabled
        if self.quality.generate_report:
            try:
                quality_profile = self.config.get('quality_profile', 'balanced')
                report_results = {
                    'profile': quality_profile,
                    'te': self.results['te'],
                    'cte': self.results['cte'],
                    'ste': self.results['ste'],
                    'gc': self.results['gc']
                }
                self.quality.generate_quality_report(report_results, self.out_dir)
            except Exception as e:
                logger.warning(f"Failed to generate quality report: {e}")
        
        logger.info(f"Pipeline completed: {run_info['users_processed']} users, {len(self.errors)} errors")
        
        return str(self.out_dir.resolve())


def main():
    """
    CLI entry point for production pipeline.

    @returns {None}
    @post Runs the configured analysis and exits with appropriate code.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Production ExtraSensory analysis pipeline with preset configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Preset configurations:
  smoke      Fast validation (2 users, k=4 fixed, 100 surrogates)
  k6_full    Pure AIS k=6 analysis (60 users, no constraints, 1000 surrogates)
  k4_fast    GUARDED_AIS k≤4 analysis (60 users, 4 modes, fast)
  24bin_cte  24-bin CTE high resolution (60 users, k≤4)

Examples:
  python run_production.py smoke
  python run_production.py k6_full --shard 0/4
  python run_production.py --config custom.yaml
        """
    )
    parser.add_argument('preset', nargs='?', help="Preset name (smoke, k6_full, k4_fast, 24bin_cte)")
    parser.add_argument('--config', help="Custom config file (overrides preset)")
    parser.add_argument('--resume', type=str, metavar='DIR', help="Resume from existing output directory")
    parser.add_argument('--shard', type=str, metavar='ID/TOTAL', help="Process shard: e.g., 0/4 means shard 0 of 4")
    parser.add_argument('--workers', type=str, metavar='N|auto', help="Launch N shard subprocesses (wrapper mode) or 'auto'")
    parser.add_argument('--no-progress', action='store_true', help="Disable console progress bars (useful for multi-worker wrapper mode)")
    parser.add_argument('--n-users', type=int, help="Override number of users from config")
    args = parser.parse_args()
    
    # Determine config file
    if args.config:
        config_path = args.config
        logger.info(f"CUSTOM CONFIG: {config_path}")
    elif args.preset:
        preset_map = {
            'smoke': 'config/presets/smoke.yaml',
            'k6_full': 'config/presets/k6_full.yaml',
            'k4_fast': 'config/presets/k4_fast.yaml',
            '24bin_cte': 'config/presets/24bin_cte.yaml'
        }
        if args.preset not in preset_map:
            logger.error(f"Unknown preset '{args.preset}'. Available: {list(preset_map.keys())}")
            sys.exit(1)
        config_path = preset_map[args.preset]
        logger.info(f"PRESET: {args.preset} → {config_path}")
    else:
        logger.error("No configuration specified. Use: python run_production.py <preset> or --config <file>")
        parser.print_help()
        sys.exit(1)

    # Resolve desired worker count
    def _parse_xmx_bytes(xmx_str: str) -> int:
        """
        Parse a JVM -Xmx string (e.g., '8g', '1024m') to bytes.

        @param {str} xmx_str - Heap size string.
        @returns {int} Equivalent number of bytes.
        @post Returns a reasonable fallback on parsing failure.
        """
        try:
            s = xmx_str.strip().lower()
            if s.endswith('g'):
                return int(float(s[:-1]) * (1024**3))
            if s.endswith('m'):
                return int(float(s[:-1]) * (1024**2))
            return int(s)
        except Exception:
            return 8 * 1024**3  # Fallback 8 GiB

    def _resolve_workers(arg_workers, config):
        """
        Resolve desired worker count from CLI arg and config.

        @param {str|int|None} arg_workers - Explicit count or 'auto' or None.
        @param {dict} config - Configuration dictionary.
        @returns {int} Number of workers to use.
        @post Uses CPU count and estimated RAM heuristic when 'auto'.
        """
        if arg_workers is None:
            return int(config.get('runtime', {}).get('concurrency', 1))
        if isinstance(arg_workers, str) and arg_workers.lower() == 'auto':
            import os
            cpu = max(os.cpu_count() or 1, 1)
            # Optional RAM-based cap
            max_by_ram = 999
            try:
                import psutil
                total = psutil.virtual_memory().total
                xmx = _parse_xmx_bytes(config.get('jvm', {}).get('xmx', '8g'))
                # keep headroom factor 0.7
                max_by_ram = max(int((total * 0.7) // max(xmx, 1)), 1)
            except Exception:
                pass
            # Heuristic: up to 4, half the CPUs, and RAM cap
            return max(min(min(cpu // 2 if cpu > 1 else 1, 4), max_by_ram), 1)
        # Numeric string
        try:
            w = int(arg_workers)
            return max(w, 1)
        except Exception:
            raise ValueError(f"Invalid --workers value: {arg_workers}")

    desired_workers = _resolve_workers(args.workers, yaml.safe_load(open(config_path))) if (args.workers or not args.shard) else 1

    # Wrapper mode: launch multiple shard subprocesses with a single command
    if not args.shard and desired_workers > 1:
        try:
            import yaml as _yaml
            cfg_for_log = _yaml.safe_load(open(config_path))
            logger.info(f"WRAPPER: Launching {desired_workers} shard subprocesses")
            import subprocess as _sp
            import sys as _sys
            procs = []
            for i in range(desired_workers):
                cmd = [_sys.executable, __file__]
                if args.preset:
                    cmd.append(args.preset)
                else:
                    cmd.extend(["--config", config_path])
                cmd.extend(["--shard", f"{i}/{desired_workers}"])
                if args.n_users:
                    cmd.extend(["--n-users", str(args.n_users)])
                # Suppress progress bars in children to avoid console thrashing
                cmd.append("--no-progress")
                if args.resume:
                    logger.warning("--resume ignored in wrapper mode (handled by child processes if needed)")
                # Do NOT propagate --workers to children
                logger.info(f"WRAPPER: Starting shard {i}/{desired_workers} → {' '.join(cmd)}")
                procs.append((_sp.Popen(cmd), i))
            # Wait for all children without rendering progress bars
            exit_code = 0
            for p, idx in procs:
                rc = p.wait()
                logger.info(f"WRAPPER: Shard {idx} exited with code {rc}")
                if rc != 0:
                    exit_code = rc
            _sys.exit(exit_code)
        except Exception as e:
            logger.error(f"WRAPPER: Failed to launch shards: {e}")
            sys.exit(1)

    pipeline = ProductionPipeline(config_path, resume_dir=args.resume, shard=args.shard, no_progress=args.no_progress)
    
    # Get user list
    data_root = Path(pipeline.config['data_root'])
    files = glob.glob(str(data_root / '*.features_labels.csv'))
    all_uuids = sorted([Path(f).stem.replace('.features_labels', '') for f in files])

    # Optional: override user list via a file of UUIDs (one per line)
    user_list_file = pipeline.config.get('user_list_file')
    user_list = None
    if user_list_file:
        try:
            lines = [l.strip() for l in Path(user_list_file).read_text(encoding='utf-8').splitlines()]
            desired = [u for u in lines if u and not u.startswith('#')]
            # Preserve order, include only those present in data_root
            user_list = [u for u in desired if u in all_uuids]
            if not user_list:
                logger.warning(f"USER LIST FILE provided but no valid UUIDs found: {user_list_file}")
        except Exception as e:
            logger.warning(f"Failed to read user_list_file '{user_list_file}': {e}")

    # Get n_users from config or override (ignored if user_list_file provided)
    if user_list is None:
        n_users = args.n_users if args.n_users else pipeline.config.get('n_users', len(all_uuids))
        user_list = all_uuids[:n_users]
    else:
        n_users = len(user_list)
    feature_modes = pipeline.config['feature_modes']
    
    logger.info(f"CONFIG: {n_users} users, {len(feature_modes)} modes, k_strategy={pipeline.config['k_selection']['strategy']}")
    
    # Apply user sharding if specified
    if args.shard:
        try:
            shard_id, total_shards = map(int, args.shard.split('/'))
            if shard_id < 0 or shard_id >= total_shards:
                raise ValueError(f"Invalid shard_id={shard_id}, must be 0 <= shard_id < {total_shards}")
            
            # Partition users: take every Nth user starting from shard_id
            original_count = len(user_list)
            user_list = user_list[shard_id::total_shards]
            logger.info(f"SHARD MODE: Processing shard {shard_id}/{total_shards} ({len(user_list)}/{original_count} users)")
        except Exception as e:
            logger.error(f"Failed to parse --shard argument '{args.shard}': {e}")
            logger.error("Expected format: --shard ID/TOTAL (e.g., --shard 0/4)")
            sys.exit(1)
    
    out_dir = pipeline.run(user_list, feature_modes)
    
    print(json.dumps({
        'status': 'completed',
        'OUT_DIR': out_dir,
        'config': config_path,
        'preset': args.preset if args.preset else 'custom',
        'users': len(user_list),
        'modes': len(feature_modes),
        'next_step': f'python tools/validate_outputs.py --dir {out_dir}'
    }, separators=(',', ':')))


if __name__ == "__main__":
    main()
```

## File: config/presets/production_k6_true_cte.yaml
```yaml
# FINAL RUN CONFIG
#
# This is the final production preset:
# - 60 users; AIS k-selection up to k=6 (73% users select k=6 in diagnostics)
# - True CTE is the core method; Global TE retained but allowed to OOM at k>=5 (record NaN and continue)
# - Adaptive surrogate stages up to 10k; FDR corrected per (family × tau)
#
# Full production run: 60 users, AIS up to k=6, 10k surrogates with adaptive early stop, Global TE + True CTE only

data_root: "data/ExtraSensory.per_uuid_features_labels"
out_dir: "analysis/out/production_k6_true_cte_<STAMP>"

n_users: 60
feature_modes: [composite]

quality_profile: "balanced"
taus: [1, 2]

# Hour bins
hour_bins: 6
conditional_transfer_entropy:
  hour_bins: 6
  method: "STRATIFIED-CTE"  # not used in this preset (analysis_modes excludes stratified_te)

# K-selection via AIS up to k=6
k_selection:
  strategy: "AIS"
  k_grid: [1, 2, 3, 4, 5, 6]
  k_max: 6
  undersampling_guard: true

# Significance testing: 10k upper bound with adaptive stages
surrogates: 10000
statistical:
  adaptive_surrogates:
    enabled: true
    stages: [1000, 3000, 10000]
    p_sig: 0.01
    p_nonsig: 0.20

fdr:
  families: [TE, TRUE_CTE]
  by_tau: true
  alpha: 0.05

# JVM: 6 workers × 8g
jvm:
  xms: "4g"
  xmx: "8g"
  opts:
    - "-XX:+UseG1GC"
    - "-XX:MaxGCPauseMillis=300"
    - "-Djava.awt.headless=true"

runtime:
  concurrency: 6
  checkpoint: true
  heartbeat_interval: 60

# Only run Global TE and True CTE
analysis_modes: [global_te, true_cte]

schema_version: "v1.0"
```

## File: config/presets/sanity_check_tau_neg_1.yaml
```yaml
data_root: "data/ExtraSensory.per_uuid_features_labels"
out_dir: "analysis/out/sanity_tau_neg1_<STAMP>"

n_users: 60
feature_modes: [composite]

# Discrete path settings
hour_bins: 6
taus: [1]  # Keep tau loop at 1; DELAY property controls lag=-1 internally

# K selection (unused by runner for discrete fixed k,l, but kept for completeness)
k_selection:
  strategy: "FIXED"
  k_fixed: 2

# Surrogates count for manual block permutation
surrogates: 1000

runtime:
  concurrency: 4
  checkpoint: true
  heartbeat_interval: 60

analysis_modes: [true_cte]

schema_version: "v1.0"
```
