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
