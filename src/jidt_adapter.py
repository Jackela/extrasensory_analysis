"""JIDT adapters with 48G heap support and STRATIFIED-CTE implementation."""
import logging
import gc
import numpy as np
import jpype
from jpype.types import JArray, JInt
from typing import Tuple, Optional
from src.params import TEParams, CTEParams, STEParams

logger = logging.getLogger(__name__)


def validate_series(series: np.ndarray, base: int, name: str) -> np.ndarray:
    """Validate and convert series to int32."""
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
    """Convert numpy int32 array to Java int[]."""
    if not isinstance(py_array, np.ndarray):
        py_array = np.array(py_array, dtype=np.int32)
    if py_array.dtype != np.int32:
        py_array = py_array.astype(np.int32)
    return JArray(JInt)(py_array)


class DiscreteTE:
    """Wrapper for TransferEntropyCalculatorDiscrete with native 6-arg initialise for tau."""
    
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
        """Compute TE and p-value."""
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
            
            # Significance
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
        """Clean up calculator."""
        self.calc = None
        gc.collect()


class StratifiedCTE:
    """STRATIFIED-TE implementation of CTE: CTE(A→S|H) = Σ_h p(h)·TE_h(A→S)."""
    
    def __init__(self, params: CTEParams):
        self.params = params
    
    def compute(self, source: np.ndarray, dest: np.ndarray, cond: np.ndarray) -> Tuple[float, float]:
        """Compute stratified CTE and aggregate p-value.
        
        For tau>1: applies data-level lag BEFORE stratification (global alignment).
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
                
                if n_h < 50:  # Minimum samples per stratum
                    continue
                
                # Compute TE for this stratum
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


class SymbolicTE:
    """Wrapper for Symbolic TE using ordinal patterns and DiscreteTE."""
    
    def __init__(self, params: STEParams):
        self.params = params
    
    def ordinal_pattern_encode(self, series: np.ndarray) -> np.ndarray:
        """Encode series as ordinal patterns."""
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
        """Compute STE and p-value."""
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
