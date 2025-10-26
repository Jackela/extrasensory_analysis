# src/analysis.py
# Core JIDT analysis functions, called via jpype.
# This module implements the specific JIDT calls required by the proposal.

import logging
import jpype
import jpype.imports
from jpype.types import JArray, JInt
import numpy as np
import src.settings as settings # Use settings for consistency

logger = logging.getLogger(__name__)

# --- 1. JVM Management ---

_jvm_started = False

def start_jvm(xms="8g", xmx="16g", gc_opts=None):
    """Starts the JPype JVM with the JIDT classpath from settings.
    
    Args:
        xms: Initial heap size (default: 8g)
        xmx: Maximum heap size (default: 16g)
        gc_opts: Additional GC options (default: G1GC with 200ms pause)
    """
    global _jvm_started
    if _jvm_started or jpype.isJVMStarted():
        # print("JVM is already running.")
        _jvm_started = True
        return
    
    jar_path = settings.JIDT_JAR_PATH
    if not isinstance(jar_path, str) or not jar_path:
         raise ValueError("JIDT_JAR_PATH in settings.py is not configured correctly.")
    
    if gc_opts is None:
        gc_opts = ["-XX:+UseG1GC", "-XX:MaxGCPauseMillis=200"]
         
    try:
        jvm_args = [
            jpype.getDefaultJVMPath(),
            f"-Xms{xms}",
            f"-Xmx{xmx}",
            *gc_opts,
            "-Djava.awt.headless=true",
            f"-Djava.class.path={jar_path}",
        ]
        jpype.startJVM(*jvm_args, convertStrings=False)
        _jvm_started = True
        logger.info(f"JVM started: heap={xms}-{xmx}, GC={gc_opts}")
    except Exception as e:
        logger.error("Error starting JVM with jar path '%s': %s", jar_path, e)
        _jvm_started = False
        raise

def shutdown_jvm():
    """Shuts down the JPype JVM if it was started."""
    global _jvm_started
    if _jvm_started and jpype.isJVMStarted():
        jpype.shutdownJVM()
        _jvm_started = False

# --- 2. JIDT Class Loading ---

_jidt_classes = None

def get_jidt_classes() -> dict:
    """
    Imports and returns a dictionary of the required JIDT Java classes.
    Lazily loads classes on first call. Ensures JVM is started.
    """
    global _jidt_classes
    if _jidt_classes is None:
        start_jvm() # Ensure JVM is running before accessing Java classes
        try:
            _jidt_classes = {
                "AIS": jpype.JClass("infodynamics.measures.discrete.ActiveInformationCalculatorDiscrete"),
                "TE": jpype.JClass("infodynamics.measures.discrete.TransferEntropyCalculatorDiscrete"),
                "CTE": jpype.JClass("infodynamics.measures.discrete.ConditionalTransferEntropyCalculatorDiscrete")
            }
        except Exception as e:
            logger.error(
                "Error importing JIDT classes. Is JIDT_JAR_PATH ('%s') correct? Error: %s",
                settings.JIDT_JAR_PATH,
                e,
            )
            # Ensure JVM is shutdown if class loading fails
            shutdown_jvm() 
            raise
    return _jidt_classes

# --- 3. Helper Function ---

def java_array_int(py_array: np.ndarray) -> JArray:
    """Converts a numpy integer array to a Java int[] array for JIDT."""
    # Ensure the input is a numpy array of integers
    if not isinstance(py_array, np.ndarray):
        py_array = np.array(py_array)
    if not np.issubdtype(py_array.dtype, np.integer):
         raise TypeError(f"Input array must be integer type, not {py_array.dtype}")
         
    return JArray(JInt)(py_array)

# --- 4. Analysis Functions ---

def find_optimal_k_ais(series: np.ndarray, base: int, max_k: int) -> int:
    """
    Finds the optimal history length k for a time series by maximizing
    the Active Information Storage (AIS), as per Proposal Req: 22.
    """
    if base <= 1:
        # AIS calculation is trivial or ill-defined if base <= 1
        return 1
        
    classes = get_jidt_classes()
    AISCalculator = classes["AIS"]
    java_series = java_array_int(series)
    
    best_k = 1
    # Initialize with a very small number instead of -inf for stability
    best_ais = -1e18 

    for k in range(1, max_k + 1):
        try:
            # Construct with (base, k), Initialize, Add Observations, Compute
            calc = AISCalculator(base, k)
            calc.initialise()
            calc.addObservations(java_series)
            ais = calc.computeAverageLocalOfObservations()
            
            # Handle potential NaN or Inf results from JIDT gracefully
            if not np.isfinite(ais):
                logger.debug("Non-finite AIS value (%s) encountered for k=%d. Skipping.", ais, k)
                continue

            if ais > best_ais:
                best_ais = ais
                best_k = k
        except Exception as e:
            logger.error("Error calculating AIS for k=%d: %s", k, e)
            # Decide if we should continue searching or return current best
            continue # Let's try the next k
            
    # Add a check: if best_ais is still the initial very small number, maybe no valid AIS was computed
    if best_ais <= -1e18:
        logger.warning("Could not find a valid AIS maximum up to max_k=%d. Defaulting k to 1.", max_k)
        return 1
        
    return best_k


def run_te_analysis(series_A: np.ndarray, series_S: np.ndarray, 
                    k_A: int, k_S: int, base_A: int, base_S: int, tau: int = 1, num_surrogates: int = 1000) -> dict:
    """Computes TE(A->S), TE(S->A), and their p-values using JIDT adapter.
    
    Args:
        series_A, series_S: Discretized time series
        k_A, k_S: History lengths
        base_A, base_S: Alphabet sizes
        tau: Time delay parameter (default=1)
        num_surrogates: Number of surrogates for significance testing (default=1000)
    
    Returns:
        Dictionary with harmonized Delta=A→S−S→A.
    """
    from src.jidt_adapter import DiscreteTE
    from src.params import TEParams
    import gc
    results = {}
    
    # --- 1. Compute TE(A -> S) ---
    try:
        params_A2S = TEParams(
            base_source=base_A,
            base_dest=base_S,
            k_source=k_A,
            k_dest=k_S,
            tau=tau,
            num_surrogates=num_surrogates,
            seed=42
        )
        calc_A2S = DiscreteTE(params_A2S)
        te_A_to_S, p_A_to_S = calc_A2S.compute(series_A, series_S)
        calc_A2S.dispose()
        
        results['TE(A->S)'] = te_A_to_S
        results['p(A->S)'] = p_A_to_S
    except Exception as e:
        logger.error(f"TE(A->S) failed: {e}")
        results['TE(A->S)'] = np.nan
        results['p(A->S)'] = np.nan
    
    # --- 2. Compute TE(S -> A) ---
    try:
        params_S2A = TEParams(
            base_source=base_S,
            base_dest=base_A,
            k_source=k_S,
            k_dest=k_A,
            tau=tau,
            num_surrogates=num_surrogates,
            seed=42
        )
        calc_S2A = DiscreteTE(params_S2A)
        te_S_to_A, p_S_to_A = calc_S2A.compute(series_S, series_A)
        calc_S2A.dispose()
        
        results['TE(S->A)'] = te_S_to_A
        results['p(S->A)'] = p_S_to_A
    except Exception as e:
        logger.error(f"TE(S->A) failed: {e}")
        results['TE(S->A)'] = np.nan
        results['p(S->A)'] = np.nan
    
    # --- 3. Compute Net Transfer (Delta_TE = A→S − S→A) ---
    if np.isfinite(results.get('TE(A->S)', np.nan)) and np.isfinite(results.get('TE(S->A)', np.nan)):
        results['Delta_TE'] = results['TE(A->S)'] - results['TE(S->A)']
    else:
        results['Delta_TE'] = np.nan
    
    gc.collect()
    return results


def run_cte_analysis(series_A: np.ndarray, series_S: np.ndarray, series_cond: np.ndarray,
                     k_A: int, k_S: int, base_A: int, base_S: int, base_cond: int, tau: int = 1, num_surrogates: int = 1000) -> dict:
    """Computes CTE(A->S|cond) and CTE(S->A|cond) using JIDT adapter.
    
    Args:
        series_A, series_S, series_cond: Discretized time series (cond is hour bins)
        k_A, k_S: History lengths
        base_A, base_S, base_cond: Alphabet sizes
        tau: Time delay parameter (default=1)
        num_surrogates: Number of surrogates for significance testing (default=1000)
    
    Returns:
        Dictionary with harmonized Delta=A→S−S→A.
    """
    from src.jidt_adapter import StratifiedCTE
    from src.params import CTEParams
    import gc
    results = {}
    results['cte_k_reduced'] = False  # NEVER reduce k
    
    # --- 1. Compute CTE(A -> S | H) using STRATIFIED-TE---
    try:
        params_A2S = CTEParams(
            base_source=base_A,
            base_dest=base_S,
            base_cond=base_cond,
            k_source=k_A,
            k_dest=k_S,
            num_cond_bins=1,
            tau=tau,
            num_surrogates=num_surrogates,
            seed=42
        )
        calc_A2S = StratifiedCTE(params_A2S)
        cte_A_to_S, p_cte_A_to_S = calc_A2S.compute(series_A, series_S, series_cond)
        
        results['CTE(A->S|H_bin)'] = cte_A_to_S
        results['p_cte(A->S|H_bin)'] = p_cte_A_to_S
    except Exception as e:
        logger.error(f"CTE(A->S|H) failed: {e}")
        results['CTE(A->S|H_bin)'] = np.nan
        results['p_cte(A->S|H_bin)'] = np.nan
    
    # --- 2. Compute CTE(S -> A | H) using STRATIFIED-TE ---
    try:
        params_S2A = CTEParams(
            base_source=base_S,
            base_dest=base_A,
            base_cond=base_cond,
            k_source=k_S,
            k_dest=k_A,
            num_cond_bins=1,
            tau=tau,
            num_surrogates=num_surrogates,
            seed=42
        )
        calc_S2A = StratifiedCTE(params_S2A)
        cte_S_to_A, p_cte_S_to_A = calc_S2A.compute(series_S, series_A, series_cond)
        
        results['CTE(S->A|H_bin)'] = cte_S_to_A
        results['p_cte(S->A|H_bin)'] = p_cte_S_to_A
    except Exception as e:
        logger.error(f"CTE(S->A|H) failed: {e}")
        results['CTE(S->A|H_bin)'] = np.nan
        results['p_cte(S->A|H_bin)'] = np.nan
    
    # --- 3. Compute Net Conditional Transfer (Delta_CTE = A→S − S→A) ---
    if np.isfinite(results.get('CTE(A->S|H_bin)', np.nan)) and np.isfinite(results.get('CTE(S->A|H_bin)', np.nan)):
        results['Delta_CTE_bin'] = results['CTE(A->S|H_bin)'] - results['CTE(S->A|H_bin)']
    else:
        results['Delta_CTE_bin'] = np.nan
    
    gc.collect()
    return results

# Optional: Add a main block for basic testing if desired
# if __name__ == '__main__':
#     # Example usage (requires dummy data)
#     print("Basic analysis module check...")
#     try:
#         start_jvm()
#         classes = get_jidt_classes()
#         print("JIDT classes loaded.")
#         # Create dummy data
#         # series_a = np.random.randint(0, 5, size=100)
#         # series_s = np.random.randint(0, 2, size=100)
#         # series_h = np.random.randint(0, 24, size=100)
#         # base_a, base_s, base_h = 5, 2, 24
#         # k_a = find_optimal_k_ais(series_a, base_a, settings.MAX_K_AIS)
#         # k_s = find_optimal_k_ais(series_s, base_s, settings.MAX_K_AIS)
#         # print(f"Optimal k_A={k_a}, k_S={k_s}")
#         # te_res = run_te_analysis(series_a, series_s, k_a, k_s, base_a, base_s)
#         # print("TE results:", te_res)
#         # cte_res = run_cte_analysis(series_a, series_s, series_h_cond, k_a, k_s, base_a, base_s, base_h_cond)
#         # print("CTE results:", cte_res)
#     except Exception as e:
#         print(f"Error during basic check: {e}")
#     finally:
#         shutdown_jvm()
#     print("Check complete.")
