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

def start_jvm():
    """Starts the JPype JVM with the JIDT classpath from settings."""
    global _jvm_started
    if _jvm_started or jpype.isJVMStarted():
        # print("JVM is already running.")
        _jvm_started = True
        return
    
    jar_path = settings.JIDT_JAR_PATH
    if not isinstance(jar_path, str) or not jar_path:
         raise ValueError("JIDT_JAR_PATH in settings.py is not configured correctly.")
         
    try:
        jpype.startJVM(
            jpype.getDefaultJVMPath(),
            "-Xmx8G",  # raise JVM heap to handle large CTE computations
            f"-Djava.class.path={jar_path}",
            convertStrings=False
        )
        _jvm_started = True
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
                    k_A: int, k_S: int, base_A: int, base_S: int) -> dict:
    """
    Computes TE(A->S), TE(S->A), and their p-values using JIDT discrete calculators.
    Uses NUM_SURROGATES from settings. Returns a dictionary of results.
    """
    
    classes = get_jidt_classes()
    TECalculator = classes["TE"]
    num_surrogates = settings.NUM_SURROGATES

    results = {}

    # --- 1. Compute TE(A -> S) ---
    common_base = max(base_A, base_S)

    logger.debug(
        "Calling TE(A->S) with common_base=%d, base_dest=%d, base_source=%d, "
        "k_dest=%d, k_source=%d, len(A)=%d, len(S)=%d",
        common_base,
        base_S,
        base_A,
        k_S,
        k_A,
        len(series_A),
        len(series_S),
    )
    logger.debug("series_A[:10] = %s", series_A[:10])
    logger.debug("series_S[:10] = %s", series_S[:10])
    try:
        # Construct with (base, k), Initialize, Add Observations, Compute
        calc_A_to_S = TECalculator(common_base, k_S, k_A)
        calc_A_to_S.initialise()
        calc_A_to_S.addObservations(java_array_int(series_A), java_array_int(series_S))
        te_A_to_S = calc_A_to_S.computeAverageLocalOfObservations()
        
        # Compute Significance using computeSignificance method
        measure_dist = calc_A_to_S.computeSignificance(num_surrogates)
        p_A_to_S = measure_dist.pValue
        
        results['TE(A->S)'] = te_A_to_S if np.isfinite(te_A_to_S) else np.nan
        results['p(A->S)'] = p_A_to_S if np.isfinite(p_A_to_S) else np.nan
        
    except Exception as e:
        logger.error("Error calculating TE(A->S): %s", e)
        results['TE(A->S)'] = np.nan
        results['p(A->S)'] = np.nan

    # --- 2. Compute TE(S -> A) ---
    try:
        # Construct with (base, k), Initialize, Add Observations, Compute
        calc_S_to_A = TECalculator(common_base, k_A, k_S)
        calc_S_to_A.initialise()
        calc_S_to_A.addObservations(java_array_int(series_S), java_array_int(series_A))
        te_S_to_A = calc_S_to_A.computeAverageLocalOfObservations()
        
        # Compute Significance using computeSignificance method
        measure_dist = calc_S_to_A.computeSignificance(num_surrogates)
        p_S_to_A = measure_dist.pValue

        results['TE(S->A)'] = te_S_to_A if np.isfinite(te_S_to_A) else np.nan
        results['p(S->A)'] = p_S_to_A if np.isfinite(p_S_to_A) else np.nan
        
    except Exception as e:
        logger.error("Error calculating TE(S->A): %s", e)
        results['TE(S->A)'] = np.nan
        results['p(S->A)'] = np.nan

    # --- 3. Compute Net Transfer (Delta_TE) ---
    # Only compute if both TE values are valid numbers
    if np.isfinite(results.get('TE(A->S)', np.nan)) and np.isfinite(results.get('TE(S->A)', np.nan)):
        results['Delta_TE'] = results['TE(A->S)'] - results['TE(S->A)']
    else:
        results['Delta_TE'] = np.nan
    
    return results


def run_cte_analysis(series_A: np.ndarray, series_S: np.ndarray, series_H_binned: np.ndarray,
                     k_A: int, k_S: int, base_A: int, base_S: int, base_H_binned: int) -> dict:
    """
    Robustness Check: Computes CTE(A->S|H_bin) and CTE(S->A|H_bin) using binned hour-of-day.
    Uses NUM_SURROGATES from settings. Returns a dictionary of results.
    """
    
    classes = get_jidt_classes()
    CTECalculator = classes["CTE"]
    num_surrogates = settings.NUM_SURROGATES
    
    results = {}
    common_base_cte = max(base_A, base_S, base_H_binned)

    # --- 1. Compute CTE(A -> S | H) ---
    # Use 3-parameter constructor: (base_dest, base_source, base_cond)
    # Then set k via properties
    k_cond = 1  # Use single-step history for conditional variable (hour bins)

    num_conditionals = 1  # Single conditional variable (binned hour)

    try:
        # Construct calculator with unified base and explicit history lengths
        calc_A_to_S_H = CTECalculator(common_base_cte, k_S, num_conditionals, common_base_cte)
        calc_A_to_S_H.initialise()
        # Add Observations, Compute
        calc_A_to_S_H.addObservations(java_array_int(series_A), java_array_int(series_S), java_array_int(series_H_binned))
        cte_A_to_S = calc_A_to_S_H.computeAverageLocalOfObservations()

        # Compute Significance using computeSignificance method
        measure_dist = calc_A_to_S_H.computeSignificance(num_surrogates)
        p_cte_A_to_S = measure_dist.pValue
        
        results['CTE(A->S|H_bin)'] = cte_A_to_S if np.isfinite(cte_A_to_S) else np.nan
        results['p_cte(A->S|H_bin)'] = p_cte_A_to_S if np.isfinite(p_cte_A_to_S) else np.nan

    except Exception as e:
        logger.error("Error calculating CTE(A->S|H_bin): %s", e)
        results['CTE(A->S|H_bin)'] = np.nan
        results['p_cte(A->S|H_bin)'] = np.nan

    # --- 2. Compute CTE(S -> A | H) ---
    try:
        # Construct calculator with unified base and explicit history lengths
        calc_S_to_A_H = CTECalculator(common_base_cte, k_A, num_conditionals, common_base_cte)
        calc_S_to_A_H.initialise()
        # Add Observations, Compute
        calc_S_to_A_H.addObservations(java_array_int(series_S), java_array_int(series_A), java_array_int(series_H_binned))
        cte_S_to_A = calc_S_to_A_H.computeAverageLocalOfObservations()

        # Compute Significance using computeSignificance method
        measure_dist = calc_S_to_A_H.computeSignificance(num_surrogates)
        p_cte_S_to_A = measure_dist.pValue
        
        results['CTE(S->A|H_bin)'] = cte_S_to_A if np.isfinite(cte_S_to_A) else np.nan
        results['p_cte(S->A|H_bin)'] = p_cte_S_to_A if np.isfinite(p_cte_S_to_A) else np.nan

    except Exception as e:
        logger.error("Error calculating CTE(S->A|H_bin): %s", e)
        results['CTE(S->A|H_bin)'] = np.nan
        results['p_cte(S->A|H_bin)'] = np.nan

    # --- 3. Compute Net Conditional Transfer (Delta_CTE) ---
    if np.isfinite(results.get('CTE(A->S|H_bin)', np.nan)) and np.isfinite(results.get('CTE(S->A|H_bin)', np.nan)):
        results['Delta_CTE_bin'] = results['CTE(A->S|H_bin)'] - results['CTE(S->A|H_bin)']
    else:
        results['Delta_CTE_bin'] = np.nan
        
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
#         # cte_res = run_cte_analysis(series_a, series_s, series_h_binned, k_a, k_s, base_a, base_s, base_h_binned)
#         # print("CTE results:", cte_res)
#     except Exception as e:
#         print(f"Error during basic check: {e}")
#     finally:
#         shutdown_jvm()
#     print("Check complete.")
