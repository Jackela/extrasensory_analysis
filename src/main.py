# main.py
# Main script to run the full analysis pipeline.
# Iterates through all subjects, preprocesses their data,
# runs the JIDT analysis (TE and CTE), and aggregates results.

import logging
import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import src.settings as settings
import src.preprocessing as preprocessing
import src.analysis as analysis
import sys  # For exiting early if setup fails
import jpype

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def main_pipeline():
    """
    Defines and executes the main analysis pipeline.
    """
    logger.info("Starting analysis pipeline...")
    
    # --- 1. Initial Setup Checks and JVM Start ---
    try:
        # Verify essential settings before starting JVM
        if not os.path.exists(settings.JIDT_JAR_PATH):
            logger.error(
                "JIDT .jar file not found at path specified in src/settings.py: %s",
                settings.JIDT_JAR_PATH,
            )
            sys.exit(1)  # Exit if JIDT jar is missing
        if not os.path.isdir(settings.DATA_PATH):
            logger.error(
                "Data directory not found at path specified in src/settings.py: %s",
                settings.DATA_PATH,
            )
            sys.exit(1)  # Exit if data path is invalid

        # Attempt to start JVM and load classes early to catch issues
        analysis.start_jvm()
        _ = analysis.get_jidt_classes()  # Trigger class loading
        logger.info("JIDT JVM started and classes loaded successfully.")

    except Exception as e:
        logger.exception("FATAL ERROR during JIDT initialization: %s", e)
        logger.error(
            "Please check JIDT_JAR_PATH in src/settings.py and ensure Java environment is correctly configured."
        )
        analysis.shutdown_jvm()  # Attempt shutdown if started partially
        sys.exit(1)  # Exit if initialization fails

    try:
        # --- 2. Identify Subjects ---
        # Find all combined feature/label files based on the pattern
        subject_files = glob.glob(os.path.join(settings.DATA_PATH, "*.features_labels.csv"))
        # Extract UUIDs from filenames
        all_uuids = [os.path.basename(f).replace(".features_labels.csv", "") for f in subject_files]
        
        if not all_uuids:
            logger.warning("No subject data files ('*.features_labels.csv') found in %s.", settings.DATA_PATH)
            logger.warning("Analysis cannot proceed without data. Exiting.")
            sys.exit(0)  # Not a fatal error, but nothing to process

        logger.info("Found %d potential subjects based on files in %s.", len(all_uuids), settings.DATA_PATH)
        
        # --- 3. Process Each Subject ---
        all_results = []
        logger.info("Starting processing loop for each subject...")
        
        for uuid in tqdm(all_uuids, desc="Processing subjects"):
            subject_results = {'uuid': uuid}  # Initialize results dict for this subject
            try:
                # 3.1 Load Data
                raw_data = preprocessing.load_subject_data(uuid)
                
                # 3.2 Create Variables (A, S, H)
                series_A, series_S, series_H_binned = preprocessing.create_variables(raw_data)
                subject_results['data_length'] = len(series_A)  # Record data length after cleaning

                # 3.3 Determine Alphabet Sizes (Base) for JIDT
                # Ensure series are not empty before calculating max
                if len(series_A) == 0 or len(series_S) == 0 or len(series_H_binned) == 0:
                    raise ValueError("One or more processed series (A, S, H) are empty.")
                    
                base_A = int(np.max(series_A)) + 1 
                base_S = int(np.max(series_S)) + 1
                base_H_binned = settings.NUM_HOUR_BINS
                
                # Basic check for valid bases (should be >= 1, A=5, S=2, H=24 expected)
                if not (base_A >= 1 and base_S >= 1 and base_H_binned >= 1):
                    raise ValueError(f"Invalid base calculated: A={base_A}, S={base_S}, H_binned={base_H_binned}. Check data.")
                if base_A != 5:
                    tqdm.write(f"Warning for UUID {uuid}: Base for A is {base_A}, expected 5. Discretization might be unusual.")
                if base_S > 2:  # Should be 0 or 1
                    tqdm.write(f"Warning for UUID {uuid}: Base for S is {base_S}, expected 2. Check '{settings.COL_SITTING}' column.")
                if np.any(series_H_binned < 0) or np.any(series_H_binned >= base_H_binned):
                    raise ValueError(f"Binned hour values out of expected range [0, {base_H_binned - 1}] for UUID {uuid}.")

                # 3.4 Optimize History Length (k) using AIS (Proposal Req: 22)
                k_A = analysis.find_optimal_k_ais(series_A, base_A, settings.MAX_K_AIS)
                k_S = analysis.find_optimal_k_ais(series_S, base_S, settings.MAX_K_AIS)
                subject_results['k_A'] = k_A
                subject_results['k_S'] = k_S

                # 3.5 Run Transfer Entropy (TE) Analysis (Proposal Req: 5, 23, 24)
                te_results = analysis.run_te_analysis(
                    series_A, series_S, k_A, k_S, base_A, base_S
                )
                subject_results.update(te_results)  # Merge TE results into subject dict

                # 3.6 Run Conditional TE (CTE) Analysis (Proposal Req: 25, 26)
                cte_results = analysis.run_cte_analysis(
                    series_A, series_S, series_H_binned, k_A, k_S, base_A, base_S, base_H_binned
                )
                subject_results.update(cte_results)  # Merge CTE results

                # If all steps succeeded, add the complete results for this subject
                all_results.append(subject_results)

            # --- Error Handling for Individual Subjects ---
            except FileNotFoundError as e:
                tqdm.write(f"SKIPPING UUID {uuid}: Data file not found. Error: {e}")
            except ValueError as e:
                # Catches errors from preprocessing (e.g., insufficient data, discretization issues)
                # or base calculation issues.
                tqdm.write(f"SKIPPING UUID {uuid}: Data processing error. Error: {e}")
            except jpype.JException as e:
                # Catch Java exceptions specifically
                tqdm.write(f"SKIPPING UUID {uuid}: JIDT (Java) error during analysis. Error: {e}")
            except Exception as e:
                # Catch any other unexpected Python errors during processing for this subject
                tqdm.write(f"SKIPPING UUID {uuid}: Unexpected error during processing. Error: {type(e).__name__} - {e}")
                # Optionally add more detailed traceback logging here if needed for debugging
                
        # --- 4. Aggregate and Save Results ---
        logger.info("Subject processing loop finished.")
        if not all_results:
            logger.warning("No subjects were successfully processed. No results were saved.")
        else:
            logger.info("Successfully processed %d out of %d subjects.", len(all_results), len(all_uuids))
            results_df = pd.DataFrame(all_results)
            
            try:
                # Ensure the results directory exists
                results_dir = os.path.dirname(settings.RESULTS_FILE)
                if results_dir and not os.path.exists(results_dir):
                    os.makedirs(results_dir)
                    logger.info("Created results directory: %s", results_dir)
                    
                # Save the aggregated results to CSV
                results_df.to_csv(settings.RESULTS_FILE, index=False)
                logger.info("Aggregated results saved successfully to: %s", settings.RESULTS_FILE)
                
                # Display basic stats of the key result (Delta_TE)
                if 'Delta_TE' in results_df.columns:
                    logger.info(
                        "--- Summary Statistics for Delta_TE (Net Information Transfer) ---%s%s",
                        os.linesep,
                        results_df['Delta_TE'].describe(),
                    )
                else:
                    logger.warning("Delta_TE column not found in results, cannot display summary.")

            except IOError as e:
                logger.error("Could not write results to file '%s'. Error: %s", settings.RESULTS_FILE, e)
            except Exception as e:
                logger.exception("An unexpected error occurred while saving results. Error: %s", e)

    finally:  # Ensure JVM shutdown happens even if errors occurred during saving
        analysis.shutdown_jvm()
        logger.info("JIDT JVM shut down. Pipeline finished.")

# --- Script Entry Point ---
if __name__ == "__main__":
    main_pipeline()
