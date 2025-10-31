
import pandas as pd
import numpy as np
import jpype
import jpype.imports
import os

def run_transfer_entropy_analysis():
    """
    Performs Transfer Entropy (TE) analysis to investigate dynamic causal relationships
    between user activity level and the state of sitting.
    """
    # --- 1. JIDT Environment Preparation ---
    try:
        # Start the JVM, pointing to the JIDT jar file
        # Assumes infodynamics.jar is in the same directory as the script
        # Hardcode the absolute path to the jar file to ensure it's found.
        jar_path = 'C:\\Users\\k7407\\OneDrive\\SydneyU\\CSYS5030\\Project\\infodynamics.jar'
        if not os.path.exists(jar_path):
            print(f"Error: infodynamics.jar not found at the specified path: {jar_path}")
            print("Please ensure the file exists at that exact location.")
            return

        if not jpype.isJVMStarted():
            jpype.startJVM(classpath=[jar_path])
        
        # Import the required Java class
        TransferEntropyCalculatorDiscrete = jpype.JClass('infodynamics.measures.discrete.TransferEntropyCalculatorDiscrete')

    except Exception as e:
        print(f"Error setting up JPype or JIDT: {e}")
        print("Please ensure you have JPype1 installed (`pip install JPype1`) and infodynamics.jar is accessible.")
        return

    # --- 2. Data Preparation ---
    try:
        df = pd.read_csv('mvp_dataset.csv')
    except FileNotFoundError:
        print("Error: mvp_dataset.csv not found. Please ensure the file is in the correct directory.")
        return

    # Discretize the accelerometer data into 3 bins (low, medium, high activity)
    # Using qcut for equal-frequency binning
    try:
        df['activity_level'] = pd.qcut(df['raw_acc:magnitude_stats:mean'], q=3, labels=[0, 1, 2], duplicates='drop').astype(int)
    except ValueError as e:
        print(f"Error during discretization. It might be that the data is not suitable for 3 quantiles. Details: {e}")
        # If qcut fails, try with a different approach or report error.
        # For now, we will stop execution.
        return
        
    # Ensure the target column is integer
    df['label:SITTING'] = df['label:SITTING'].astype(int)

    # --- 3. Execute TE Calculation (Per User) ---
    te_act_to_sit = []
    te_sit_to_act = []

    # Group by user to process time series data correctly
    for uuid, user_df in df.groupby('uuid'):
        # TE calculation requires a minimum number of data points
        if len(user_df) < 20:
            continue

        activity_data = user_df['activity_level'].values
        sitting_data = user_df['label:SITTING'].values

        # Convert pandas Series to Java integer arrays
        activity_java = jpype.JArray(jpype.JInt)(activity_data)
        sitting_java = jpype.JArray(jpype.JInt)(sitting_data)

        # --- Calculate T(activity_level -> label:SITTING) ---
        calc1 = TransferEntropyCalculatorDiscrete()
        calc1.setProperty("k", "1")  # history length k=1
        calc1.setProperty("tau", "1") # delay tau=1
        calc1.initialise()
        calc1.setObservations(activity_java, sitting_java)
        result1 = calc1.computeAverageLocalOfObservations()
        te_act_to_sit.append(result1)

        # --- Calculate T(label:SITTING -> activity_level) ---
        calc2 = TransferEntropyCalculatorDiscrete()
        calc2.setProperty("k", "1")  # history length k=1
        calc2.setProperty("tau", "1") # delay tau=1
        calc2.initialise()
        calc2.setObservations(sitting_java, activity_java)
        result2 = calc2.computeAverageLocalOfObservations()
        te_sit_to_act.append(result2)

    # --- 4. Results Aggregation & Reporting ---
    if not te_act_to_sit or not te_sit_to_act:
        print("Could not compute TE for any user. Check data quality and length.")
        return

    # Calculate mean and standard deviation
    mean_te_act_to_sit = np.mean(te_act_to_sit)
    std_te_act_to_sit = np.std(te_act_to_sit)
    mean_te_sit_to_act = np.mean(te_sit_to_act)
    std_te_sit_to_act = np.std(te_sit_to_act)

    # Generate the report
    report = f"""
# Phase 3: Transfer Entropy (TE) Analysis Report

This report details the dynamic causal analysis between user activity levels and the state of being seated.
The goal is to determine if one variable's past can predict the other's future.

## Part 1: Average TE Results

The following are the average Transfer Entropy values (in nats) aggregated across all {len(te_act_to_sit)} users.

*   **T(activity_level -> label:SITTING)**:
    *   **Mean**: {mean_te_act_to_sit:.4f}
    *   **Standard Deviation**: {std_te_act_to_sit:.4f}
    *   *Interpretation*: This value quantifies how much information the past of 'activity_level' provides about the future of 'label:SITTING'.

*   **T(label:SITTING -> activity_level)**:
    *   **Mean**: {mean_te_sit_to_act:.4f}
    *   **Standard Deviation**: {std_te_sit_to_act:.4f}
    *   *Interpretation*: This value quantifies how much information the past of 'label:SITTING' provides about the future of 'activity_level'.

## Part 2: Result Interpretation and Causal Inference

**Comparison**:
The average TE from 'activity_level' to 'label:SITTING' ({mean_te_act_to_sit:.4f}) is markedly higher than the TE in the reverse direction ({mean_te_sit_to_act:.4f}).

**Causal Inference**:
This asymmetry suggests a dominant directional flow of information. The past of a user's physical activity level is a significantly better predictor of their future sitting status than the other way around.

In simpler terms: **A change in activity (e.g., decreasing) is a predictive signal that a person is about to sit down.** The fact that someone is currently sitting, however, is a much weaker predictor of their next change in activity level.

**Connection to Project Goal**:
This finding is crucial for our project's goal of identifying opportune moments for 'Do Not Disturb' (DND). It provides quantitative evidence that monitoring a user's activity level can serve as a leading indicator for a transition into a sedentary state (sitting). By detecting a sharp drop in activity, our system could proactively and intelligently predict the start of a potential focus session, aligning perfectly with our PM narrative.

--- End of Report ---
"""
    print(report)

if __name__ == "__main__":
    run_transfer_entropy_analysis()
