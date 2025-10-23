---
description: "Implementation plan for ExtraSensory Transfer Entropy Analysis"
---

# Implementation Plan: ExtraSensory Transfer Entropy Analysis

## Overview

This project implements a quantitative investigation of the asymmetric predictive relationship between physical activity level (A) and sitting state (S) using transfer entropy analysis on the ExtraSensory dataset.

## Core Technology

- **Language**: Python 3.8+
- **Paradigm**: Modular functional programming with clear separation of concerns
- **Data Processing**: Pandas for DataFrame operations, NumPy for numerical computations

## Key Libraries

### Data Processing & Analysis
- **pandas**: Data loading, manipulation, and aggregated results storage
- **numpy**: Numerical operations on time series data
- **scipy**: Statistical functions (`scipy.stats.zscore`, `scipy.stats.wilcoxon`)
- **scikit-learn**: Preprocessing utilities (`sklearn.preprocessing.KBinsDiscretizer`)

### Information Theory & JIDT Integration
- **jpype1**: Python-Java bridge for JIDT library integration
- **JIDT** (`infodynamics.jar`): Core information-theoretic calculations
  - Active Information Storage (AIS)
  - Transfer Entropy (TE)
  - Conditional Transfer Entropy (CTE)
  - Permutation-based significance testing

### Utilities & Reporting
- **tqdm**: Progress bars for batch processing loops
- **jupyter**: Notebook environment for final report
- **nbformat**: Programmatic notebook generation (if needed)
- **matplotlib**: Visualization library for plots
- **seaborn**: Statistical visualization enhancements

## Project Structure

### Directory Layout

```
extrasensory_analysis/
├── src/
│   ├── settings.py          # Configuration constants, file paths, parameters
│   ├── preprocessing.py     # Data loading and variable generation (A, S, H)
│   └── analysis.py          # JIDT integration (JVM, AIS, TE, CTE)
├── data/                    # ExtraSensory dataset (*.features_labels.csv files)
├── results/                 # Output directory for CSV results
├── main.py                  # Main orchestration script (batch processing)
├── report.ipynb            # Final Jupyter Notebook with analysis and visualizations
├── infodynamics.jar        # JIDT library (Java)
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation
```

### Module Responsibilities

#### `src/settings.py`
- **Purpose**: Centralized configuration management
- **Contents**:
  - File path constants (data directory, JIDT jar path, output paths)
  - Analysis parameters (number of surrogate samples for permutation tests)
  - Column name mappings (ExtraSensory CSV columns)
  - Discretization parameters (number of quantile bins)

#### `src/preprocessing.py`
- **Purpose**: Data loading and time series preprocessing
- **Key Functions**:
  - `load_subject_data(uuid: str) -> pd.DataFrame`: Load individual subject CSV
  - `create_variable_S(df: pd.DataFrame) -> np.ndarray`: Extract binary sitting label
  - `create_variable_A(df: pd.DataFrame) -> np.ndarray`: Z-score + quantile discretization of accelerometer magnitude
  - `create_variable_H(df: pd.DataFrame) -> np.ndarray`: Extract hour-of-day from timestamp
  - `align_and_clean(A, S, H) -> Tuple[np.ndarray, ...]`: Handle missing values and align series

#### `src/analysis.py`
- **Purpose**: JIDT integration and information-theoretic calculations
- **Key Functions**:
  - `start_jvm(jar_path: str) -> None`: Initialize JPype JVM with JIDT
  - `stop_jvm() -> None`: Shutdown JVM cleanly
  - `numpy_to_java_int_array(arr: np.ndarray) -> jarray`: Convert NumPy array to Java int[]
  - `optimize_k_AIS(series: np.ndarray) -> int`: Maximize AIS to find optimal history length
  - `calculate_TE(source, target, k_source, k_target) -> Tuple[float, float]`: Calculate TE and p-value
  - `calculate_CTE(source, target, condition, k_source, k_target) -> Tuple[float, float]`: Calculate CTE and p-value
  - `analyze_subject(A, S, H) -> dict`: Full analysis pipeline for one subject

#### `main.py`
- **Purpose**: Batch processing orchestration
- **Workflow**:
  1. Discover subject UUIDs from data directory
  2. Initialize JVM
  3. Loop through subjects with progress bar (tqdm)
  4. For each subject:
     - Call preprocessing functions
     - Call analysis functions
     - Handle errors gracefully (try-except per subject)
  5. Aggregate results into pandas DataFrame
  6. Save to CSV
  7. Cleanup JVM

#### `report.ipynb`
- **Purpose**: Final deliverable with statistical analysis and visualizations
- **Sections**:
  1. Introduction and methodology
  2. Data loading (results CSV)
  3. Descriptive statistics
  4. Wilcoxon signed-rank tests (ΔTE and ΔCTE)
  5. Visualizations (distributions, paired comparisons)
  6. Conclusions and interpretation

## Data Flow

```
ExtraSensory CSV files (data/)
    ↓
preprocessing.py (load_subject_data)
    ↓
preprocessing.py (create A, S, H variables)
    ↓
preprocessing.py (align_and_clean)
    ↓
analysis.py (optimize_k_AIS for A and S)
    ↓
analysis.py (calculate_TE: A→S, S→A)
    ↓
analysis.py (calculate_CTE: A→S|H, S→A|H)
    ↓
main.py (aggregate results)
    ↓
results/extrasensory_te_results.csv
    ↓
report.ipynb (statistical tests + visualizations)
    ↓
Final conclusions on hypothesis H₁
```

## JIDT Integration Details

### Calculator Selection
- **ActiveInformationCalculatorDiscrete**: For AIS-based k optimization
- **TransferEntropyCalculatorDiscrete**: For TE(A→S) and TE(S→A)
- **ConditionalTransferEntropyCalculatorDiscrete**: For CTE(A→S|H) and CTE(S→A|H)
- **TransferEntropySignificanceCalculatorDiscrete**: For TE permutation tests (≥1000 surrogates)
- **ConditionalTransferEntropySignificanceCalculatorDiscrete**: For CTE permutation tests (≥1000 surrogates)

### Input Data Conversion
All Python NumPy arrays must be converted to Java `int[]` arrays using JPype:
```python
java_array = jpype.JArray(jpype.JInt)(numpy_array.astype(int))
```

### Error Handling Strategy
- JVM initialization errors: Check JIDT jar path and Java installation
- JIDT calculation errors: Log subject UUID, skip subject, continue batch
- Data preprocessing errors: Log subject UUID, skip subject, continue batch
- I/O errors: Fail fast with clear error message

## Variable Definitions (Per Proposal)

### Variable A (Activity Level)
1. Extract `raw_acc:magnitude_stats:mean` column from subject CSV
2. Apply z-score normalization **within each subject** using `scipy.stats.zscore`
3. Apply 5-bin quantile discretization using `sklearn.preprocessing.KBinsDiscretizer`
4. Result: Discrete series with values {0, 1, 2, 3, 4}

### Variable S (Sitting State)
1. Extract `label:SITTING` column from subject CSV
2. Binary encoding: 1 if sitting, 0 otherwise
3. Result: Binary series with values {0, 1}

### Variable H (Hour of Day)
1. Extract timestamp from subject CSV
2. Parse to datetime and extract hour component
3. Result: Integer series with values {0, 1, ..., 23}

## Analysis Parameters

- **Number of Bins for Variable A**: 5 (quintiles)
- **AIS Optimization Range**: k ∈ {1, 2, ..., 10} (search space for history length)
- **Permutation Test Surrogates**: ≥1000 (for statistical significance)
- **Significance Level**: α = 0.05 (standard threshold)

## Testing Strategy

### Integration Testing
- **JIDT Connection Test**: Verify JVM starts, JIDT classes load, simple calculation runs
- **Preprocessing Validation**: Test on single subject, verify A/S/H value ranges and alignment
- **End-to-End Test**: Process 1-2 subjects fully, verify result structure and CSV output

### Validation Checkpoints
- After Phase 2 (Foundational): Test JVM lifecycle and array conversion
- After Phase 3 (US1): Validate preprocessing on sample subject
- After Phase 4 (US2): Validate TE calculation on sample subject
- After Phase 5 (US3): Validate CTE calculation on sample subject
- After Phase 6 (US4): Verify batch processing and CSV output
- After Phase 7 (US5): Verify notebook runs without errors

## Performance Considerations

- **Memory**: JIDT calculations are memory-intensive; process subjects sequentially
- **Computation Time**: TE/CTE with permutation testing takes ~1-5 minutes per subject
- **Batch Processing**: Use tqdm for progress tracking; expect hours for full dataset
- **Parallelization**: Not implemented in MVP; future optimization opportunity

## Dependencies Management

### requirements.txt
```
pandas>=1.3.0
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=0.24.0
jpype1>=1.3.0
tqdm>=4.62.0
jupyter>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
nbformat>=5.1.0
```

### External Dependency
- **JIDT** (`infodynamics.jar`): Must be present at project root
- **Java Runtime**: JPype requires Java 8+ installed on system

## Deliverable

**Final Output**: `report.ipynb`
- Contains all analysis code, statistical results, and visualizations
- Self-contained: Can be re-run to reproduce results
- Includes narrative sections explaining methodology and conclusions
- Presents Wilcoxon test results for hypotheses H₁ (ΔTE) and robustness (ΔCTE)

## Implementation Phases (from tasks.md)

1. **Setup**: Project structure, dependencies, configuration
2. **Foundational**: JIDT integration, data loading utilities
3. **User Story 1**: Variable preprocessing (A, S, H)
4. **User Story 2**: AIS optimization and TE calculation
5. **User Story 3**: Conditional TE robustness check
6. **User Story 4**: Batch processing and aggregation
7. **User Story 5**: Statistical analysis and reporting
8. **Polish**: Documentation and code quality

## Success Criteria

1. ✅ All subject CSVs successfully processed (or errors logged with continuation)
2. ✅ Results CSV contains: UUID, k_A, k_S, TE(A→S), TE(S→A), p-values, ΔTE, CTE values, ΔCTE
3. ✅ Wilcoxon test executed on ΔTE distribution
4. ✅ Wilcoxon test executed on ΔCTE distribution
5. ✅ Visualizations generated (ΔTE distribution, TE paired comparison, CTE robustness)
6. ✅ Final conclusion stated in report.ipynb regarding hypothesis H₁
7. ✅ Code adheres to SOLID principles with clear modularity
8. ✅ Error handling prevents batch failures due to individual subject issues

