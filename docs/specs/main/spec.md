---
description: "Feature specification for ExtraSensory Transfer Entropy Analysis"
---

# Specification: ExtraSensory Transfer Entropy Analysis

## Overview

This project implements a quantitative investigation of the asymmetric predictive relationship between physical activity level (A) and sitting state (S) using the ExtraSensory dataset, as outlined in the CSYS5030 research proposal (Proposal_WeixuanKong.pdf).

## Research Goal

**Primary Hypothesis (H₁)**: Activity level (A) is more predictive of an upcoming transition into a sitting state (S) than the reverse.

**Formal Statement**: E[ΔTE] > 0, where ΔTE = TE(A → S) - TE(S → A)

**Purpose**: Test whether physical activity patterns can predict sitting behavior better than sitting behavior predicts activity patterns. This asymmetry could inform the design of mobile health interventions (e.g., just-in-time notifications to reduce sedentary behavior).

## User Stories

### User Story 1 (P1): Variable Preprocessing 🎯 MVP Foundation

**As a** researcher
**I want to** transform raw ExtraSensory sensor data into analysis-ready discrete time series
**So that** I can perform information-theoretic analysis using JIDT

**Acceptance Criteria**:
- Variable A (Activity Level) is created from `raw_acc:magnitude_stats:mean`:
  - Z-score normalized within each subject
  - Discretized into 5 quantile bins {0, 1, 2, 3, 4}
- Variable S (Sitting State) is extracted from `label:SITTING`:
  - Binary encoding {0, 1}
- Variable H (Hour of Day) is extracted from timestamp:
  - Integer encoding {0, 1, ..., 23}
- Missing values are handled appropriately
- Time series A, S, and H are properly aligned
- Preprocessing can be validated on a single subject file

**Definition of Done**:
- Function `load_subject_data(uuid)` successfully loads CSV
- Function `create_variable_A(df)` produces discrete series in range [0, 4]
- Function `create_variable_S(df)` produces binary series in range [0, 1]
- Function `create_variable_H(df)` produces integer series in range [0, 23]
- Function `align_and_clean(A, S, H)` returns aligned series without NaN values
- Validation test on sample subject confirms correct value ranges

---

### User Story 2 (P2): Transfer Entropy Analysis 🎯 MVP Core

**As a** researcher
**I want to** calculate transfer entropy TE(A→S) and TE(S→A) with optimized history lengths
**So that** I can quantify the directional predictive relationship between activity and sitting

**Acceptance Criteria**:
- History length k_A is optimized by maximizing AIS for variable A
- History length k_S is optimized by maximizing AIS for variable S
- TE(A→S) is calculated using JIDT's discrete calculator with optimized k values
- TE(S→A) is calculated using JIDT's discrete calculator with optimized k values
- Statistical significance is determined using permutation testing (≥1000 surrogates)
- ΔTE = TE(A→S) - TE(S→A) is computed for each subject
- Results include: k_A, k_S, TE(A→S), TE(S→A), p_AS, p_SA, ΔTE

**Definition of Done**:
- Function `optimize_k_AIS(series)` returns optimal k value
- Function `calculate_TE(source, target, k_source, k_target)` returns TE value and p-value
- JVM lifecycle is properly managed (start → calculations → stop)
- NumPy arrays are correctly converted to Java int[] arrays
- Validation test on sample subject produces valid TE values and p-values
- Error handling prevents JVM crashes

---

### User Story 3 (P3): Conditional Transfer Entropy Robustness Check

**As a** researcher
**I want to** calculate conditional transfer entropy controlling for hour-of-day
**So that** I can verify that the observed asymmetry is not confounded by circadian patterns

**Acceptance Criteria**:
- CTE(A→S|H) is calculated using JIDT's conditional discrete calculator
- CTE(S→A|H) is calculated using JIDT's conditional discrete calculator
- Statistical significance is determined using permutation testing (≥1000 surrogates)
- ΔCTE = CTE(A→S|H) - CTE(S→A|H) is computed for each subject
- Results include: CTE_AS, CTE_SA, p_CTE_AS, p_CTE_SA, ΔCTE

**Definition of Done**:
- Function `calculate_CTE(source, target, condition, k_source, k_target)` returns CTE value and p-value
- Validation test on sample subject produces valid CTE values and p-values
- Error handling prevents calculation failures
- Results extend existing subject result dictionary with CTE fields

---

### User Story 4 (P4): Batch Processing and Aggregation

**As a** researcher
**I want to** process all subjects in the ExtraSensory dataset in batch
**So that** I can obtain population-level results for statistical testing

**Acceptance Criteria**:
- All subject UUIDs are discovered by scanning data directory for `*.features_labels.csv`
- Each subject is processed sequentially with progress tracking (tqdm)
- Per-subject errors are caught and logged without halting the batch
- Results from all successfully processed subjects are aggregated into a single DataFrame
- Aggregated results are saved to `results/extrasensory_te_results.csv`
- Processing summary is logged (total subjects, successful, failed, mean ΔTE)

**Definition of Done**:
- Function in `main.py` discovers all subject files
- Loop processes all subjects with try-except per subject
- Progress bar displays current subject and percentage complete
- Failed subjects are logged with error messages
- CSV output contains all expected columns: UUID, k_A, k_S, TE_AS, TE_SA, p_AS, p_SA, Delta_TE, CTE_AS, CTE_SA, p_CTE_AS, p_CTE_SA, Delta_CTE
- JVM is properly cleaned up after batch processing

---

### User Story 5 (P5): Statistical Analysis and Reporting

**As a** researcher
**I want to** perform statistical tests on the aggregated results and generate visualizations
**So that** I can draw conclusions about hypothesis H₁ and present findings

**Acceptance Criteria**:
- Results CSV is loaded into Jupyter Notebook
- Descriptive statistics are computed (mean, median, std of ΔTE and ΔCTE)
- Wilcoxon signed-rank test is performed on ΔTE distribution to test H₁: E[ΔTE] > 0
- Wilcoxon signed-rank test is performed on ΔCTE distribution for robustness verification
- Visualization 1: Distribution of ΔTE across subjects (histogram or violin plot)
- Visualization 2: Paired comparison of TE(A→S) vs TE(S→A) (scatter plot with identity line)
- Visualization 3: Comparison of TE vs CTE distributions (robustness check)
- Conclusion section interprets Wilcoxon results and states finding on H₁

**Definition of Done**:
- Notebook cell loads CSV successfully
- Descriptive statistics table is displayed
- Wilcoxon test for ΔTE is executed with test statistic and p-value reported
- Wilcoxon test for ΔCTE is executed with test statistic and p-value reported
- All three visualizations are rendered correctly
- Conclusion section provides clear interpretation of results
- Notebook can be re-run from top to bottom without errors

---

## Non-Functional Requirements

### Research Fidelity
- **MUST**: Follow methodology exactly as specified in Proposal_WeixuanKong.pdf
- **MUST**: Use JIDT discrete calculators (no continuous estimators)
- **MUST**: Use AIS-based k-optimization (not fixed k values)
- **MUST**: Use permutation testing for significance (≥1000 surrogates)
- **MUST**: Report both TE and CTE results for robustness

### Code Quality
- **MUST**: Adhere to SOLID principles (single responsibility, clear interfaces)
- **MUST**: Use modular structure (src/settings.py, src/preprocessing.py, src/analysis.py)
- **MUST**: Include docstrings for all functions
- **MUST**: Use clear variable names and comments

### Error Handling
- **MUST**: Implement robust per-subject error handling (try-except in batch loop)
- **MUST**: Log all errors with subject UUID and error message
- **MUST**: Ensure batch processing continues despite individual subject failures
- **MUST**: Handle JIDT exceptions gracefully (invalid input, calculation failures)

### Performance
- **SHOULD**: Display progress bar during batch processing
- **SHOULD**: Log processing time per subject for performance monitoring
- **MAY**: Implement caching for preprocessed variables (future optimization)

### Documentation
- **MUST**: Include README.md with installation and usage instructions
- **MUST**: Include requirements.txt with pinned dependency versions
- **MUST**: Include code comments explaining JIDT integration
- **SHOULD**: Include docstrings with parameter types and return types

## Out of Scope

The following are explicitly **NOT** included in this implementation:

- ❌ Parallel processing of subjects (sequential only in MVP)
- ❌ Real-time analysis or streaming data processing
- ❌ Interactive dashboard or web interface
- ❌ Database storage (CSV output only)
- ❌ Automated hyperparameter tuning beyond AIS optimization
- ❌ Alternative information-theoretic measures (e.g., mutual information, Granger causality)
- ❌ Cross-subject analysis or clustering
- ❌ Longitudinal analysis or temporal dynamics beyond TE
- ❌ Feature engineering beyond the specified variables A, S, H

## Dependencies

### Required
- Python 3.8+
- JIDT library (infodynamics.jar) at project root
- Java 8+ (for JPype/JIDT)
- ExtraSensory dataset CSV files in `data/` directory

### Python Libraries
- pandas, numpy, scipy, scikit-learn (data processing)
- jpype1 (Java integration)
- tqdm (progress tracking)
- jupyter, matplotlib, seaborn (reporting)

## Assumptions

1. ExtraSensory CSV files follow standard format with required columns
2. JIDT jar file is version-compatible with JPype
3. Sufficient memory is available for JIDT calculations (~2-4GB recommended)
4. Subject CSV files contain sufficient non-missing data for analysis
5. Python environment has Java properly configured for JPype

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| JIDT calculation failures | Medium | High | Robust error handling per subject |
| Missing data in subject CSVs | High | Medium | Preprocessing validation and NaN handling |
| Memory issues with large datasets | Low | High | Sequential processing, not parallel |
| Java/JPype configuration issues | Medium | High | Clear installation instructions in README |
| Long computation time | High | Low | Progress tracking with tqdm |

## Success Metrics

1. **Completeness**: ≥90% of subjects successfully processed
2. **Validity**: All TE/CTE values are non-negative and p-values in [0, 1]
3. **Statistical Power**: Sufficient sample size for Wilcoxon test (N ≥ 30 subjects)
4. **Reproducibility**: Report notebook can be re-run to reproduce results
5. **Code Quality**: No code smells, clear module boundaries, comprehensive docstrings

