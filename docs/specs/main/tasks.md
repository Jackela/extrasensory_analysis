---
description: "Task list for ExtraSensory Transfer Entropy Analysis"
---

# Tasks: ExtraSensory Transfer Entropy Analysis

**Input**: Design documents from specification and plan
**Prerequisites**: Proposal_WeixuanKong.pdf (research methodology), infodynamics.jar (JIDT library)

**Tests**: No explicit test tasks included per specification. JIDT provides built-in validation via permutation testing.

**Organization**: Tasks are grouped by analysis phase to enable modular implementation and verification.

## Format: `[ID] [P?] [Story] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which analysis phase this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions
- Single project structure: `src/`, results at repository root
- Modular Python architecture per implementation plan

---

## Phase 1: Setup (Project Infrastructure)

**Purpose**: Project initialization, dependencies, and basic structure

- [x] T001 Create project directory structure (src/, results/, data/)
- [x] T002 Initialize Python virtual environment and install core dependencies (pandas, numpy, scipy, scikit-learn, jpype1)
- [x] T003 [P] Verify JIDT library (infodynamics.jar) is accessible at project root
- [x] T004 [P] Create src/settings.py with configuration constants (file paths, parameters, column names)
- [x] T005 [P] Create placeholder modules: src/preprocessing.py, src/analysis.py, main.py

**Checkpoint**: Project structure ready, dependencies installed, JIDT accessible

---

## Phase 2: Foundational (Core Infrastructure)

**Purpose**: Core utilities and JIDT integration that ALL analysis phases depend on

**⚠️ CRITICAL**: No analysis work can begin until this phase is complete

- [x] T006 Implement JVM lifecycle management in src/analysis.py (start_jvm, stop_jvm functions using jpype)
- [x] T007 [P] Implement Java array conversion utilities in src/analysis.py (numpy array → Java int[] conversion)
- [x] T008 [P] Create error handling framework for JIDT calls in src/analysis.py (try-except wrappers, logging)
- [x] T009 Implement data loading function in src/preprocessing.py (load subject CSV, extract timestamp, labels, features)
- [x] T010 [US1] Implement variable S creation in src/preprocessing.py (extract label:SITTING as binary series)

**Checkpoint**: JIDT integration working, data loading functional, ready for variable preprocessing

---

## Phase 3: User Story 1 - Variable Preprocessing (Priority: P1) 🎯 MVP

**Goal**: Transform raw ExtraSensory data into analysis-ready discrete time series (A, S, H) for each subject

**Independent Test**: Successfully preprocess one subject file and verify A is discrete {0,1,2,3,4}, S is binary {0,1}, H is integer {0-23}

### Implementation for User Story 1

- [x] T011 [US1] Implement variable A preprocessing in src/preprocessing.py (z-score normalization within-subject for raw_acc:magnitude_stats:mean)
- [x] T012 [US1] Implement quantile discretization for variable A in src/preprocessing.py (5-bin quantile → discrete {0,1,2,3,4})
- [x] T013 [US1] Implement variable H extraction in src/preprocessing.py (extract hour-of-day from timestamp → integer {0-23})
- [x] T014 [US1] Implement time series alignment and cleaning in src/preprocessing.py (handle missing values, align A/S/H series)
- [x] T015 [US1] Add preprocessing validation function in src/preprocessing.py (verify A/S/H ranges, check for NaN handling)
- [x] T016 [US1] Add preprocessing logging in src/preprocessing.py (log per-subject statistics, missing value counts)

**Checkpoint**: Preprocessing pipeline complete - can convert raw subject CSVs to clean A/S/H time series

---

## Phase 4: User Story 2 - AIS Optimization & TE Calculation (Priority: P2)

**Goal**: Optimize history length parameters (k_A, k_S) and calculate transfer entropy TE(A→S) and TE(S→A) using JIDT

**Independent Test**: Successfully run AIS optimization and TE calculation on one subject, obtain k values and TE values with significance

### Implementation for User Story 2

- [x] T017 [US2] Implement AIS optimization for k_A in src/analysis.py (use ActiveInformationCalculatorDiscrete, maximize AIS for variable A)
- [x] T018 [US2] Implement AIS optimization for k_S in src/analysis.py (use ActiveInformationCalculatorDiscrete, maximize AIS for variable S)
- [x] T019 [US2] Implement TE(A→S) calculation in src/analysis.py (use TransferEntropyCalculatorDiscrete with optimized k_A, k_S)
- [x] T020 [US2] Implement TE(S→A) calculation in src/analysis.py (use TransferEntropyCalculatorDiscrete with optimized k_A, k_S)
- [x] T021 [US2] Implement TE significance testing in src/analysis.py (use TransferEntropySignificanceCalculatorDiscrete, ≥1000 surrogates)
- [x] T022 [US2] Implement Delta_TE computation in src/analysis.py (ΔTE = TE(A→S) - TE(S→A))
- [x] T023 [US2] Add per-subject result collection in src/analysis.py (return dict with k_A, k_S, TE_AS, TE_SA, p_AS, p_SA, Delta_TE)
- [x] T024 [US2] Add TE calculation logging in src/analysis.py (log optimization results, TE values, p-values per subject)

**Checkpoint**: Core TE analysis pipeline complete - can calculate TE and significance for individual subjects

---

## Phase 5: User Story 3 - Conditional TE Robustness Check (Priority: P3)

**Goal**: Calculate conditional transfer entropy CTE(A→S|H) and CTE(S→A|H) to verify robustness when conditioning on hour-of-day

**Independent Test**: Successfully run CTE calculation on one subject, obtain CTE values with significance and Delta_CTE

### Implementation for User Story 3

- [x] T025 [US3] Implement CTE(A→S|H) calculation in src/analysis.py (use ConditionalTransferEntropyCalculatorDiscrete, condition on H)
- [x] T026 [US3] Implement CTE(S→A|H) calculation in src/analysis.py (use ConditionalTransferEntropyCalculatorDiscrete, condition on H)
- [x] T027 [US3] Implement CTE significance testing in src/analysis.py (use ConditionalTransferEntropySignificanceCalculatorDiscrete, ≥1000 surrogates)
- [x] T028 [US3] Implement Delta_CTE computation in src/analysis.py (ΔCTE = CTE(A→S|H) - CTE(S→A|H))
- [x] T029 [US3] Extend per-subject result collection in src/analysis.py (add CTE_AS, CTE_SA, p_CTE_AS, p_CTE_SA, Delta_CTE to result dict)
- [x] T030 [US3] Add CTE calculation logging in src/analysis.py (log CTE values, p-values per subject)

**Checkpoint**: Full analysis pipeline complete including robustness check - ready for batch processing

---

## Phase 6: User Story 4 - Batch Processing & Aggregation (Priority: P4)

**Goal**: Process all subjects in the dataset, aggregate results, and save to CSV for final statistical analysis

**Independent Test**: Successfully process all subjects with robust error handling, generate aggregated results CSV with no crashes

### Implementation for User Story 4

- [x] T031 [US4] Implement subject UUID discovery in main.py (scan data/ directory for *.features_labels.csv files)
- [x] T032 [US4] Implement per-subject processing loop in main.py (call preprocessing → analysis for each subject)
- [x] T033 [US4] Implement robust per-subject error handling in main.py (try-except per subject, log failures, continue batch)
- [x] T034 [US4] Add progress tracking in main.py (use tqdm for progress bar over subjects)
- [x] T035 [US4] Implement result aggregation in main.py (collect all subject results into pandas DataFrame)
- [x] T036 [US4] Implement CSV export in main.py (save DataFrame to results/extrasensory_te_results.csv)
- [x] T037 [US4] Add batch processing summary logging in main.py (total subjects, successful, failed, summary statistics)
- [x] T038 [US4] Add JVM cleanup in main.py (ensure stop_jvm called after all subjects processed)

**Checkpoint**: Batch processing complete - results CSV ready for statistical analysis

---

## Phase 7: User Story 5 - Statistical Analysis & Reporting (Priority: P5)

**Goal**: Perform final statistical tests (Wilcoxon signed-rank) on aggregated results and generate the final report notebook

**Independent Test**: Open report.ipynb and successfully run all cells to reproduce the statistical conclusions

### Implementation for User Story 5

- [x] T039 [US5] Create report.ipynb Jupyter notebook in project root
- [x] T040 [US5] Implement data loading section in report.ipynb (load results/extrasensory_te_results.csv)
- [x] T041 [US5] Implement descriptive statistics section in report.ipynb (summarize k values, TE values, CTE values across subjects)
- [x] T042 [US5] Implement Wilcoxon test for Delta_TE in report.ipynb (test H1: E[ΔTE] > 0 using scipy.stats.wilcoxon)
- [x] T043 [US5] Implement Wilcoxon test for Delta_CTE in report.ipynb (test robustness check using scipy.stats.wilcoxon)
- [x] T044 [P] [US5] Create Delta_TE distribution visualization in report.ipynb (histogram/violin plot of ΔTE across subjects)
- [x] T045 [P] [US5] Create paired TE comparison visualization in report.ipynb (scatter plot TE(A→S) vs TE(S→A) with identity line)
- [x] T046 [P] [US5] Create CTE robustness visualization in report.ipynb (compare TE vs CTE distributions)
- [x] T047 [US5] Implement conclusion section in report.ipynb (interpret Wilcoxon results, state final conclusion on H1)
- [x] T048 [US5] Add markdown narrative sections in report.ipynb (introduction, methodology, results interpretation)

**Checkpoint**: Final report complete with statistical analysis and visualizations

---

## Phase 8: Polish & Documentation

**Purpose**: Code quality improvements and documentation

- [x] T049 [P] Add module-level docstrings to src/settings.py, src/preprocessing.py, src/analysis.py
- [x] T050 [P] Add function-level docstrings to all functions in src/preprocessing.py
- [x] T051 [P] Add function-level docstrings to all functions in src/analysis.py
- [x] T052 [P] Create requirements.txt with all dependencies and versions
- [x] T053 [P] Create README.md with project overview, installation instructions, usage guide
- [x] T054 Code review: Verify SOLID principles adherence across all modules
- [x] T055 Code review: Verify error handling robustness per constitution
- [x] T056 Verify research fidelity: Cross-check implementation against Proposal_WeixuanKong.pdf methodology

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all analysis phases
- **User Story 1 (Phase 3)**: Depends on Foundational (T006-T010) completion
- **User Story 2 (Phase 4)**: Depends on User Story 1 (T011-T016) completion - needs preprocessed variables
- **User Story 3 (Phase 5)**: Depends on User Story 2 (T017-T024) completion - extends TE with CTE
- **User Story 4 (Phase 6)**: Depends on User Story 3 (T025-T030) completion - needs full analysis pipeline
- **User Story 5 (Phase 7)**: Depends on User Story 4 (T031-T038) completion - needs aggregated results CSV
- **Polish (Phase 8)**: Depends on all implementation phases being complete

### User Story Dependencies

- **User Story 1 (P1)**: Data preprocessing - foundation for all analysis
- **User Story 2 (P2)**: Core TE analysis - depends on US1 preprocessed variables
- **User Story 3 (P3)**: Robustness check - depends on US2 TE pipeline
- **User Story 4 (P4)**: Batch processing - depends on US3 full analysis pipeline
- **User Story 5 (P5)**: Reporting - depends on US4 aggregated results

### Within Each User Story

- Foundational utilities (JVM, conversions) before analysis functions
- Data loading before preprocessing
- Preprocessing (A, S, H) before analysis
- AIS optimization before TE calculation
- TE calculation before CTE calculation
- Analysis pipeline complete before batch processing
- Batch processing complete before statistical reporting

### Parallel Opportunities

- T003, T004, T005 in Phase 1 can run in parallel (different files)
- T007, T008 in Phase 2 can run in parallel (different utilities in same module)
- T044, T045, T046 in Phase 7 can run in parallel (independent visualizations)
- T049, T050, T051, T052, T053 in Phase 8 can run in parallel (different documentation tasks)

---

## Parallel Example: User Story 5 (Reporting Visualizations)

```bash
# Launch all visualization tasks for User Story 5 together:
Task: "Create Delta_TE distribution visualization in report.ipynb"
Task: "Create paired TE comparison visualization in report.ipynb"
Task: "Create CTE robustness visualization in report.ipynb"
```

---

## Implementation Strategy

### MVP First (Core TE Analysis - User Stories 1-2)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all analysis)
3. Complete Phase 3: User Story 1 (Preprocessing)
4. Complete Phase 4: User Story 2 (TE Calculation)
5. **STOP and VALIDATE**: Test on 1-2 subjects, verify TE values and significance
6. If validated, proceed to robustness check (US3) and batch processing (US4)

### Incremental Delivery

1. Complete Setup + Foundational → Infrastructure ready
2. Add User Story 1 → Test preprocessing on sample subjects
3. Add User Story 2 → Test TE calculation on sample subjects → **MVP: Core hypothesis test possible**
4. Add User Story 3 → Test CTE on sample subjects → Robustness verified
5. Add User Story 4 → Process all subjects → Full dataset analyzed
6. Add User Story 5 → Generate final report → **Complete deliverable**

### Sequential Execution (Single Developer)

1. Complete Setup (T001-T005)
2. Complete Foundational (T006-T010)
3. Complete US1 Preprocessing (T011-T016) → Validate on 1 subject
4. Complete US2 TE Analysis (T017-T024) → Validate on 1 subject
5. Complete US3 CTE Analysis (T025-T030) → Validate on 1 subject
6. Complete US4 Batch Processing (T031-T038) → Process all subjects
7. Complete US5 Reporting (T039-T048) → Final deliverable
8. Polish (T049-T056)

---

## Notes

- [P] tasks = different files or independent sections, no dependencies
- [Story] label maps task to specific analysis phase for traceability
- Each user story builds incrementally on previous phases
- JIDT provides built-in validation via permutation testing (no separate test files needed)
- Commit after each task or logical group (e.g., after each US checkpoint)
- Stop at any checkpoint to validate on sample subjects before proceeding
- Constitution priority: Research fidelity > Code quality > Performance
- Final deliverable: report.ipynb with statistical conclusions on hypothesis H1

---

## Summary

- **Total Tasks**: 56
- **Parallelizable Tasks**: 13 (marked with [P])
- **User Stories**: 5 analysis phases
  - US1 (Preprocessing): 6 tasks
  - US2 (TE Analysis): 8 tasks
  - US3 (CTE Robustness): 6 tasks
  - US4 (Batch Processing): 8 tasks
  - US5 (Reporting): 10 tasks
- **MVP Scope**: Phases 1-4 (Setup → Foundational → Preprocessing → TE Analysis)
- **Full Deliverable**: All phases including US5 (report.ipynb)

