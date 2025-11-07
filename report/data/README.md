Report Data Package
===================

This folder contains the curated CSVs used in the final report and appendix. Each file’s provenance and purpose are listed below.

Files
-----

- final_results_summary_n60.csv
  - Purpose: Core N=60 conclusion table (aggregate True CTE results).
  - Source: analysis/out/FINAL_RUN_k60_COMPLETE/final_results_summary_n60.csv

- per_user_true_cte.csv
  - Purpose: Per-user True CTE results (used for forest plots and user-level summaries).
  - Source: analysis/out/FINAL_RUN_k60_COMPLETE/per_user_true_cte.csv

- k_selected_by_user_ALL.csv
  - Purpose: AIS k-selection diagnostics across all 60 users (evidence that k=6 is widely selected).
  - Source: analysis/diagnostics/k_selected_by_user_ALL.csv

- sensitivity_12cell_matrix.csv
  - Purpose: 12-cell sensitivity matrix (A_bins ∈ {3,5,7} × S_mode ∈ {binary, quantile3} × H_bin_hours ∈ {2,4}).
  - Notes: Built under a global AIS cap k ≤ 4 to ensure computational stability and avoid JIDT exceptions during surrogate tests.
  - Source run: analysis/out/sensitivity/20251107_1146/*/summary.csv (aggregated)

Appendix (Raw Sensitivity Summaries)
------------------------------------

Folder: appendix_sensitivity_raw/

- Contents: 12 raw combo-level summary.csv exports, renamed as <combo>_summary.csv, where <combo> ∈ {
  A3_Sbinary_H2h, A3_Sbinary_H4h, A3_Squantile3_H2h, A3_Squantile3_H4h,
  A5_Sbinary_H2h, A5_Sbinary_H4h, A5_Squantile3_H2h, A5_Squantile3_H4h,
  A7_Sbinary_H2h, A7_Sbinary_H4h, A7_Squantile3_H2h, A7_Squantile3_H4h }.
- Source: analysis/out/sensitivity/20251107_1146/<combo>/summary.csv

Provenance & Reproduction
-------------------------

- Final run (N=60):
  - Config: config/presets/production_k6_true_cte.yaml
  - Command: python run_production.py --config config/presets/production_k6_true_cte.yaml --workers 6

- Sensitivity analysis (N=10; 12 cells):
  - Script: analysis/scripts/run_sensitivity_grid.py (global AIS cap k ∈ [1..4])
  - Latest run used here: analysis/out/sensitivity/20251107_1146
  - Aggregate table helper: analysis/scripts/sensitivity_report.py

Notes
-----

- True CTE is computed via JIDT ConditionalTransferEntropyCalculatorDiscrete with 4-arg initialise:
  initialise(base=max(base_A,base_S), history=k_S, numOtherInfoContributors=1, base_others=base_H).
- Under k ≤ 4, all 12 sensitivity cells show mean ΔCTE_true < 0 (robust directional conclusion). Binary S-mode shows strong significance; quantile3 is consistently negative with marginal p-values.

