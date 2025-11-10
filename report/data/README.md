Report Data Package
===================

This folder contains the curated CSVs used to generate the final report figures and tables. For every file, we document its provenance (source) and purpose (how it is used in the report).

Files
-----

- final_results_summary_n60.csv
  - Purpose: Final N=60 core run summary. Used to report group-level ΔTE mean and significance in the main text.
  - Provenance: Produced by the core run using `config/presets/production_k6_true_cte.yaml`.
  - Source path example: `analysis/out/FINAL_RUN_k60_COMPLETE/final_results_summary_n60.csv`.

- per_user_true_cte.csv
  - Purpose: Per-user True CTE results. Used to build the N=60 forest plot (Fig. 2) at `tau == 1`, showing per-user ΔTE and 95% CI, and to compute the random-effects summary (μ̂ and I²).
  - Provenance: Produced by the core run using `config/presets/production_k6_true_cte.yaml`.
  - Source path example: `analysis/out/FINAL_RUN_k60_COMPLETE/per_user_true_cte.csv`.

- k_selected_by_user_ALL.csv
  - Purpose: Diagnostics for adaptive k-selection across the N=60 users. Used to justify the choice `k = 6` via a distribution plot (Fig. 1), where the `k=6` bar is dominant (N=44).
  - Provenance: Produced by the diagnostic pipeline using `config/presets/diagnostic_k_qc.yaml`.
  - Source path example: `analysis/out/production_k6_true_cte_merged/k_selected_by_user_ALL.csv`.

- sensitivity_12cell_matrix.csv
  - Purpose: 12-cell sensitivity matrix (A_bins ∈ {3,5,7} × S_mode ∈ {binary, quantile3} × H_bin_hours ∈ {2,4}). Used to argue robustness of conclusions across discretization schemes (Fig. 3).
  - Provenance: Aggregated from the sensitivity grid script `analysis/scripts/run_sensitivity_grid.py` (N=10 subset) with a global AIS cap k ≤ 4 for stability.
  - Source path example: `analysis/out/sensitivity/<run_id>/*/summary.csv` (then aggregated into this matrix).

Appendix (Raw Sensitivity Summaries)
------------------------------------

Folder: `appendix_sensitivity_raw/`

- Contents: 12 raw combo-level `summary.csv` exports, renamed as `<combo>_summary.csv`, where `<combo>` ∈ {
  A3_Sbinary_H2h, A3_Sbinary_H4h, A3_Squantile3_H2h, A3_Squantile3_H4h,
  A5_Sbinary_H2h, A5_Sbinary_H4h, A5_Squantile3_H2h, A5_Squantile3_H4h,
  A7_Sbinary_H2h, A7_Sbinary_H4h, A7_Squantile3_H2h, A7_Squantile3_H4h }.
- Example source: `analysis/out/sensitivity/<run_id>/<combo>/summary.csv`.

Provenance & Reproduction
-------------------------

- Core run (N=60):
  - Config: `config/presets/production_k6_true_cte.yaml`
  - Command: `python run_production.py --config config/presets/production_k6_true_cte.yaml --workers 6`

- Diagnostics (k-selection):
  - Config: `config/presets/diagnostic_k_qc.yaml`
  - Output: `k_selected_by_user_ALL.csv` (used for Fig. 1)

- Sensitivity analysis (N=10; 12 cells):
  - Script: `analysis/scripts/run_sensitivity_grid.py` (global AIS cap k ∈ [1..4])
  - Aggregation helper: `analysis/scripts/sensitivity_report.py`

Notes
-----

- True CTE is computed via JIDT `ConditionalTransferEntropyCalculatorDiscrete` with 4-arg initialise:
  `initialise(base=max(base_A,base_S), history=k_S, numOtherInfoContributors=1, base_others=base_H)`.
- Across all 12 sensitivity cells, mean ΔCTE_true is negative, supporting robustness of the directional conclusion. Binary S-mode cells are highly significant; quantile3 cells are consistently negative with marginal p-values.
