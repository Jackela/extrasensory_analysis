Archived configs used during methodological evolution.

Archived notice: These configs are not maintained and are preserved only for historical reference.

Why archived:
- validation_N6_k4_1k.yaml: Phase 1/1b validation config to compare stratified CTE vs True CTE at fixed k=4. This phase showed the stratified (Fisher-merged) method is unreliable (opposite to True CTE), so it was deprecated.
- diagnostic_k_qc.yaml: Phase 4 diagnostic config to extract k-selection and hour-bin sample statistics (QC). Diagnostics show 73% of users (44/60) select k=6 under AIS, supporting the pivot to True CTE.

What to use now:
- Use `config/presets/production_k6_true_cte.yaml` for final runs (FINAL RUN CONFIG), aligned with `analysis/out/FINAL_RUN_k60_COMPLETE`.

Notes:
- Archived configs remain to reproduce scenarios and decision history; do not use them for production runs.
