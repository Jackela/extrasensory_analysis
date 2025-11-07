# ExtraSensory Causal Information Analysis (Final, English Only)

This repository provides an information-theoretic causal analysis pipeline for the ExtraSensory dataset, built on JIDT. Docs are now English-only and aligned with the final implementation pivot: True Conditional Transfer Entropy (True CTE) is the basis for conclusions; Global TE is computed opportunistically and recorded as NaN when high-k causes OOM.

## Final Conclusion (N=60)

- Hypothesis falsified: We reject the original hypothesis H1: E[ΔTE] > 0 (i.e., A→S stronger than S→A) on the full N=60 dataset [cite: 1-41].
- Using True CTE at τ=1, the final mean ΔTE (bits) is -0.033426 [cite: 13-14].
- Interpretation: After conditioning on hour-of-day (H), information flow from sitting to activity (S→A) is, on average, greater than the flow from activity to sitting (A→S).
- A→S signal remains real and prevalent: at τ=1, 55.9% (33/60) of users show statistically significant A→S information flow after FDR correction (q < 0.05) [cite: 13-14].

## Key Context (5 concise points)

1) Original plan: Global TE + stratified CTE (per-hour TE + Fisher p-merge).
2) Validation showed stratified CTE is methodologically unreliable at k=4, so it is deprecated.
3) Resource reality: when k≥5, Global TE often OOMs with 8–12GB JVM heap due to state space explosion.
4) Diagnostics: AIS-based k-selection shows 73% of users select k=6 (44/60) — see evidence file below.
5) Final pivot: adopt True CTE as the core method; accept Global TE OOM at high k and record NaN.

## Results and Evidence

- Final outputs: `analysis/out/FINAL_RUN_k60_COMPLETE`
  - Includes `per_user_true_cte.csv` (primary), `per_user_te.csv` (NaN for OOM at high k), `run_info.yaml`, `k_selected_by_user.csv`.
- Pivot evidence: `analysis/out/production_k6_true_cte_merged/k_selected_by_user_ALL.csv`
  - Shows 44/60 users selecting k=6 with AIS, supporting the True CTE pivot and acceptance of TE OOM at high k.

- Sensitivity analysis (N=10; 12-cell grid): `analysis/out/sensitivity/20251107_1146/summary.csv`
  - Evaluates discretization choices across A_bins ∈ {3,5,7}, S_mode ∈ {binary, quantile3}, H bins ∈ {6,12}, with a global AIS cap k ≤ 4 for stability.
  - Outcome: all 12 combinations produce mean ΔCTE_true < 0; binary S-mode is strongly significant; quantile3 S-mode is consistently negative with marginal/non-significant p-values (~0.09–0.13). Directional conclusion is robust across all schemes under k ≤ 4.

## Reproduction (Final Run Config)

- Recommended config: `config/presets/production_k6_true_cte.yaml` (marked as FINAL RUN CONFIG).
  - Runs Global TE and True CTE; TE may OOM at k≥5 and is recorded as NaN; True CTE produces the conclusions.

Examples:
```bash
# Direct (single-process)
python run_production.py --config config/presets/production_k6_true_cte.yaml

# Sharded execution (0-based shard/total)
python run_production.py --config config/presets/production_k6_true_cte.yaml --shard 0/4
# Or use helpers: run_parallel_4shards.sh / run_parallel_4shards.bat
```

Environment and resources:
- Python 3.12+, Java 8+, JIDT available (`jidt/infodynamics.jar`)
- JVM heap per process: 8–12GB; parallel processes require linear aggregate RAM

## Exact Implementation

- Features
  - Composite mode: includes SMA and tri-axis variance with hour-of-day conditioning for CTE.
  - Alternatives (configurable): `sma_only`, `variance_only`, `magnitude_only`.

- K-selection (AIS)
  - AIS(k) = I(X_t; X_{t-k:t-1}), select k = argmax_k AIS(k) over grid [1..6].
  - Strategies: `AIS` (unbounded), `GUARDED_AIS` (e.g., k_max=4 + undersampling guard), `FIXED`.

- Global TE (unconditional)
  - JIDT `TransferEntropyCalculatorDiscrete` via 0-arg constructor + 6-arg `initialise(base, k_dest, 1, k_source, 1, delay)`.
  - Delay equals `tau`; histories use consecutive lag (`k_tau=1`).
  - Surrogates: fixed `surrogates` or staged `adaptive_stages` per config.
  - If OOM at high k, TE value recorded as NaN by design; pipeline continues.

- True Conditional TE (core)
  - JIDT `ConditionalTransferEntropyCalculatorDiscrete` with 4-arg initialise signature:
    - `initialise(base=max(base_A, base_S), history=k_S, numOtherInfoContributors=1, base_others=base_H)`; a single conditional variable (hour-of-day bin).
  - For `tau>1`, inputs are data-lagged prior to passing into JIDT: `source[:-tau]`, `dest[tau:]`, `cond[tau:]`.
  - Observations added as Java `int[]`; computes `computeAverageLocalOfObservations()`.
  - Significance: fixed surrogates or staged `adaptive_stages`; last stage p-value is returned.

- Statistical testing
  - FDR: Benjamini–Hochberg per (family, tau). Families: `TE`, `CTE`, `STE`, `GC`. Alpha=0.05.
  - Outputs include raw p-values and FDR-corrected q-values.

- Performance notes
  - State space at k=6: 5^6 × 2^6 ≈ 1e6 states; TE runtime jumps from seconds (k=4) to minutes (k=6).
  - Parallel sharding across users strongly recommended for k=6.

## Project Structure

```
extrasensory_analysis/
├── config/
│   ├── template.yaml           # Parameter reference
│   ├── presets/                # Preset profiles
│   ├── README.md               # Config docs (English only)
│   └── MIGRATION_NOTES.md      # Legacy-to-config migration
├── src/
│   ├── analysis.py             # TE/CTE/STE/GC pipeline
│   ├── preprocessing.py        # Data loading + features
│   ├── k_selection.py          # AIS k-selection
│   ├── fdr_utils.py            # FDR utilities
│   ├── granger_analysis.py     # VAR Granger causality
│   ├── symbolic_te.py          # Symbolic TE
│   ├── jidt_adapter.py         # JIDT bridge (TE, True CTE)
│   └── settings.py             # Legacy constants
├── tools/                      # Validators and diagnostics
├── tests/                      # Unit tests
├── run_production.py           # Main entrypoint (supports --shard)
├── run_parallel_4shards.*      # Parallel helpers
├── merge_shard_results.py      # Merge outputs from shards
└── docs/                       # Methods and specs
```

## Troubleshooting

- TE OOM at high k: expected; values recorded as NaN; True CTE drives conclusions.
- Long runtime at k=6: use 4-process sharding and ensure sufficient RAM.
- JIDT not found: ensure `jidt/infodynamics.jar` is present and Java 8+ is installed.

**JVM Out of Memory (k=6)**:
```
Error: Requested memory for base 5, k=6, l=6 is too large
Solution: Use k6_full preset (includes 12GB heap) or set xmx=12g in custom config
```

**Process Crashes**:
```bash
# Resume failed shard
python run_production.py --full --shard 2/4 \
  --resume analysis/out/full_bins6_20251026_1432
```

**Slow Execution**:
- k=6 is expected to be 760x slower than k=4 (state space explosion)
- Use parallel execution to mitigate: 4 processes reduce 240h → 60h

## Documentation

- **[config/README.md](config/README.md)** - Configuration reference
- **[PARALLEL_EXECUTION.md](PARALLEL_EXECUTION.md)** - Parallel execution guide
- **[EXECUTION_PLAN.md](EXECUTION_PLAN.md)** - Performance analysis
- **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - Implementation status
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines
- **[CHANGELOG.md](CHANGELOG.md)** - Version history

## Citation

If you use this code in your research, please cite:

```bibtex
@software{extrasensory_te_analysis,
  title = {ExtraSensory Transfer Entropy Analysis},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/extrasensory_analysis}
}
```

**ExtraSensory Dataset**:
```bibtex
@article{vaizman2017recognizing,
  title={Recognizing Detailed Human Context in the Wild from Smartphones and Smartwatches},
  author={Vaizman, Yonatan and Ellis, Katherine and Lanckriet, Gert},
  journal={IEEE Pervasive Computing},
  volume={16},
  number={4},
  pages={62--74},
  year={2017}
}
```

**JIDT**:
```bibtex
@article{lizier2014jidt,
  title={JIDT: An information-theoretic toolkit for studying the dynamics of complex systems},
  author={Lizier, Joseph T},
  journal={Frontiers in Robotics and AI},
  volume={1},
  pages={11},
  year={2014}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Code style guidelines (PEP 8, Black formatting)
- Testing standards
- Pull request process

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/extrasensory_analysis/issues)
- **Questions**: See documentation or open a discussion

## Acknowledgments

- **ExtraSensory Dataset**: Yonatan Vaizman, Katherine Ellis, Gert Lanckriet (UCSD)
- **JIDT**: Joseph T. Lizier (University of Sydney)
- **Python Scientific Stack**: NumPy, Pandas, SciPy, scikit-learn, statsmodels
## Limitations and Future Work

- Permutation testing: This study uses JIDT's standard surrogate testing (the default permutation tester) for significance assessment. While widely adopted, these surrogates do not fully preserve the temporal autocorrelation structure of time series. Future work should consider block permutation (block bootstrap style surrogates) or phase-randomized surrogates to better respect serial dependence when evaluating TE/CTE significance.
