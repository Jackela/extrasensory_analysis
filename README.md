# ExtraSensory Information Transfer Analysis

This repository contains a JIDT-based information-transfer analysis pipeline and historical result exports for the ExtraSensory dataset. Documentation and maintenance are handled by AI; source code, configuration and committed data remain the reference for each claim.

## Committed results (N=60)

The current result package is [report/data](report/data/README.md). Its CSV exports contain 60 distinct users at each of τ=1 and τ=2. At τ=1, mean ΔTrue CTE is **−0.033426 bits**, and **33/60 users (55.0%)** have an A→S adjusted p-value below 0.05. The k-selection diagnostic gives k=6 for 44 distinct users. These are descriptive checks of the committed exports, with file hashes and calculation fields in [manifest.json](report/data/manifest.json).

The primary file is `per_user_true_cte.csv`. The legacy `final_results_summary_n60.csv` uses TE-labelled columns for the same True CTE values; the verification command checks their numerical agreement. The raw run directories previously named `analysis/out/FINAL_RUN_k60_COMPLETE` and `analysis/out/production_k6_true_cte_merged` are historical producer paths and are not included in this checkout. The committed exports support the numbers above; they do not establish that the full pipeline has been rerun.

The [12-cell sensitivity export](report/data/sensitivity_12cell_matrix.csv) and its raw summary appendix are also included. Its grid uses A_bins ∈ {3,5,7}, S_mode ∈ {binary, quantile3}, and H_bin_hours ∈ {2,4}, with N=10 and k≤4. All 12 recorded mean differences are negative. The analysis remains conditional on discretization, sample selection and the surrogate testing method described below.

```bash
python tools/verify_report_data.py
python -m unittest discover -s tests -p 'test_report_evidence.py'
```

The core method is True Conditional Transfer Entropy through JIDT, with hour-of-day as the conditioning variable. Global TE may exhaust the JVM heap at high k and is recorded as NaN. Reproduction requires the original dataset and JIDT environment.

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
  author = {{extrasensory_analysis contributors}},
  year = {2025},
  url = {https://github.com/Jackela/extrasensory_analysis}
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

- **Issues**: [GitHub Issues](https://github.com/Jackela/extrasensory_analysis/issues)
- **Questions**: See documentation or open a discussion

## Acknowledgments

- **ExtraSensory Dataset**: Yonatan Vaizman, Katherine Ellis, Gert Lanckriet (UCSD)
- **JIDT**: Joseph T. Lizier (University of Sydney)
- **Python Scientific Stack**: NumPy, Pandas, SciPy, scikit-learn, statsmodels
## Limitations and Future Work

- Permutation testing: This study uses JIDT's standard surrogate testing (the default permutation tester) for significance assessment. While widely adopted, these surrogates do not fully preserve the temporal autocorrelation structure of time series. Future work should consider block permutation (block bootstrap style surrogates) or phase-randomized surrogates to better respect serial dependence when evaluating TE/CTE significance.
