# ExtraSensory Transfer Entropy Analysis

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Production-grade pipeline for information-theoretic analysis of bidirectional causality between physical activity and sedentary behavior using the ExtraSensory dataset and JIDT (Java Information Dynamics Toolkit).

## Overview

This project implements a comprehensive pipeline for computing:
- **Transfer Entropy (TE)**: Directed information flow Activity ↔ Sitting
- **Conditional Transfer Entropy (CTE)**: Hour-of-day stratified causality analysis
- **Symbolic Transfer Entropy (STE)**: Ordinal pattern-based causality
- **Granger Causality (GC)**: Linear baseline comparison

**Key Features**:
- ✅ Pure AIS (Active Information Storage) k-selection with optional computational guards
- ✅ Multi-process parallel execution for large-scale analysis (4x speedup)
- ✅ 6-bin hour-of-day stratification for CTE (4-hour windows)
- ✅ Benjamini-Hochberg FDR correction per (method, tau) family
- ✅ Checkpoint/resume capability for fault tolerance
- ✅ Comprehensive tracking (run_info.yaml, status.json, errors.log)
- ✅ YAML-based configuration with schema validation

## Quick Start

### Prerequisites
- Python 3.12+
- Java 8+ (for JIDT)
- **RAM**: 12GB minimum for k=6 analysis (48GB recommended for parallel execution)

### Installation

**Quick Setup**:
```bash
git clone https://github.com/yourusername/extrasensory_analysis.git
cd extrasensory_analysis
pip install -r requirements.txt
```

**Complete Installation Guide**: See [INSTALLATION.md](INSTALLATION.md) for detailed setup including:
- JIDT download and configuration
- ExtraSensory dataset acquisition
- Environment setup and verification

### Run Analysis

**All execution is preset-based with zero hardcoding.**

**Smoke Test (2 users, k=4 fixed, ~2 minutes)**:
```bash
python run_production.py smoke
```

**Full k=6 Analysis (60 users, 4-process parallel, 60 hours)**:
```bash
# Linux/macOS/Git Bash
./run_parallel_4shards.sh k6_full

# Windows
run_parallel_4shards.bat k6_full
```

**Fast k≤4 Analysis (60 users, 4 modes, ~4 hours)**:
```bash
python run_production.py k4_fast
```

**Custom Configuration**:
```bash
python run_production.py --config path/to/custom.yaml
```

**Resume from Checkpoint**:
```bash
python run_production.py k6_full --shard 2/4 \
  --resume analysis/out/k6_full_20251026_1432
```

**Merge Parallel Results**:
```bash
python merge_shard_results.py \
  --shards analysis/out/k6_full_20251026_143{0,1,2,3} \
  --output analysis/out/merged_k6_full
```

## Configuration Presets

**Four pre-configured analysis profiles** (zero hardcoding):

| Preset | Users | Modes | K | Runtime | Use Case |
|--------|-------|-------|---|---------|----------|
| **smoke** | 2 | 1 | k=4 (fixed) | 2 min | Fast validation |
| **k6_full** | 60 | 1 | k≤6 (pure AIS) | 60h (parallel) | Scientific optimal |
| **k4_fast** | 60 | 4 | k≤4 (guarded) | 4h | Fast comprehensive |
| **24bin_cte** | 60 | 1 | k≤4 (guarded) | 5h | High temporal resolution |

**Example Preset** (`config/presets/k6_full.yaml`):
```yaml
n_users: 60
feature_modes: [composite]
taus: [1, 2]

k_selection:
  strategy: "AIS"
  k_grid: [1, 2, 3, 4, 5, 6]
  k_max: null  # No constraints
  undersampling_guard: false

hour_bins: 6  # 4-hour windows
surrogates: 1000

jvm:
  xms: "6g"
  xmx: "12g"  # Validated for k=6
```

**Documentation**:
- [config/presets/README.md](config/presets/README.md) - Preset comparison and selection guide
- [config/README.md](config/README.md) - Full parameter reference
- [config/MIGRATION_NOTES.md](config/MIGRATION_NOTES.md) - Hardcoded → preset migration

## Parallel Execution

**Performance Comparison**:

| Configuration | Users/Process | Total Time | Speedup |
|---------------|---------------|------------|---------|
| Single-process | 60 | 240 hours (10 days) | 1x |
| **4-process parallel** | **15** | **60 hours (2.5 days)** | **4x** |

**User Sharding** (modulo partitioning):
```bash
--shard 0/4: Users [0,4,8,12,...,56] (15 users)
--shard 1/4: Users [1,5,9,13,...,57] (15 users)
--shard 2/4: Users [2,6,10,14,...,58] (15 users)
--shard 3/4: Users [3,7,11,15,...,59] (15 users)
```

**Memory Requirements**:
- 4 processes × 12GB heap = 48GB total
- Validated with smoke tests (8GB fails, 12GB succeeds)

**Execution Guides**:
- [PARALLEL_EXECUTION.md](PARALLEL_EXECUTION.md) - Detailed parallel execution guide
- [EXECUTION_PLAN.md](EXECUTION_PLAN.md) - Performance analysis and timeline
- [QUICK_START.md](QUICK_START.md) - Quick reference card

## Output Structure

**Per-Shard Output**:
```
analysis/out/full_bins6_YYYYMMDD_HHMM/
├── per_user_te.csv          # TE results (A→S, S→A)
├── per_user_cte.csv         # CTE results with 6-bin stratification
├── per_user_ste.csv         # STE results
├── per_user_gc.csv          # Granger causality results
├── k_selected_by_user.csv   # AIS k-selection tracking
├── hbin_counts.csv          # Hour bin sample distribution
├── run_info.yaml            # Metadata (JIDT, JVM, git commit, config)
├── status.json              # Real-time progress with ETA
└── errors.log               # Error tracking
```

**Merged Output** (after combining shards):
```
analysis/out/merged_k6_full/
├── per_user_te.csv          # 60 users × 2 taus = 120 rows
├── per_user_cte.csv         # 60 users × 2 taus = 120 rows
├── per_user_ste.csv         # 60 users × 2 taus = 120 rows
├── per_user_gc.csv          # 60 users = 60 rows
├── k_selected_by_user.csv   # 60 rows (k=6 selections)
├── hbin_counts.csv          # 60 rows × 24 hour columns
└── run_info.yaml            # Aggregated metadata
```

## Methodology

### Data Preprocessing

**NaN Handling Strategy**:
- **Sitting labels (S)**: `fillna(0)` - assumes NaN = not sitting
- **Activity features (A)**: `dropna()` - removes rows with missing accelerometer data
- **Quality control**: Reject users with <200 valid samples after cleaning

**Variable Creation**:

| Variable | Description | Processing | Output |
|----------|-------------|------------|--------|
| **A** | Activity Level | Composite feature → z-score → 5-bin quantile discretization | {0,1,2,3,4} |
| **S** | Sitting State | Binary label from `label:SITTING` | {0,1} |
| **H_raw** | Hour-of-Day | Extracted from timestamp | {0-23} |
| **H_binned** | Hour Bins | 6 bins (4-hour windows) | {0-5} |

### Feature Engineering

**Composite Feature** (current default):
```
Activity = 0.6 × SMA + 0.4 × Variance
```

Where:
- **SMA** (Signal Magnitude Area) = (|ax| + |ay| + |az|) / 3
- **Variance** = √(std_x² + std_y² + std_z²)

**Alternative Modes** (configurable):
- `sma_only`: SMA only
- `variance_only`: Tri-axis variance only
- `magnitude_only`: Raw magnitude mean (baseline)

### K-Selection via AIS

**Active Information Storage (AIS)**:
```
AIS(k) = I(X_t; X_{t-k:t-1})
k_selected = argmax_k AIS(k)
```

**Strategies**:

1. **AIS** (Pure, current):
   - No constraints, selects optimal k from grid [1-6]
   - Risk: Large k → state space explosion

2. **GUARDED_AIS** (Available):
   - Optional `k_max` hard cap
   - Optional `undersampling_guard` (≥25 samples/state)

3. **FIXED**:
   - Use fixed k for all users (fast, no optimization)

**State Space Analysis** (k=6):
```
A states: 5^6 = 15,625
S states: 2^6 = 64
Total: 5^6 × 2^6 = 1,000,000 states

Performance impact:
- k=4: 1.5s per TE computation
- k=6: 19 min per TE computation (760x slower, expected)
```

### Transfer Entropy Implementation

**TE (Unconditional)**:
```
TE(A→S, τ) = I(S_t; A_{t-τ:t-τ-k+1} | S_{t-1:t-l})
```

**CTE (Conditional on Hour-of-Day)**:
- Method: **STRATIFIED-CTE**
- Key: Data-level lag **before** stratification
- Bins: 6 (4-hour windows: 00-04, 04-08, 08-12, 12-16, 16-20, 20-24)

**Implementation**:
- JIDT v1.5 `DiscreteTE` with 6-arg initialization
- Permutation surrogates: 1000 (configurable)
- Direction: Both A→S and S→A computed

### Statistical Testing

**FDR Correction** (Benjamini-Hochberg):
- Grouping: Per (family, tau) for independence
- Families: TE, CTE, STE, GC
- Alpha: 0.05
- Output: Both p-values and q-values (FDR-corrected)

**Significance**:
- Individual: q < 0.05
- Directionality: Delta_TE with corrected p-value

## Performance Optimization

**Memory Optimization**:
- JVM heap: 12GB minimum for k=6 (validated empirically)
- Multi-process user sharding to utilize all CPU cores

**Computational Complexity**:
```
Time per user (k=6, composite, 2 taus):
- TE tau=1:    19 min
- TE tau=2:    20 min
- CTE tau=1:   99 min
- CTE tau=2:  100 min
- STE + GC:     1 min
Total:         ~4 hours

Full analysis (60 users, 4 processes):
- 15 users/process × 4 hours = 60 hours
```

## Project Structure

```
extrasensory_analysis/
├── config/
│   ├── proposal.yaml           # Main configuration (YAML)
│   ├── README.md               # Config documentation
│   └── MIGRATION_NOTES.md      # Hardcoded → config migration
├── src/
│   ├── analysis.py             # TE/CTE/STE/GC implementations
│   ├── preprocessing.py        # Data loading + feature engineering
│   ├── k_selection.py          # AIS-based k selection
│   ├── fdr_utils.py            # FDR correction utilities
│   ├── granger_analysis.py     # VAR-based Granger causality
│   ├── symbolic_te.py          # Symbolic transfer entropy
│   ├── jidt_adapter.py         # JIDT Java bridge
│   └── settings.py             # Legacy constants (being phased out)
├── tools/
│   └── validate_outputs.py     # Schema validation
├── tests/
│   └── ...                     # Unit tests
├── run_production.py           # Main pipeline with --shard support
├── run_parallel_4shards.sh     # Parallel execution launcher (Linux/macOS)
├── run_parallel_4shards.bat    # Parallel execution launcher (Windows)
├── merge_shard_results.py      # Results merger for parallel runs
├── PARALLEL_EXECUTION.md       # Parallel execution guide
├── EXECUTION_PLAN.md           # Performance analysis
├── PROJECT_STATUS.md           # Current implementation status
├── QUICK_START.md              # Quick reference
└── README.md                   # This file
```

## Troubleshooting

**JVM Out of Memory (k=6)**:
```
Error: Requested memory for base 5, k=6, l=6 is too large
Solution: Increase heap to 12GB in config/proposal.yaml
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
