# Configuration Guide

## Overview
All analysis parameters are configured via `config/proposal.yaml`. The pipeline validates required fields on startup and raises errors for missing/invalid values.

## Required Fields

### Core Settings
- `data_root`: Path to ExtraSensory data directory
- `out_dir`: Output directory template with `<STAMP>` placeholder
- `feature_modes`: List of feature engineering modes to analyze
  - Options: `composite`, `sma_only`, `variance_only`, `magnitude_only`

### Analysis Parameters
- `hour_bins`: Number of bins for CTE hour-of-day discretization (int ≥ 1)
  - Example: `6` for 4-hour bins (00-04, 04-08, ..., 20-24)
  - Example: `24` for 1-hour bins (0, 1, 2, ..., 23)
- `taus`: List of time delays for TE/CTE/STE (e.g., `[1, 2]`)

### K-Selection Strategy
```yaml
k_selection:
  strategy: "AIS"  # Options: "AIS", "GUARDED_AIS", "FIXED"
  k_fixed: 4       # Fallback value for FIXED strategy
  k_grid: [1, 2, 3, 4, 5, 6]  # Grid to search for AIS
  epsilon_bits: 0.005
  reuse_for: ["TE", "CTE", "STE"]
```

**Strategies:**
- `AIS`: Pure AIS k-selection (max_ais criterion), no constraints
- `GUARDED_AIS`: AIS with k_max=4 cap and undersampling guard
- `FIXED`: Use k_fixed value for all users

### Surrogate Testing
- `surrogates`: Number of surrogates for permutation testing (int > 0)

### FDR Correction
```yaml
fdr:
  families: ["TE", "CTE", "STE", "GC"]
  by_tau: true  # Separate FDR per (family, tau)
  alpha: 0.05
```

### JVM Configuration
```yaml
jvm:
  xms: "16g"  # Initial heap size
  xmx: "24g"  # Maximum heap size
  opts:
    - "-XX:+UseG1GC"
    - "-XX:MaxGCPauseMillis=300"
    - "-Djava.awt.headless=true"
```

### Runtime Settings
```yaml
runtime:
  concurrency: 2  # Number of parallel processes (currently unused)
  checkpoint: true
  heartbeat_interval: 60  # Status update interval (seconds)
```

## Configuration Best Practices

1. **No Hardcoded Defaults**: All parameters must be explicitly set in YAML
2. **Schema Validation**: Pipeline validates config on startup
3. **Fail-Fast**: Missing/invalid fields raise errors immediately
4. **Type Safety**: Config values are type-checked (e.g., hour_bins must be int)

## Example Configurations

### 6-Bin CTE with AIS k-selection
```yaml
hour_bins: 6
k_selection:
  strategy: "AIS"
  k_grid: [1, 2, 3, 4, 5, 6]
feature_modes: ["composite", "sma_only", "variance_only", "magnitude_only"]
taus: [1, 2]
```

### 24-Bin CTE with GUARDED_AIS (k_max=4)
```yaml
hour_bins: 24
k_selection:
  strategy: "GUARDED_AIS"
  k_grid: [1, 2, 3, 4, 5, 6]
feature_modes: ["composite"]
taus: [1, 2]
```

### Fixed k=4 for Quick Testing
```yaml
hour_bins: 6
k_selection:
  strategy: "FIXED"
  k_fixed: 4
feature_modes: ["composite"]
taus: [1]
surrogates: 100  # Reduced for speed
```

## Migration from Legacy Settings

**Before** (hardcoded in `src/settings.py`):
```python
NUM_HOUR_BINS = 6
MAX_K_AIS = 4
```

**After** (configured in `config/proposal.yaml`):
```yaml
hour_bins: 6
k_selection:
  strategy: "GUARDED_AIS"
  k_grid: [1, 2, 3, 4, 5, 6]
```

All code now reads from config, not hardcoded constants.
