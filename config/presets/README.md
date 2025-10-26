# Configuration Presets

Pre-configured analysis profiles for common use cases.

## Available Presets

### `smoke` - Fast Validation
**Purpose**: Quick smoke test to validate setup and configuration

**Configuration**:
- Users: 2
- Feature modes: composite only
- K-selection: Fixed k=4 (no AIS)
- Surrogates: 100 (reduced for speed)
- JVM heap: 8GB
- Runtime: ~2 minutes

**Usage**:
```bash
python run_production.py smoke
```

**When to use**:
- Validating code changes
- Testing new configurations
- Verifying data pipeline

---

### `k6_full` - Pure AIS k=6 Analysis
**Purpose**: Unrestricted k-selection for maximum information capture

**Configuration**:
- Users: 60
- Feature modes: composite only
- K-selection: Pure AIS (k_max=null, no constraints)
- Expected k: 6 (1M state space)
- Surrogates: 1000
- JVM heap: 12GB (minimum for k=6)
- Runtime: 240 hours single-process, **60 hours with 4-process parallel**

**Usage**:
```bash
# Single-process (slow)
python run_production.py k6_full

# 4-process parallel (recommended)
./run_parallel_4shards.sh k6_full
run_parallel_4shards.bat k6_full  # Windows
```

**Performance**:
- k=6 is 760x slower than k=4 (state space explosion)
- Parallelization provides 4x speedup

**When to use**:
- Scientific analysis requiring optimal k
- Willing to accept long runtime for accuracy
- Have 48GB+ RAM for parallel execution

---

### `k4_fast` - GUARDED_AIS k≤4 Analysis
**Purpose**: Fast analysis with computational safety guards

**Configuration**:
- Users: 60
- Feature modes: All 4 (composite, sma_only, variance_only, magnitude_only)
- K-selection: GUARDED_AIS (k_max=4, undersampling_guard=true)
- Expected k: ≤4
- Surrogates: 1000
- JVM heap: 8GB
- Runtime: ~4 hours single-process

**Usage**:
```bash
python run_production.py k4_fast
```

**When to use**:
- Need fast results (4 hours vs 240 hours)
- Want to compare multiple feature modes
- Limited computational resources

---

### `24bin_cte` - High Temporal Resolution CTE
**Purpose**: Fine-grained hour-of-day stratification

**Configuration**:
- Users: 60
- Feature modes: composite only
- K-selection: GUARDED_AIS (k_max=4, k_grid=[1,2,3,4])
- Hour bins: 24 (1-hour windows)
- Surrogates: 1000
- JVM heap: 8GB
- Runtime: ~5 hours

**Usage**:
```bash
python run_production.py 24bin_cte
```

**Trade-offs**:
- Higher temporal resolution vs lower statistical power per bin
- Requires more samples per user (min 600 for 25 samples/bin)
- k capped at 4 to ensure bin stability

**When to use**:
- Studying fine-grained circadian patterns
- Users have high sample density
- Willing to accept k≤4 constraint

---

## Custom Configurations

Create your own YAML config and use:

```bash
python run_production.py --config path/to/custom.yaml
```

**Required fields**:
```yaml
data_root: "data/ExtraSensory.per_uuid_features_labels"
out_dir: "analysis/out/custom_<STAMP>"
n_users: 60
feature_modes: [composite]
taus: [1, 2]
k_selection: { strategy: "AIS", k_grid: [...] }
hour_bins: 6
surrogates: 1000
fdr: { alpha: 0.05, by_tau: true }
jvm: { xms: "6g", xmx: "12g" }
```

See [config/README.md](../README.md) for full parameter reference.

---

## Comparison Table

| Preset | Users | Modes | K-selection | Expected k | Runtime (single) | Parallel? |
|--------|-------|-------|-------------|------------|-----------------|-----------|
| **smoke** | 2 | 1 | Fixed | 4 | 2 min | No |
| **k6_full** | 60 | 1 | Pure AIS | 6 | 240 hours | **Yes (60h)** |
| **k4_fast** | 60 | 4 | GUARDED_AIS | ≤4 | 4 hours | Optional |
| **24bin_cte** | 60 | 1 | GUARDED_AIS | ≤4 | 5 hours | Optional |

---

## Preset Selection Guide

```
┌─────────────────────────┐
│  Need fast validation?  │
│         Yes             │
└────────┬────────────────┘
         │
         v
    [smoke preset]


┌──────────────────────────┐
│  Need optimal k (no cap)?│
│         Yes              │
└────────┬─────────────────┘
         │
         v
  [k6_full preset]
  + Use parallel execution


┌──────────────────────────┐
│  Need fast results (<5h)?│
│         Yes              │
└────────┬─────────────────┘
         │
         v
  [k4_fast preset]


┌──────────────────────────────┐
│  Need fine temporal detail?  │
│  (1-hour vs 4-hour bins)     │
│         Yes                  │
└────────┬─────────────────────┘
         │
         v
  [24bin_cte preset]
```

---

## Override Examples

**Override number of users**:
```bash
python run_production.py k4_fast --n-users 10
```

**Use preset with sharding**:
```bash
python run_production.py k6_full --shard 0/4
```

**Resume failed shard**:
```bash
python run_production.py k6_full --shard 2/4 \
  --resume analysis/out/k6_full_20251026_1430
```

---

## Creating New Presets

1. Copy existing preset:
   ```bash
   cp config/presets/k4_fast.yaml config/presets/my_custom.yaml
   ```

2. Edit parameters

3. Add to preset map in `run_production.py`:
   ```python
   preset_map = {
       # ...
       'my_custom': 'config/presets/my_custom.yaml'
   }
   ```

4. Use:
   ```bash
   python run_production.py my_custom
   ```
