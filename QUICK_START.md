# Quick Start Guide

## TL;DR - Zero Hardcoding Execution

**All configuration is preset-based. No hardcoded values.**

```bash
# Fast validation (2 users, k=4, 2 minutes)
python run_production.py smoke

# Full k=6 analysis (60 users, 4-process parallel, 60 hours)
./run_parallel_4shards.sh k6_full

# Fast k≤4 analysis (60 users, 4 modes, 4 hours)
python run_production.py k4_fast

# High-resolution 24-bin CTE (60 users, 5 hours)
python run_production.py 24bin_cte
```

## Available Presets

| Preset | Users | Modes | K | Runtime | Use Case |
|--------|-------|-------|---|---------|----------|
| **smoke** | 2 | 1 | k=4 (fixed) | 2 min | Validation |
| **k6_full** | 60 | 1 | k≤6 (AIS) | 60h (parallel) | Scientific optimal |
| **k4_fast** | 60 | 4 | k≤4 (guarded) | 4h | Fast comprehensive |
| **24bin_cte** | 60 | 1 | k≤4 (guarded) | 5h | Fine temporal detail |

## Execution Patterns

### 1. Smoke Test (Validate Setup)
```bash
python run_production.py smoke
```

### 2. Production k=6 (Parallel Recommended)
```bash
# Linux/macOS/Git Bash
./run_parallel_4shards.sh k6_full

# Windows
run_parallel_4shards.bat k6_full
```

### 3. Single-Process Execution
```bash
python run_production.py k6_full
# 240 hours (not recommended, use parallel instead)
```

### 4. Custom User Count
```bash
python run_production.py k4_fast --n-users 10
```

### 5. Custom Configuration
```bash
python run_production.py --config path/to/custom.yaml
```

## Monitoring Progress

```bash
# View logs
tail -f analysis/shard0.log  # Linux/Git Bash
type analysis\shard0.log     # Windows

# Check status
cat analysis/out/*/status.json
```

## Merge Results (After Parallel Execution)

```bash
python merge_shard_results.py \
  --shards analysis/out/k6_full_20251026_143{0,1,2,3} \
  --output analysis/out/merged_k6_full
```

## Troubleshooting

### "Unknown preset"
```bash
# Available: smoke, k6_full, k4_fast, 24bin_cte
python run_production.py --help
```

### Process Crashed
```bash
# Resume specific shard
python run_production.py k6_full --shard 2/4 \
  --resume analysis/out/k6_full_20251026_1432
```

### Out of Memory (k=6)
- Check JVM heap ≥12GB in `config/presets/k6_full.yaml`
- Verify only 4 processes running (not more)

## Configuration Details

All presets are in `config/presets/`:
- `smoke.yaml` - Fast validation
- `k6_full.yaml` - Pure AIS k=6
- `k4_fast.yaml` - GUARDED_AIS k≤4
- `24bin_cte.yaml` - 24-bin CTE

See [config/presets/README.md](config/presets/README.md) for full documentation.
