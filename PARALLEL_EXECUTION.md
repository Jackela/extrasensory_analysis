# Parallel Execution Guide

Multi-process user sharding for k=6 analysis optimization.

## Overview

**Problem**: Single-process k=6 analysis requires 552 hours (23 days)
**Solution**: 4-process parallel execution reduces to ~60 hours (2.5 days)

**Configuration**:
- 4 processes × 12GB heap = 48GB total memory
- Each process handles 15 users (60 users / 4)
- Time per user: ~4 hours (k=6)

## Quick Start

### Windows (CMD/PowerShell)

```cmd
run_parallel_4shards.bat
```

### Linux/macOS/Git Bash

```bash
chmod +x run_parallel_4shards.sh
./run_parallel_4shards.sh
```

## Manual Execution

Launch 4 processes manually:

```bash
# Shard 0: Users 0, 4, 8, 12, 16, ...
python run_production.py --full --shard 0/4 > analysis/shard0.log 2>&1 &

# Shard 1: Users 1, 5, 9, 13, 17, ...
python run_production.py --full --shard 1/4 > analysis/shard1.log 2>&1 &

# Shard 2: Users 2, 6, 10, 14, 18, ...
python run_production.py --full --shard 2/4 > analysis/shard2.log 2>&1 &

# Shard 3: Users 3, 7, 11, 15, 19, ...
python run_production.py --full --shard 3/4 > analysis/shard3.log 2>&1 &
```

## Monitoring Progress

### Real-time Log Monitoring

```bash
# Windows (CMD)
type analysis\shard0.log

# Linux/Git Bash
tail -f analysis/shard0.log
tail -f analysis/shard1.log
tail -f analysis/shard2.log
tail -f analysis/shard3.log
```

### Check Individual Shard Status

```bash
# Each shard creates its own output directory
ls -lh analysis/out/full_bins6_*/status.json

# View status
cat analysis/out/full_bins6_20251026_1430/status.json
```

## Merging Results

After all shards complete, merge results:

```bash
python merge_shard_results.py \
  --shards analysis/out/full_bins6_20251026_1430 \
           analysis/out/full_bins6_20251026_1431 \
           analysis/out/full_bins6_20251026_1432 \
           analysis/out/full_bins6_20251026_1433 \
  --output analysis/out/merged_k6_full
```

## Configuration Details

### Current Setup (config/proposal.yaml)

```yaml
hour_bins: 6          # 6-bin CTE (4-hour windows)
feature_modes:
  - composite         # Single mode for k=6 run

k_selection:
  strategy: "AIS"     # Pure AIS (no constraints)
  k_grid: [1,2,3,4,5,6]
  k_max: null         # No hard cap
  undersampling_guard: false

jvm:
  xms: "6g"
  xmx: "12g"          # Minimum for k=6 (validated)
```

### Memory Requirements

| k | State Space | JVM Heap | Status |
|---|-------------|----------|--------|
| 4 | 10,000      | 8GB      | ✓ Fast |
| 6 | 1,000,000   | 12GB     | ✓ Slow |
| 6 | 1,000,000   | 8GB      | ✗ OOM  |

### Performance Metrics

| Configuration | Time/User | Total Time (60 users) |
|---------------|-----------|----------------------|
| k=4 single    | 4 min     | 4 hours              |
| k=6 single    | 4 hours   | 552 hours (23 days)  |
| k=6 4-process | 4 hours   | 60 hours (2.5 days)  |

## User Distribution

With `--shard ID/TOTAL`, users are distributed using modulo partitioning:

**Example (60 users, 4 shards)**:
- Shard 0/4: Users [0,4,8,12,16,20,24,28,32,36,40,44,48,52,56] (15 users)
- Shard 1/4: Users [1,5,9,13,17,21,25,29,33,37,41,45,49,53,57] (15 users)
- Shard 2/4: Users [2,6,10,14,18,22,26,30,34,38,42,46,50,54,58] (15 users)
- Shard 3/4: Users [3,7,11,15,19,23,27,31,35,39,43,47,51,55,59] (15 users)

## Troubleshooting

### Process Killed/OOM
- **Symptom**: Process exits with "Killed" message
- **Cause**: JVM heap too small (<12GB for k=6)
- **Solution**: Increase heap to 12GB in config/proposal.yaml

### All Processes Slow
- **Symptom**: All shards running but very slow
- **Cause**: Insufficient CPU cores or disk I/O bottleneck
- **Solution**: Reduce to 2 processes or increase --concurrency

### Shard Output Missing
- **Symptom**: Some shard directories don't exist
- **Cause**: Process crashed or never started
- **Solution**: Check shard log files for errors

### Duplicate Users in Merged Results
- **Symptom**: Same user_id appears multiple times in merged CSV
- **Cause**: Resume mode or incorrect shard partitioning
- **Solution**: Delete shard outputs and restart from scratch

## Advanced Usage

### Custom Shard Count

```bash
# 2 processes (24GB each)
python run_production.py --full --shard 0/2 &
python run_production.py --full --shard 1/2 &

# 6 processes (8GB each - may fail with OOM)
python run_production.py --full --shard 0/6 &
python run_production.py --full --shard 1/6 &
# ... (not recommended for k=6)
```

### Resume Failed Shard

```bash
# Resume shard 2 that crashed
python run_production.py --full --shard 2/4 \
  --resume analysis/out/full_bins6_20251026_1432
```

## Expected Timeline

**4-Process Parallel Execution**:
- Start: Day 0 14:30
- Completion: Day 3 02:30 (~60 hours)
- Merge: Day 3 03:00 (+30 min)
- Validation: Day 3 03:30 (+30 min)

**Monitoring Schedule**:
- Hour 1: Check all 4 processes started
- Hour 12: Verify first users completed
- Hour 24: Halfway check (~50% progress)
- Hour 48: Final stretch (~80% progress)
- Hour 60: Completion expected

## Next Steps

1. **Start Parallel Run**:
   ```bash
   ./run_parallel_4shards.sh
   ```

2. **Monitor Progress** (every 12 hours):
   ```bash
   tail -n 20 analysis/shard*.log
   ```

3. **Merge Results** (after completion):
   ```bash
   python merge_shard_results.py --shards <dir1> <dir2> <dir3> <dir4> --output analysis/merged
   ```

4. **Validate**:
   ```bash
   python tools/validate_outputs.py --dir analysis/merged
   ```
