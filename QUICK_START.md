# Quick Start Guide - K=6 Parallel Execution

## TL;DR

**Goal**: Analyze 60 users with k=6 using 4 parallel processes
**Time**: 60 hours (2.5 days)
**Memory**: 4 processes × 12GB = 48GB
**Speedup**: 4x faster than single-process (240h → 60h)

## 1. Start Execution

### Windows
```cmd
run_parallel_4shards.bat
```

### Linux/macOS/Git Bash
```bash
chmod +x run_parallel_4shards.sh
./run_parallel_4shards.sh
```

## 2. Monitor Progress

```bash
# View logs
tail -f analysis/shard0.log  # Linux/Git Bash
type analysis\shard0.log     # Windows

# Check status
cat analysis/out/full_bins6_*/status.json
```

## 3. Merge Results (After Completion)

```bash
python merge_shard_results.py \
  --shards analysis/out/full_bins6_20251026_1430 \
           analysis/out/full_bins6_20251026_1431 \
           analysis/out/full_bins6_20251026_1432 \
           analysis/out/full_bins6_20251026_1433 \
  --output analysis/out/merged_k6_full
```

## 4. Validate

```bash
python tools/validate_outputs.py --dir analysis/out/merged_k6_full
```

## Manual Launch (Alternative)

```bash
python run_production.py --full --shard 0/4 > analysis/shard0.log 2>&1 &
python run_production.py --full --shard 1/4 > analysis/shard1.log 2>&1 &
python run_production.py --full --shard 2/4 > analysis/shard2.log 2>&1 &
python run_production.py --full --shard 3/4 > analysis/shard3.log 2>&1 &
```

## Expected Timeline

| Time | Progress | What to Check |
|------|----------|---------------|
| Hour 1 | 0-5% | All 4 processes started |
| Hour 12 | 20% | First users completed |
| Hour 24 | 40% | Halfway point |
| Hour 48 | 80% | Final stretch |
| Hour 60 | 100% | All shards done |

## Troubleshooting

### Process Crashed
```bash
# Resume failed shard (e.g., shard 2)
python run_production.py --full --shard 2/4 \
  --resume analysis/out/full_bins6_20251026_1432
```

### Too Slow
- Expected: 19 min per TE tau=1 (k=6 is 760x slower than k=4)
- NOT a bug - state space explosion with 1 million states

### Out of Memory
- Check JVM heap is 12GB (config/proposal.yaml)
- Verify only 4 processes running (not more)

## Files Created

**Execution scripts**:
- `run_parallel_4shards.bat` - Windows launcher
- `run_parallel_4shards.sh` - Linux/macOS launcher
- `merge_shard_results.py` - Results merger

**Documentation**:
- `PARALLEL_EXECUTION.md` - Detailed guide
- `EXECUTION_PLAN.md` - Timeline analysis
- `PROJECT_STATUS.md` - Current status
- `QUICK_START.md` - This file

## Configuration

**config/proposal.yaml**:
```yaml
hour_bins: 6
feature_modes: [composite]
k_selection:
  strategy: "AIS"
  k_max: null
  undersampling_guard: false
jvm:
  xmx: "12g"
```

## Need Help?

See detailed documentation:
- **PARALLEL_EXECUTION.md** - Complete parallel execution guide
- **EXECUTION_PLAN.md** - Performance analysis and timeline
- **PROJECT_STATUS.md** - Current configuration and status
