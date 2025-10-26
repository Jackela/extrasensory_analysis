# K=6 Parallel Execution Plan

## Current Status

**Configuration Validated**:
- ✓ JVM heap: 12GB minimum for k=6 (tested, no OOM errors)
- ✓ Config: Pure AIS (k_max=null, undersampling_guard=false)
- ✓ CTE: 6-bin stratification (4-hour windows)
- ✓ Mode: composite only (single mode)
- ✓ Shard logic: Implemented and validated

**System Resources**:
- Total RAM: 64GB (48GB usable after OS)
- CPU cores: 8
- Optimal config: 4 processes × 12GB = 48GB

## Performance Analysis

### Single-Process Timeline (UNACCEPTABLE)

| Phase | Time/User | Total (60 users) |
|-------|-----------|------------------|
| TE tau=1 | 19 min | 19 hours |
| TE tau=2 | 20 min | 20 hours |
| CTE tau=1 | 99 min | 99 hours |
| CTE tau=2 | 100 min | 100 hours |
| STE + GC | 1 min | 1 hour |
| **TOTAL** | **4 hours** | **240 hours (10 days)** |

**With 4 modes**: 240 × 4 = 960 hours (40 days)
**With composite only**: 240 hours (10 days)

### 4-Process Parallel Timeline (OPTIMAL)

| Configuration | Users/Process | Total Time |
|---------------|---------------|------------|
| 4 processes × 12GB | 15 users | **60 hours (2.5 days)** |

**Speedup**: 240 hours / 60 hours = **4x faster**

## State Space Analysis

| k | State Space | Samples | Samples/State | Performance |
|---|-------------|---------|---------------|-------------|
| 4 | 10,000 | 2,287 | 0.23 | ✓ Fast (1.5s) |
| 6 | 1,000,000 | 2,287 | 0.0023 | ✓ Slow (19min) |

**k=6 Slowdown**: 760x - 1485x slower than k=4 (expected, not a bug)

## Execution Commands

### Option 1: Automated Scripts (Recommended)

**Windows**:
```cmd
run_parallel_4shards.bat
```

**Linux/macOS/Git Bash**:
```bash
chmod +x run_parallel_4shards.sh
./run_parallel_4shards.sh
```

### Option 2: Manual Launch

```bash
# Launch 4 parallel processes
python run_production.py --full --shard 0/4 > analysis/shard0.log 2>&1 &
python run_production.py --full --shard 1/4 > analysis/shard1.log 2>&1 &
python run_production.py --full --shard 2/4 > analysis/shard2.log 2>&1 &
python run_production.py --full --shard 3/4 > analysis/shard3.log 2>&1 &
```

## User Distribution

**60 users across 4 shards (modulo partitioning)**:

- **Shard 0/4**: Users [0,4,8,12,16,20,24,28,32,36,40,44,48,52,56] = 15 users
- **Shard 1/4**: Users [1,5,9,13,17,21,25,29,33,37,41,45,49,53,57] = 15 users
- **Shard 2/4**: Users [2,6,10,14,18,22,26,30,34,38,42,46,50,54,58] = 15 users
- **Shard 3/4**: Users [3,7,11,15,19,23,27,31,35,39,43,47,51,55,59] = 15 users

## Monitoring

### Real-time Progress

```bash
# View logs
tail -f analysis/shard0.log
tail -f analysis/shard1.log
tail -f analysis/shard2.log
tail -f analysis/shard3.log

# Check status files
cat analysis/out/full_bins6_*/status.json
```

### Expected Milestones

| Time | Progress | Status |
|------|----------|--------|
| Hour 1 | 0-5% | All processes started |
| Hour 12 | 20% | First users completed |
| Hour 24 | 40% | Halfway point |
| Hour 36 | 60% | More than half done |
| Hour 48 | 80% | Final stretch |
| Hour 60 | 100% | Completion |

## Post-Execution

### Merge Results

```bash
python merge_shard_results.py \
  --shards <dir1> <dir2> <dir3> <dir4> \
  --output analysis/out/merged_k6_full
```

### Validate Output

```bash
python tools/validate_outputs.py --dir analysis/out/merged_k6_full
```

## Risk Assessment

### Known Issues (Solved)

✓ **JVM OOM with 8GB heap**: Increased to 12GB (validated)
✓ **Single-thread slowness**: Implemented multi-process sharding
✓ **Unimplemented concurrency**: Now using process-level parallelism

### Remaining Risks

⚠️ **Disk Space**: Each shard generates ~100MB output (400MB total)
⚠️ **Process Crashes**: Monitor logs for JVM crashes or errors
⚠️ **Long Runtime**: 60 hours = 2.5 days continuous execution

### Mitigation

- Resume capability: Use `--resume <dir>` if shard crashes
- Checkpointing: Results saved after each user completion
- Logging: All errors captured in errors.log per shard

## Expected Results

**Output Files (per shard)**:
- per_user_te.csv (15 users × 2 taus = 30 rows)
- per_user_cte.csv (15 users × 2 taus = 30 rows)
- per_user_ste.csv (15 users × 2 taus = 30 rows)
- per_user_gc.csv (15 users = 15 rows)
- k_selected_by_user.csv (15 rows)
- hbin_counts.csv (15 rows)

**Merged Output**:
- per_user_te.csv (60 users × 2 taus = 120 rows)
- per_user_cte.csv (60 users × 2 taus = 120 rows)
- per_user_ste.csv (60 users × 2 taus = 120 rows)
- per_user_gc.csv (60 users = 60 rows)
- k_selected_by_user.csv (60 rows)
- hbin_counts.csv (60 rows)

## Alternative Configurations

### If 60 hours is too long

**Option A: Reduce to k=4** (NOT user-requested):
- Time: 60 hours → 4 hours (15x faster)
- Quality: Lower k, may miss higher-order dependencies

**Option B: Reduce users** (NOT full analysis):
- 30 users: 60 hours → 30 hours
- 15 users: 60 hours → 15 hours

**Option C: Increase processes** (Risk: OOM):
- 6 processes × 8GB = 48GB (may fail with OOM)
- Time: 60 hours → 40 hours

## Decision: Proceed with 4-Process k=6

**Rationale**:
1. User explicitly requested k=6 (pure AIS, no constraints)
2. 12GB heap validated (no OOM errors in smoke test)
3. 60 hours acceptable for scientific analysis
4. 4x speedup over single-process (240h → 60h)
5. Resume capability available if crashes occur

**Next Action**: Execute run_parallel_4shards.bat or .sh
