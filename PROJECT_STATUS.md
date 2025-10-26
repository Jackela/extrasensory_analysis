# Project Status (2025-10-26)

## Summary
Multi-process parallel execution pipeline for k=6 transfer entropy analysis with pure AIS k-selection and 6-bin CTE stratification.

## Current Configuration
- **K-Selection**: Pure AIS (max_ais), no constraints (k_max=null, undersampling_guard=false)
- **Hour Bins**: 6 (4-hour windows: 00-04, 04-08, 08-12, 12-16, 16-20, 20-24)
- **Feature Modes**: 1 (composite only for k=6 run)
- **Users**: 60
- **Taus**: [1, 2]
- **Surrogates**: 1000
- **JVM Heap**: 12GB (minimum for k=6, validated)
- **Parallelization**: 4 processes × 12GB = 48GB total

## Performance Metrics

### State Space Analysis
- **k=4**: 10,000 states, 1.5s per TE computation
- **k=6**: 1,000,000 states, 19 min per TE computation (760x slower)

### Runtime Estimates
| Configuration | Time/User | Total (60 users) |
|---------------|-----------|------------------|
| k=4 single-process | 4 min | 4 hours |
| k=6 single-process | 4 hours | 240 hours (10 days) |
| **k=6 4-process** | **4 hours** | **60 hours (2.5 days)** |

## Completed Tasks
- [x] Remove hardcoded configuration constants
- [x] Implement centralized YAML-based configuration
- [x] Add config validation with fail-fast behavior
- [x] Implement pure AIS k-selection with optional constraints
- [x] Clean project structure (deprecated files archived)
- [x] Create comprehensive English documentation
- [x] Add GitHub best practice files (LICENSE, CONTRIBUTING.md, CHANGELOG.md, .gitignore)
- [x] Validate JVM heap requirements (8GB fails, 12GB works for k=6)
- [x] Implement multi-process user sharding (--shard argument)
- [x] Create parallel execution scripts (Windows .bat + Linux .sh)
- [x] Create results merger script (merge_shard_results.py)
- [x] Write comprehensive execution documentation

## Ready to Run

### Option 1: Automated Parallel Execution (Recommended)

**Windows**:
```cmd
run_parallel_4shards.bat
```

**Linux/macOS/Git Bash**:
```bash
chmod +x run_parallel_4shards.sh
./run_parallel_4shards.sh
```

### Option 2: Manual Parallel Execution

```bash
# Launch 4 parallel processes
python run_production.py --full --shard 0/4 > analysis/shard0.log 2>&1 &
python run_production.py --full --shard 1/4 > analysis/shard1.log 2>&1 &
python run_production.py --full --shard 2/4 > analysis/shard2.log 2>&1 &
python run_production.py --full --shard 3/4 > analysis/shard3.log 2>&1 &
```

### Option 3: Single-Process (Slower)

```bash
python run_production.py --full
# Time: 240 hours (10 days)
```

## User Distribution (4 Shards)

Modulo partitioning across 4 processes:

- **Shard 0/4**: 15 users [0,4,8,12,16,20,24,28,32,36,40,44,48,52,56]
- **Shard 1/4**: 15 users [1,5,9,13,17,21,25,29,33,37,41,45,49,53,57]
- **Shard 2/4**: 15 users [2,6,10,14,18,22,26,30,34,38,42,46,50,54,58]
- **Shard 3/4**: 15 users [3,7,11,15,19,23,27,31,35,39,43,47,51,55,59]

## Expected Outputs

### Per-Shard Outputs
Each shard creates: `analysis/out/full_bins6_YYYYMMDD_HHMM/`
- `per_user_te.csv` - TE results (15 users × 2 taus = 30 rows)
- `per_user_cte.csv` - CTE results (15 users × 2 taus = 30 rows)
- `per_user_ste.csv` - STE results (15 users × 2 taus = 30 rows)
- `per_user_gc.csv` - Granger causality (15 users = 15 rows)
- `k_selected_by_user.csv` - AIS k-selection tracking (15 rows)
- `hbin_counts.csv` - Hour bin sample counts (15 rows)
- `run_info.yaml` - Metadata
- `status.json` - Real-time progress
- `errors.log` - Error tracking

### Merged Outputs
After completion, merge with:
```bash
python merge_shard_results.py \
  --shards <dir1> <dir2> <dir3> <dir4> \
  --output analysis/out/merged_k6_full
```

Results:
- `per_user_te.csv` (60 users × 2 taus = 120 rows)
- `per_user_cte.csv` (60 users × 2 taus = 120 rows)
- `per_user_ste.csv` (60 users × 2 taus = 120 rows)
- `per_user_gc.csv` (60 users = 60 rows)
- `k_selected_by_user.csv` (60 rows with k=6 selections)
- `hbin_counts.csv` (60 rows × 24 hour columns)

## Monitoring Progress

### Real-time Logs
```bash
# Windows
type analysis\shard0.log

# Linux/Git Bash
tail -f analysis/shard0.log
tail -f analysis/shard1.log
tail -f analysis/shard2.log
tail -f analysis/shard3.log
```

### Status Files
```bash
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

## Post-Execution Validation

```bash
# Merge results
python merge_shard_results.py --shards <dirs> --output analysis/merged

# Validate
python tools/validate_outputs.py --dir analysis/merged
```

## Risk Assessment

### Solved Issues
✓ JVM OOM with 8GB heap (increased to 12GB)
✓ Single-thread performance (implemented multi-process)
✓ Unimplemented concurrency config (now process-level)
✓ k=6 slowness (accepted as expected behavior, 760x slower than k=4)

### Remaining Risks
⚠️ **Long Runtime**: 60 hours = 2.5 days continuous execution
⚠️ **Process Crashes**: JVM errors, system restarts, power loss
⚠️ **Disk Space**: 400MB total output (100MB per shard)

### Mitigation
- **Resume capability**: `--resume <dir>` for crashed shards
- **Checkpointing**: Per-user incremental saves
- **Error logging**: All errors captured in errors.log

## Configuration Reference

**config/proposal.yaml** (current):
```yaml
hour_bins: 6
feature_modes: [composite]

k_selection:
  strategy: "AIS"
  k_grid: [1, 2, 3, 4, 5, 6]
  k_max: null
  undersampling_guard: false

jvm:
  xms: "6g"
  xmx: "12g"
```

## Documentation

- **PARALLEL_EXECUTION.md**: Detailed parallel execution guide
- **EXECUTION_PLAN.md**: Timeline and performance analysis
- **README.md**: Project overview
- **CONTRIBUTING.md**: Contribution guidelines
- **CHANGELOG.md**: Version history
- **config/README.md**: Configuration reference

## Next Actions

1. **Start parallel execution**:
   ```bash
   ./run_parallel_4shards.sh
   ```

2. **Monitor progress** (every 12 hours):
   ```bash
   tail -n 20 analysis/shard*.log
   ```

3. **Merge results** (after 60 hours):
   ```bash
   python merge_shard_results.py --shards <dirs> --output analysis/merged
   ```

4. **Validate outputs**:
   ```bash
   python tools/validate_outputs.py --dir analysis/merged
   ```

## Timeline

**Estimated completion**: 60 hours from start (2.5 days)

**Example**:
- Start: 2025-10-26 15:00
- Completion: 2025-10-29 03:00
- Merge: 2025-10-29 03:30
- Validation: 2025-10-29 04:00
