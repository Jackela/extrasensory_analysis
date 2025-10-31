# ExtraSensory 因果信息分析（最终归档版）

本仓库包含一个针对 ExtraSensory 数据集的信息论因果分析流水线（基于 JIDT）。由于多次“方法论转向（pivot）”，历史代码与配置较为复杂。该 README 对核心背景进行一次性澄清，并给出最终归档与可复现入口。

## 核心上下文（务必理解的 5 点）

1) 最初的计划：Global TE + Stratified-TE（分箱 + Fisher 合并）
- 早期设计中，因果方向以 Transfer Entropy（TE）为主；条件因果以“先按小时分箱、在各箱内跑 TE、最后用 Fisher 合并 p 值”的 Stratified-TE 近似 CTE。

2) 验证阶段（第 1 阶段/1b 阶段）发现：Stratified-TE 在方法论上不可靠
- 在 k=4 的验证条件下，Stratified-TE 的结论与 True CTE 的结论相反，暴露出方法论缺陷。因此 Stratified-TE 被标记为“已废弃”。

3) 资源瓶颈：当 k≥5 时，Global TE 与 Stratified-TE 均可能 OOM
- 在 8–12GB JVM 堆规模下，随着历史阶数 k 增大（尤其是 k=6），全局 TE 与分箱 TE 的状态空间爆炸，实践中会出现内存不足（OOM）。

4) 诊断阶段（第 4 阶段）：k-Selection 诊断显示 73% 用户需要 k=6
- 通过 AIS 的 k-selection 诊断，我们发现 73% 的用户（44/60）最优 k=6。
- 该证据来自合并产物 `analysis/out/production_k6_true_cte_merged/k_selected_by_user_ALL.csv`，这是我们最终转向 True CTE 的核心依据。

5) 最终的转向（The Pivot）：抛弃 Stratified-TE，接受 Global TE 的高 k OOM，核心采用 True CTE
- 丢弃 Stratified-TE：因其在“第 1b 阶段”验证中被证明不可靠。
- 接受 Global TE 在 k≥5 的 OOM：在最终运行中，Global TE 的 OOM 率约 73.3%，与“73% 用户需要 k=6”一致。这是有意为之、可接受的记录方式（记为 NaN），因为我们的核心方法 True CTE 能在 k=6 与 8GB 内存下稳定得出结果。
- 选择 True CTE：方法论更稳健、内存效率更高，是最终结论的依据。

## 最终结果与证据

- 最终结果目录：`analysis/out/FINAL_RUN_k60_COMPLETE`
  - 包含 `per_user_true_cte.csv`（核心结果）、`per_user_te.csv`（其中高 k OOM 记为 NaN）、`run_info.yaml`、`k_selected_by_user.csv` 等。
- 转向证据：`analysis/out/production_k6_true_cte_merged/k_selected_by_user_ALL.csv`
  - 该文件汇总 AIS 选 k 的诊断结果，显示 44/60 用户选到 k=6（73%），直接支撑我们转向 True CTE 并接受 Global TE 在高 k OOM 的决策。

## 如何复现实验（最终运行配置）

- 使用最终运行配置：`config/presets/production_k6_true_cte.yaml`（文件头已标注 FINAL RUN CONFIG）。
  - 仅运行 Global TE 与 True CTE；其中 Global TE 在 k≥5 可能 OOM，这一情况会被有意记录为 NaN 并继续流水线；True CTE 负责产出核心结论。

示例：
```bash
# 使用最终配置直接运行（推荐）
python run_production.py --config config/presets/production_k6_true_cte.yaml

# 或按需分片/并行（Windows 下可用 *.bat 脚本等价方式）
python run_production.py --config config/presets/production_k6_true_cte.yaml --shard 0/4
```

环境与资源建议：
- Python 3.12+，Java 8+，JIDT 已配置
- 单进程 8–12GB JVM 堆；并行时按进程数线性叠加内存（例：4 进程≈32–48GB）

## 代码与配置清理说明

- `run_production.py`：在 Global TE 的 try...except 上方新增注释，明确这是“有意处理的 OOM”，当 k≥5 发生内存不足时记 NaN 并继续，因为 True CTE 在 k=6 能成功运行。
- `src/jidt_adapter.py`：在 Stratified-TE（Fisher 合并法）函数上方添加“[已废弃]”注释，说明其在“第 1b 阶段”验证中不可靠，已由 `compute_true_cte` 取代。
- `config/`：
  - `config/presets/production_k6_true_cte.yaml` 已标注为“FINAL RUN CONFIG”。
  - 旧配置移至 `config/archive/`，并附 `config/archive/README.md` 解释其在方法论演进中的角色（如验证用 `validation_N6_k4_1k.yaml`、诊断用 `diagnostic_k_qc.yaml`）。

## 你可能关心的问题（FAQ）

- 为什么接受 Global TE 的 OOM？
  - 因为大多数用户需要 k=6 才能建模最优，这时 Global TE 的状态空间导致 OOM 是可预期的。我们用 NaN 记录，并将结论基于能在 k=6 成功运行的 True CTE。

- Stratified-TE 为什么被弃用？
  - 在 k=4 验证中它与 True CTE 的结论相反，显示出方法论不可靠。因此从“第 1b 阶段”起弃用，改用 True CTE。

- 我应查看哪个结果文件？
  - 核心看 `analysis/out/FINAL_RUN_k60_COMPLETE/per_user_true_cte.csv`。Global TE 的 `per_user_te.csv` 可作参考，但其中高 k 的 NaN 属于有意记录。

## 目录索引

- 最终结果：`analysis/out/FINAL_RUN_k60_COMPLETE`
- 转向依据：`analysis/out/production_k6_true_cte_merged/k_selected_by_user_ALL.csv`
- 最终配置：`config/presets/production_k6_true_cte.yaml`
- 归档配置：`config/archive/`（含说明 README）


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
│   ├── template.yaml           # Configuration reference template
│   ├── presets/                # Pre-configured analysis profiles
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
