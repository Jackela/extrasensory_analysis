Archived configs used during methodological evolution.

Why archived:
- validation_N6_k4_1k.yaml: 第 1 阶段/1b 阶段的验证配置，用于在固定 k=4 的条件下对比 Stratified-TE 与 True CTE。该阶段发现 Stratified-TE（Fisher 合并法）在方法论上不可靠（其结论与 True CTE 相反），因此后续被弃用。
- diagnostic_k_qc.yaml: 第 4 阶段的诊断配置，用于提取 k-selection 与小时分箱样本统计（QC）。诊断结果显示 73% 的用户（44/60）在 AIS 策略下选到 k=6，这是我们转向 True CTE 的核心依据之一。

What to use now:
- 最终运行请使用 `config/presets/production_k6_true_cte.yaml`（FINAL RUN CONFIG）。该配置与最终产出目录 `analysis/out/FINAL_RUN_k60_COMPLETE` 对齐。

Notes:
- 归档配置文件保留以重现场景与决策过程，但不再用于生产运行。
