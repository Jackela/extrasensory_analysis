# Group-level Statistics (k = 4, τ = 1)

## Transfer Entropy (A→S vs S→A)
- N = 60
- Median ΔTE = -0.077150 bits
- Hodges–Lehmann ΔTE = -0.078348 bits
- 95% CI (bootstrap) = [-0.086296, -0.070102] bits
- Wilcoxon W = 1.0000
- Two-sided p = 1.71415e-11
- One-sided p (ΔTE < 0) = 8.57075e-12
- Effect size r = -0.8686
- BH-FDR adjusted p = 3.4283e-11

## Conditional Transfer Entropy (A→S|H vs S→A|H)
- N = 60
- Median ΔCTE = -0.042465 bits
- Hodges–Lehmann ΔCTE = -0.041802 bits
- 95% CI (bootstrap) = [-0.048960, -0.034408] bits
- Wilcoxon W = 90.0000
- Two-sided p = 1.2529e-09
- One-sided p (ΔCTE < 0) = 6.26452e-10
- Effect size r = -0.7841
- BH-FDR adjusted p = 1.2529e-09

## Robustness across k (fraction of users with negative delta)
- ΔTE: k=1: 1.000, k=2: 1.000, k=3: 1.000, k=4: 0.983
- ΔCTE: k=1: 1.000, k=2: 1.000, k=3: 0.983, k=4: 0.933

## Users with n_samples < 100 at k=4
- None
