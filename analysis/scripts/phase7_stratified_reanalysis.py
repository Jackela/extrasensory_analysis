import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats


BASE_DIR = Path("analysis/out/FINAL_RUN_k60_COMPLETE")
KSEL_PATH = BASE_DIR / "k_selected_by_user.csv"
SUMMARY_PATH = BASE_DIR / "final_results_summary_n60.csv"
OUT_PATH = BASE_DIR / "phase7_stratified_tau1.csv"


@dataclass
class GroupStats:
    group: str
    n: int
    mean: float
    median: float
    ci_lower: float
    ci_upper: float
    t_stat: float
    p_value: float


def bootstrap_ci_mean(x: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, rng: np.random.Generator = None) -> Tuple[float, float]:
    if rng is None:
        rng = np.random.default_rng(42)
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    if n == 0:
        return float("nan"), float("nan")
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = x[rng.integers(0, n, size=n, endpoint=False)]
        boot_means[i] = sample.mean()
    lo = np.quantile(boot_means, alpha / 2)
    hi = np.quantile(boot_means, 1 - alpha / 2)
    return float(lo), float(hi)


def compute_group_stats(name: str, x: np.ndarray) -> GroupStats:
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    mean = float(np.mean(x)) if n else float("nan")
    median = float(np.median(x)) if n else float("nan")
    ci_l, ci_u = bootstrap_ci_mean(x, n_boot=1000, alpha=0.05)
    if n >= 2 and np.isfinite(x).all():
        t_res = stats.ttest_1samp(x, popmean=0.0, alternative="two-sided")
        t_stat = float(t_res.statistic)
        p_val = float(t_res.pvalue)
    else:
        t_stat = float("nan")
        p_val = float("nan")
    return GroupStats(name, n, mean, median, ci_l, ci_u, t_stat, p_val)


def main() -> int:
    if not KSEL_PATH.exists():
        raise FileNotFoundError(f"Missing {KSEL_PATH}")
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing {SUMMARY_PATH}")

    ksel = pd.read_csv(KSEL_PATH)
    summ = pd.read_csv(SUMMARY_PATH)

    required_ksel_cols = {"user_id", "k_selected"}
    if not required_ksel_cols.issubset(ksel.columns):
        raise ValueError(f"k_selected_by_user.csv missing columns: {required_ksel_cols - set(ksel.columns)}")

    required_summ_cols = {"user_id", "tau", "Delta_TE"}
    if not required_summ_cols.issubset(summ.columns):
        raise ValueError(f"final_results_summary_n60.csv missing columns: {required_summ_cols - set(summ.columns)}")

    # Merge on user_id so each row has k_selected
    merged = summ.merge(ksel[["user_id", "k_selected"]], on="user_id", how="left", validate="many_to_one")
    if merged["k_selected"].isna().any():
        missing = merged.loc[merged["k_selected"].isna(), "user_id"].unique().tolist()
        raise ValueError(f"Some rows missing k_selected after merge. Example user_ids: {missing[:5]}")

    # Filter tau == 1
    m_tau1 = merged[merged["tau"] == 1].copy()

    # Define strata based on k_selected
    high_mask = m_tau1["k_selected"] >= 5
    low_mask = m_tau1["k_selected"] <= 4

    high_vals = m_tau1.loc[high_mask, "Delta_TE"].to_numpy()
    low_vals = m_tau1.loc[low_mask, "Delta_TE"].to_numpy()

    high_stats = compute_group_stats("High-k (k>=5)", high_vals)
    low_stats = compute_group_stats("Low-k (k<=4)", low_vals)

    # Prepare output table
    out_rows = []
    for gs in (high_stats, low_stats):
        out_rows.append({
            "group": gs.group,
            "N": gs.n,
            "mean_Delta_TE": gs.mean,
            "median_Delta_TE": gs.median,
            "ci95_lower": gs.ci_lower,
            "ci95_upper": gs.ci_upper,
            "t_stat": gs.t_stat,
            "p_value": gs.p_value,
        })

    out_df = pd.DataFrame(out_rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)

    # Pretty console output
    print("Stratified ΔTE results (tau=1) saved to:", OUT_PATH)
    print()
    print(out_df.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    # Also print a compact JSON for programmatic checks
    print()
    print(json.dumps(out_rows, indent=2))

    # Quick sanity on counts to match expectation (~44 high-k, ~16 low-k)
    print()
    print("Sanity counts:")
    print("  High-k rows:", int(high_mask.sum()))
    print("  Low-k rows:", int(low_mask.sum()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

