"""
Run True CTE discretization sensitivity grid for N=10 users.

Grid:
  - A_bins: [3, 5, 7]
  - S_mode: ['binary', 'quantile3']
  - H_bin_hours: [4, 2] -> hour_bins = [6, 12]

For each combo and user:
  - Load subject data
  - Create variables with specified A_bins, S_mode, and hour_bins
  - Select k via AIS with a global cap k ∈ [1..4] for ALL combos
  - Compute True CTE at tau=1 using (A_disc, S_disc, H_disc)
  - Save per-combo CSV and a summary CSV
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, str(Path.cwd()))
from src import preprocessing, analysis
from src.k_selection import select_k_via_ais


BASE_DIR = Path("analysis/out/FINAL_RUN_k60_COMPLETE")
USER_LIST_PATH = Path("analysis/config/sensitivity_users_n10.txt")


@dataclass
class Combo:
    A_bins: int
    S_mode: str
    H_bin_hours: int

    @property
    def hour_bins(self) -> int:
        return int(24 / self.H_bin_hours)

    @property
    def name(self) -> str:
        return f"A{self.A_bins}_S{self.S_mode}_H{self.H_bin_hours}h"


def bootstrap_ci_mean(x: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    if n == 0:
        return float("nan"), float("nan")
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = x[rng.integers(0, n, size=n, endpoint=False)]
        means[i] = sample.mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def compute_summary(delta: np.ndarray) -> Dict[str, Any]:
    delta = np.asarray(delta, dtype=float)
    n = int(delta.shape[0])
    mean = float(np.nanmean(delta)) if n else float("nan")
    median = float(np.nanmedian(delta)) if n else float("nan")
    ci_l, ci_u = bootstrap_ci_mean(delta)
    if n >= 2 and np.isfinite(delta).all():
        t_res = stats.ttest_1samp(delta, popmean=0.0)
        t_stat = float(t_res.statistic)
        p_val = float(t_res.pvalue)
    else:
        t_stat = float("nan")
        p_val = float("nan")
    return {
        "N": n,
        "mean_Delta_CTE_true": mean,
        "median_Delta_CTE_true": median,
        "ci95_lower": float(ci_l),
        "ci95_upper": float(ci_u),
        "t_stat": t_stat,
        "p_value": p_val,
    }


def run_combo(users: List[str], combo: Combo, out_dir: Path) -> Path:

    records = []
    for uid in users:
        try:
            df = preprocessing.load_subject_data(uid)
            A, S, H_raw, H_binned = preprocessing.create_variables(
                df, feature_mode='composite', hour_bins=combo.hour_bins, a_bins=combo.A_bins, s_mode=combo.S_mode
            )

            base_A = combo.A_bins
            base_S = 2 if combo.S_mode == 'binary' else 3
            base_H = combo.hour_bins

            # Global AIS-based k selection with cap k ∈ [1..4] for ALL combos.
            # Reason: k<=4 ensures computational stability across all 12 discretization schemes
            # and avoids JIDT ArrayIndexOutOfBoundsException during True CTE significance testing.
            k_range = list(range(1, 5))  # 1..4 inclusive
            k_info = select_k_via_ais(
                A.astype(np.int32), base=combo.A_bins,
                k_range=k_range, num_surrogates=100, criterion='max_ais'
            )
            k = int(k_info['k_selected'])
            k_A, k_S = k, k

            # True CTE using (A, S, H_binned) with selected/fixed k_A, k_S
            cte = analysis.run_true_cte_analysis(
                A.astype(int), S.astype(int), H_binned.astype(int),
                k_A, k_S,
                base_A, base_S, base_H,
                tau=1, num_surrogates=300,
                adaptive_stages=None,
                early_stop_sig=None,
                early_stop_nonsig=None
            )

            records.append({
                "user_id": uid,
                "A_bins": combo.A_bins,
                "S_mode": combo.S_mode,
                "H_bin_hours": combo.H_bin_hours,
                "hour_bins": combo.hour_bins,
                "k": k,
                "tau": 1,
                "CTE_true_A2S": cte.get('CTE_true(A->S|H)'),
                "CTE_true_S2A": cte.get('CTE_true(S->A|H)'),
                "Delta_CTE_true": cte.get('Delta_CTE_true'),
                "p_A2S": cte.get('p_true_cte(A->S|H)'),
                "p_S2A": cte.get('p_true_cte(S->A|H)')
            })
        except Exception as e:
            records.append({
                "user_id": uid,
                "A_bins": combo.A_bins,
                "S_mode": combo.S_mode,
                "H_bin_hours": combo.H_bin_hours,
                "hour_bins": combo.hour_bins,
                "k": np.nan,
                "tau": 1,
                "TE_A2S": np.nan,
                "TE_S2A": np.nan,
                "Delta_TE": np.nan,
                "p_A2S": np.nan,
                "p_S2A": np.nan,
                "error": str(e)
            })

    df = pd.DataFrame(records)
    out_csv = out_dir / f"{combo.name}.csv"
    df.to_csv(out_csv, index=False)

    # Summary
    summary = compute_summary(df["Delta_CTE_true"].dropna().values)
    summary_row = {
        "combo": combo.name,
        **summary
    }
    (out_dir / "summary.csv").write_text("") if not (out_dir / "summary.csv").exists() else None
    # Append summary
    if (out_dir / "summary.csv").stat().st_size == 0:
        pd.DataFrame([summary_row]).to_csv(out_dir / "summary.csv", index=False)
    else:
        prev = pd.read_csv(out_dir / "summary.csv")
        pd.concat([prev, pd.DataFrame([summary_row])], ignore_index=True).to_csv(out_dir / "summary.csv", index=False)

    return out_csv


def main() -> int:
    users = [u.strip() for u in USER_LIST_PATH.read_text().splitlines() if u.strip()]
    # Ensure JVM is started for both AIS selection and True CTE with a large heap (48G machine)
    analysis.start_jvm(xms='16g', xmx='45g')
    stamp = datetime.now().strftime('%Y%m%d_%H%M')
    out_root = Path(f"analysis/out/sensitivity/{stamp}")
    out_root.mkdir(parents=True, exist_ok=True)

    grid = [
        Combo(A_bins=a, S_mode=s, H_bin_hours=h)
        for a in [3, 5, 7]
        for s in ['binary', 'quantile3']
        for h in [4, 2]
    ]

    meta = {
        "users": users,
        "grid": [c.__dict__ for c in grid],
        "note": "Grid prepared. S_mode 'quantile3' now implemented; both 'binary' and 'quantile3' will run."
    }
    (out_root / "run_info.json").write_text(json.dumps(meta, indent=2))

    # Iterate grid and produce per-combo results
    for combo in grid:
        combo_dir = out_root / combo.name
        combo_dir.mkdir(parents=True, exist_ok=True)
        run_combo(users, combo, combo_dir)

    print(f"Prepared sensitivity grid at {out_root}")
    print("Note: If executed, this will compute TE for implemented combos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
