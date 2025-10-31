import sys
from pathlib import Path

import pandas as pd
import numpy as np

try:
    from scipy.stats import spearmanr
except Exception as e:  # pragma: no cover
    print("scipy not available; attempting to install...", file=sys.stderr)
    import subprocess, sys as _sys

    subprocess.check_call([_sys.executable, "-m", "pip", "install", "scipy", "--quiet"])  # noqa: S603,S607
    from scipy.stats import spearmanr  # type: ignore


BASE_DIR = Path("analysis/out/FINAL_RUN_k60_COMPLETE")
KSEL_PATH = BASE_DIR / "k_selected_by_user.csv"
TE_PATH = BASE_DIR / "per_user_te.csv"
TRUE_CTE_PATH = BASE_DIR / "per_user_true_cte.csv"


def load_subset_user_ids(k_selected_csv: Path) -> pd.Index:
    df = pd.read_csv(k_selected_csv)
    cols = {c.lower(): c for c in df.columns}
    # Prefer explicit per-modality k columns if present; otherwise fallback to a single selected k.
    if "k_a" in cols and "k_s" in cols:
        ka_col, ks_col = cols["k_a"], cols["k_s"]
        subset = df[(df[ka_col] <= 4) & (df[ks_col] <= 4)][cols.get("user_id", "user_id")]
    elif "k_selected" in cols:
        subset = df[df[cols["k_selected"]] <= 4][cols.get("user_id", "user_id")]
    else:
        raise ValueError(
            "Cannot find k selection columns (k_A/k_S or k_selected) in k_selected_by_user.csv"
        )
    return subset.astype(str).drop_duplicates().reset_index(drop=True)


def prepare_te(df: pd.DataFrame, user_ids: pd.Index) -> pd.DataFrame:
    # Filter tau==1 and subset of users; keep a single row per user if duplicates exist
    df = df[df["tau"] == 1]
    df = df[df["user_id"].astype(str).isin(set(user_ids))]
    # Ensure numeric
    df["Delta_TE"] = pd.to_numeric(df["Delta_TE"], errors="coerce")
    # Drop NaNs in the metric
    df = df.dropna(subset=["Delta_TE"])
    df = df[["user_id", "Delta_TE"]].drop_duplicates(subset=["user_id"], keep="first")
    df = df.rename(columns={"Delta_TE": "Delta_TE_Global"})
    return df


def prepare_true_cte(df: pd.DataFrame, user_ids: pd.Index) -> pd.DataFrame:
    df = df[df["tau"] == 1]
    df = df[df["user_id"].astype(str).isin(set(user_ids))]
    # Ensure numeric
    df["Delta_CTE_true"] = pd.to_numeric(df["Delta_CTE_true"], errors="coerce")
    df = df.dropna(subset=["Delta_CTE_true"])
    df = df[["user_id", "Delta_CTE_true"]].drop_duplicates(subset=["user_id"], keep="first")
    df = df.rename(columns={"Delta_CTE_true": "Delta_TE_TrueCTE"})
    return df


def main() -> int:
    # Load subset
    user_ids = load_subset_user_ids(KSEL_PATH)

    # Load TE data
    te = pd.read_csv(TE_PATH)
    true_cte = pd.read_csv(TRUE_CTE_PATH)

    te_sub = prepare_te(te, user_ids)
    true_cte_sub = prepare_true_cte(true_cte, user_ids)

    merged = pd.merge(te_sub, true_cte_sub, on="user_id", how="inner")

    N = len(merged)
    mean_global = float(merged["Delta_TE_Global"].mean()) if N else np.nan
    mean_true = float(merged["Delta_TE_TrueCTE"].mean()) if N else np.nan

    if N >= 2:
        r, p = spearmanr(merged["Delta_TE_Global"], merged["Delta_TE_TrueCTE"], nan_policy="omit")
    else:
        r, p = np.nan, np.nan

    # Print results
    print("Subset Consistency Test (tau=1, k<=4 for A and S)")
    print(f"N_users: {N}")
    print(f"Delta_TE (Global) mean: {mean_global:.6f}" if np.isfinite(mean_global) else "Delta_TE (Global) mean: NA")
    print(
        f"Delta_TE (True CTE) mean: {mean_true:.6f}" if np.isfinite(mean_true) else "Delta_TE (True CTE) mean: NA"
    )
    if np.isfinite(r):
        print(f"Spearman r: {r:.6f}")
        print(f"Spearman p-value: {p:.6g}")
    else:
        print("Spearman r: NA")
        print("Spearman p-value: NA")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

