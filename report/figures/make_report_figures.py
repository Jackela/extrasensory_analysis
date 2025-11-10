#!/usr/bin/env python3
"""
Generate core figures for the final report:

Fig. 1  (k-selection distribution): report/figures/fig_k_distribution.png
Fig. 2  (Forest plot, N=60):        report/figures/fig_forest_plot_n60.png
Fig. 3  (12-cell sensitivity):       report/figures/fig_sensitivity_heatmap.png

All code, comments, and docs are in English per project guidelines.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "report" / "data"
FIG_DIR = REPO_ROOT / "report" / "figures"


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


# Acklam's inverse normal CDF approximation (probit) for two-sided p->z conversion
# Source: https://web.archive.org/web/20150910022746/http://home.online.no/~pjacklam/notes/invnorm/
def _norm_ppf(p: float) -> float:
    if not 0.0 < p < 1.0:
        if p == 0.0:
            return -math.inf
        if p == 1.0:
            return math.inf
        raise ValueError("p must be in (0,1)")

    # Coefficients in rational approximations
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    plow = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    if phigh < p:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )

    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
        )
    )


def two_sided_p_to_abs_z(p: float, clamp: Tuple[float, float] = (1e-12, 1 - 1e-12)) -> float:
    """Convert a two-sided p-value to |z| under a standard normal test.

    For edge cases (p<=0 or p>=1) we clamp to avoid inf/0.
    """
    p_clamped = min(max(p, clamp[0]), clamp[1])
    # two-sided p => tail prob per side = p/2; |z| = Phi^{-1}(1 - p/2)
    side = 1.0 - p_clamped / 2.0
    return abs(_norm_ppf(side))


def der_simonian_laird(yi: np.ndarray, vi: np.ndarray) -> Tuple[float, float, float, float, float]:
    """DerSimonian-Laird random-effects meta-analysis.

    Args:
        yi: effect sizes per study
        vi: within-study variances per study
    Returns:
        mu_hat, se_mu, ci_low, ci_high, I2
    """
    wi = 1.0 / vi
    # Fixed-effect pooled
    mu_fixed = np.sum(wi * yi) / np.sum(wi)
    k = yi.size
    # Cochran's Q
    Q = np.sum(wi * (yi - mu_fixed) ** 2)
    df = k - 1
    C = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
    tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0
    wi_star = 1.0 / (vi + tau2)
    mu_hat = np.sum(wi_star * yi) / np.sum(wi_star)
    se_mu = math.sqrt(1.0 / np.sum(wi_star))
    z = 1.96
    ci_low, ci_high = mu_hat - z * se_mu, mu_hat + z * se_mu
    I2 = max(0.0, (Q - df) / Q) if Q > 0 else 0.0
    return mu_hat, se_mu, ci_low, ci_high, I2


def plot_k_distribution():
    """Plot k-selection distribution filtered to the core N=60 users.

    The whitelist of 60 users is derived from per_user_true_cte.csv at tau==1.
    Mixed-format rows in k_selected_by_user_ALL.csv are robustly handled and
    de-duplicated by user_id before counting.
    """
    # Build whitelist (N=60) from per_user_true_cte.csv at tau==1
    core_path = DATA_DIR / "per_user_true_cte.csv"
    core_df = pd.read_csv(core_path)
    core_users = (
        core_df[core_df["tau"] == 1]
        .drop_duplicates(subset=["user_id"])["user_id"].tolist()
    )
    core_set = set(core_users)

    # Load k-selection raw file robustly
    ks_src = DATA_DIR / "k_selected_by_user_ALL.csv"
    try:
        ksdf = pd.read_csv(ks_src)
        # Try to normalize columns
        if "k_selected" not in ksdf.columns:
            last_col = ksdf.columns[-1]
            ksdf = ksdf.rename(columns={last_col: "k_selected"})
        if "user_id" not in ksdf.columns:
            ksdf = ksdf.rename(columns={ksdf.columns[0]: "user_id"})
        ksdf = ksdf[["user_id", "k_selected"]]
        ksdf["k_selected"] = pd.to_numeric(ksdf["k_selected"], errors="coerce")
        ksdf = ksdf.dropna(subset=["k_selected"]).copy()
        ksdf["k_selected"] = ksdf["k_selected"].astype(int)
    except Exception:
        import csv

        rows = []
        with open(ks_src, "r", newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            header = next(r, None)
            for row in r:
                if not row:
                    continue
                user = row[0].strip().strip('"')
                try:
                    kval = int(row[-1].strip().strip('"'))
                except Exception:
                    continue
                rows.append((user, kval))
        ksdf = pd.DataFrame(rows, columns=["user_id", "k_selected"])

    # Filter to core users and deduplicate
    ks_core = (
        ksdf[ksdf["user_id"].isin(core_set)]
        .drop_duplicates(subset=["user_id"])
        .copy()
    )

    # Count per k in 1..6
    k_bins = list(range(1, 7))
    counts = (
        ks_core["k_selected"].value_counts().reindex(k_bins, fill_value=0).sort_index()
    )

    plt.figure(figsize=(6, 4))
    sns.barplot(x=counts.index, y=counts.values, color="#4C78A8")
    plt.xlabel("Optimal k")
    plt.ylabel("User Count")
    plt.title("k-Selection Distribution (N=60 diagnostics; filtered to core users)")
    # Annotate k=6 count
    k6 = int(counts.get(6, 0))
    plt.text(
        5,
        k6 + max(1, int(0.03 * (counts.max() or 1))),
        f"k=6: N={k6}",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#333",
    )
    plt.tight_layout()
    out = FIG_DIR / "fig_k_distribution.png"
    plt.savefig(out, dpi=300)
    plt.close()


def plot_forest_n60():
    """Caterpillar plot for N=60 at tau==1 using per-user Delta_CTE_true only.

    Removes p→z→SE approximation and random-effects meta-analysis.
    Displays all 60 user-level Delta_CTE_true points (sorted), and reports
    group-level mean and 95% bootstrap CI (B=10000, user-level resampling).
    """
    src = DATA_DIR / "per_user_true_cte.csv"
    df = pd.read_csv(src)
    df = df[df["tau"] == 1].drop_duplicates(subset=["user_id"]).copy()

    # Sort by effect size for visual clarity
    df_sorted = df.sort_values("Delta_CTE_true").reset_index(drop=True)
    deltas = df_sorted["Delta_CTE_true"].to_numpy()
    n = deltas.size

    # Group-level mean and bootstrap CI
    rng = np.random.default_rng(12345)
    B = 10000
    mu_hat = float(np.mean(deltas))
    boots = np.mean(deltas[rng.integers(0, n, size=(B, n))], axis=1)
    ci_low, ci_high = np.quantile(boots, [0.025, 0.975])

    # Plot caterpillar (points only)
    fig, ax = plt.subplots(figsize=(7, max(6, 0.12 * n + 2)))
    y_positions = np.arange(n)
    ax.plot(df_sorted["Delta_CTE_true"], y_positions, "o", color="#A05195", ms=4)
    ax.axvline(0.0, color="#777", ls=":", lw=1)

    # Labels and formatting
    ax.set_xlabel("Delta TE (A→S − S→A) (bits)")
    ax.set_yticks([])
    ax.set_title("Caterpillar Plot: Core Results (N=60, tau=1)")

    # Summary annotation (bootstrap CI)
    summary_text = (
        f"Bootstrap mean μ̂ = {mu_hat:.6f}\n"
        f"95% CI [{ci_low:.6f}, {ci_high:.6f}]\n"
        f"B = {B}"
    )
    ax.text(
        0.99,
        0.02,
        summary_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc"),
    )

    plt.tight_layout()
    out = FIG_DIR / "fig_forest_plot_n60.png"
    plt.savefig(out, dpi=300)
    plt.close()


def plot_sensitivity_heatmap():
    """Plot 12-cell sensitivity heatmap using BH-FDR q-values for stars.

    - Computes BH-FDR across the 12 cells to obtain q_bh.
    - Annotates each cell with mean value and significance stars based on q_bh.
    - Adds units (bits) to the colorbar and clarifies sample regime in the title.
    """
    src = DATA_DIR / "sensitivity_12cell_matrix.csv"
    df = pd.read_csv(src)

    # Compute BH-FDR q-values over the 12 cells
    m = len(df)
    df = df.sort_values("p_value").reset_index(drop=True)
    df["rank"] = np.arange(1, m + 1)
    df["q_raw"] = df["p_value"] * m / df["rank"]
    q_adj = []
    curr = 1.0
    for i in range(m - 1, -1, -1):
        curr = min(curr, df.loc[i, "q_raw"])  # monotone
        q_adj.append(curr)
    df["q_bh"] = list(reversed(q_adj))
    df = df.sort_values(["S_mode", "H_bin_hours", "A_bins"]).reset_index(drop=True)

    # Prepare Y labels
    df["Y_label"] = df.apply(lambda r: f"{r['S_mode']}, H={int(r['H_bin_hours'])}h", axis=1)

    # Pivot to A_bins (x) vs Y_label (y)
    pivot = df.pivot_table(index="Y_label", columns="A_bins", values="mean_Delta_CTE_true", aggfunc="first")
    y_order = df["Y_label"].drop_duplicates().tolist()
    x_order = sorted(df["A_bins"].unique())
    pivot = pivot.reindex(index=y_order, columns=x_order)

    plt.figure(figsize=(7, 4.8))
    cmap = sns.color_palette("RdBu_r", as_cmap=True)
    vlim = max(abs(np.nanmin(pivot.values)), abs(np.nanmax(pivot.values)))
    ax = sns.heatmap(
        pivot,
        cmap=cmap,
        center=0.0,
        vmin=-vlim,
        vmax=vlim,
        cbar_kws={"label": "Mean ΔTE_true (bits)"},
        linewidths=0.5,
        linecolor="white",
        annot=False,
        fmt=".3f",
    )
    ax.set_xlabel("A_bins")
    ax.set_ylabel("S_mode, H_bin_hours")
    ax.set_title("Sensitivity (12 cells, N=10, k≤4): effect on ΔTE mean")

    # Stars by q-value
    def stars_q(q):
        if pd.isna(q):
            return ""
        if q <= 0.001:
            return "***"
        if q <= 0.01:
            return "**"
        if q <= 0.05:
            return "*"
        return ""

    # Map Y,A_bins -> (mean, q)
    q_map = df.set_index(["Y_label", "A_bins"])["q_bh"].to_dict()
    for i, ylab in enumerate(y_order):
        for j, abin in enumerate(x_order):
            val = pivot.loc[ylab, abin]
            qv = q_map.get((ylab, abin), np.nan)
            s = f"{val:.3f}\n{stars_q(qv)}"
            ax.text(j + 0.5, i + 0.5, s, ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()
    out = FIG_DIR / "fig_sensitivity_heatmap.png"
    plt.savefig(out, dpi=300)
    plt.close()


def main():
    ensure_dirs()
    print(f"Repo root: {REPO_ROOT}")
    plot_k_distribution()
    print("Saved fig_k_distribution.png")
    plot_forest_n60()
    print("Saved fig_forest_plot_n60.png")
    plot_sensitivity_heatmap()
    print("Saved fig_sensitivity_heatmap.png")


if __name__ == "__main__":
    main()
