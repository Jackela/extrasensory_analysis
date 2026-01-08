"""
Reporting for lag sweep: aggregate means per tau and plot curves.

Input: per_user_lag_sweep_FINAL.csv (from src/run_lag_sweep.py)
Output: fig_lag_sweep_curves.png in the same directory
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon, t as student_t


def main():
    ap = argparse.ArgumentParser(description='Report lag sweep results (means per tau and curves).')
    ap.add_argument('--csv', required=True, help='Path to per_user_lag_sweep_FINAL.csv')
    args = ap.parse_args()

    p = Path(args.csv)
    df = pd.read_csv(p)
    # Ensure numeric
    df['tau'] = pd.to_numeric(df['tau'], errors='coerce')
    df['TE_A_to_S'] = pd.to_numeric(df['TE_A_to_S'], errors='coerce')
    df['TE_S_to_A'] = pd.to_numeric(df['TE_S_to_A'], errors='coerce')

    # Aggregate means per tau
    means = df.groupby('tau', as_index=False).agg(
        mean_A2S=('TE_A_to_S', 'mean'),
        mean_S2A=('TE_S_to_A', 'mean'),
        n=('user_id', 'count'),
    ).sort_values('tau')

    # Print a text table
    print('Lag sweep means (bits):')
    for _, r in means.iterrows():
        print(f"tau={int(r['tau']):2d}  mean_A2S={r['mean_A2S']:.6f}  mean_S2A={r['mean_S2A']:.6f}  n={int(r['n'])}")

    # Plot curves
    out_png = p.parent / 'fig_lag_sweep_curves.png'
    plt.figure(figsize=(7, 4.2))
    plt.plot(means['tau'], means['mean_A2S'], '-o', color='#4C78A8', label='TE(A→S|H)')
    plt.plot(means['tau'], means['mean_S2A'], '-o', color='#E45756', label='TE(S→A|H)')
    plt.axhline(0.0, color='#777', ls=':', lw=1)
    plt.xlabel('tau (samples)')
    plt.ylabel('TE (bits)')
    plt.title('Lag sweep: mean TE vs tau (N=60)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved: {out_png}")

    # Paired Tests: per-user S→A at tau ∈ {-1, 0, 1}
    print("\nPaired Tests (per-user S→A at tau = 1 vs -1, and 1 vs 0):")
    per_user = (
        df[["user_id", "tau", "TE_S_to_A"]]
        .dropna(subset=["tau"])
        .copy()
    )
    per_user = per_user.groupby(["user_id", "tau"], as_index=False)["TE_S_to_A"].mean()
    pv = per_user.pivot_table(index="user_id", columns="tau", values="TE_S_to_A", aggfunc="mean")
    def paired_arrays(t1, t2):
        sub = pv[[t1, t2]].dropna()
        return sub[t1].to_numpy(), sub[t2].to_numpy()

    try:
        a1, a_1 = paired_arrays(1, -1)
        t_stat_1, p_val_1 = ttest_rel(a1, a_1)
        w_stat_1, w_p_1 = wilcoxon(a1, a_1)
    except Exception:
        t_stat_1 = p_val_1 = w_stat_1 = w_p_1 = float("nan")
    print(f"Test 1 (g(1) vs g(-1)) t={t_stat_1:.6f} p={p_val_1:.6g}")
    print(f"Test 1 (g(1) vs g(-1)) Wilcoxon W={w_stat_1:.6f} p={w_p_1:.6g}")

    try:
        a1, a0 = paired_arrays(1, 0)
        t_stat_2, p_val_2 = ttest_rel(a1, a0)
        w_stat_2, w_p_2 = wilcoxon(a1, a0)
    except Exception:
        t_stat_2 = p_val_2 = w_stat_2 = w_p_2 = float("nan")
    print(f"Test 2 (g(1) vs g(0))  t={t_stat_2:.6f} p={p_val_2:.6g}")
    print(f"Test 2 (g(1) vs g(0))  Wilcoxon W={w_stat_2:.6f} p={w_p_2:.6g}")
    # Bar chart of mean differences with 95% CI
    try:
        sub10 = pv[[1, 0]].dropna()
        sub1m1 = pv[[1, -1]].dropna()
        d2 = (sub10[1] - sub10[0]).to_numpy()
        d1 = (sub1m1[1] - sub1m1[-1]).to_numpy()
        def mean_ci(a):
            n = len(a)
            mu = float(a.mean()) if n>0 else float('nan')
            sd = float(a.std(ddof=1)) if n>1 else float('nan')
            if n > 1:
                tcrit = float(student_t.ppf(0.975, df=n-1))
                se = sd / (n ** 0.5)
                lo, hi = mu - tcrit * se, mu + tcrit * se
            else:
                lo = hi = float('nan')
            return mu, lo, hi
        mu1, lo1, hi1 = mean_ci(d1)
        mu2, lo2, hi2 = mean_ci(d2)
        def p_str(p):
            try:
                return 'p < 1e-7' if p < 1e-7 else f"p = {p:.2e}"
            except Exception:
                return 'p = nan'
        p1s = p_str(p_val_1)
        p2s = p_str(p_val_2)
        out_bar = p.parent / 'fig_paired_tests.png'
        labels = ['g(1)-g(-1)', 'g(1)-g(0)']
        means_bar = [mu1, mu2]
        ci_lows = [mu1 - lo1 if np.isfinite(lo1) else 0.0, mu2 - lo2 if np.isfinite(lo2) else 0.0]
        ci_highs = [hi1 - mu1 if np.isfinite(hi1) else 0.0, hi2 - mu2 if np.isfinite(hi2) else 0.0]
        x = np.arange(len(labels))
        plt.figure(figsize=(5.5, 4))
        plt.bar(x, means_bar, yerr=[ci_lows, ci_highs], capsize=6, color=['#4C78A8', '#E45756'], alpha=0.9)
        for i, (m, ps) in enumerate(zip(means_bar, [p1s, p2s])):
            y_annot = m + (ci_highs[i] if np.isfinite(ci_highs[i]) else 0) + 0.002
            plt.text(i, y_annot, ps, ha='center', va='bottom', fontsize=9)
        plt.xticks(x, labels)
        plt.axhline(0.0, color='#777', ls=':', lw=1)
        plt.ylabel('Mean difference in TE (bits)')
        plt.title('Paired differences with 95% CI')
        plt.tight_layout()
        plt.savefig(out_bar, dpi=300)
        plt.close()
        print(f"Saved: {out_bar}")
    except Exception as e:
        print(f"WARN: Could not generate paired tests plot: {e}")


if __name__ == '__main__':
    main()



