"""
Reporting utilities for the final constrained discrete CTE plan.

Reads per_user_true_cte_discrete_k2l1_blockperm.csv and computes:
- Bonferroni-corrected significance counts (raw p * N_users)
- Robust group statistics over the 60 Delta_TE values (tau==1):
  * Mean, sample SD, and N
  * One-sample t-test vs 0
  * Wilcoxon signed-rank test (median vs 0)
  * Sign test summary (counts of negatives / positives / zeros)
- Saves a histogram of the 60 Delta_TE values as fig_delta_te_distribution.png
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ttest_1samp, wilcoxon
import matplotlib.pyplot as plt


def summarize_results(csv_path: str | Path) -> dict:
    p = Path(csv_path)
    df = pd.read_csv(p)
    # Expect columns: user_id, tau, k, l, hour_bins, CTE_true_A2S, CTE_true_S2A, Delta_TE, p_A2S, p_S2A
    # Filter to tau==1 and keep 1 row per user (N=60)
    df1 = df[df['tau'] == 1].copy()
    df1 = df1.drop_duplicates(subset=['user_id'])
    # Delta values
    delta = pd.to_numeric(df1['Delta_TE'], errors='coerce').dropna().to_numpy()
    n = int(delta.size)
    # Bonferroni: multiply p by N and test against alpha
    alpha = 0.05
    bonf = n if n > 0 else 1
    # Use tau==1 p-values if present; else fall back to zeros for counts
    p_a2s = pd.to_numeric(df1.get('p_A2S'), errors='coerce') if 'p_A2S' in df1.columns else pd.Series([np.nan]*n)
    p_s2a = pd.to_numeric(df1.get('p_S2A'), errors='coerce') if 'p_S2A' in df1.columns else pd.Series([np.nan]*n)
    sig_A2S = int(((p_a2s * bonf) < alpha).fillna(False).sum())
    sig_S2A = int(((p_s2a * bonf) < alpha).fillna(False).sum())

    mu = float(np.mean(delta)) if n else float('nan')
    sd = float(np.std(delta, ddof=1)) if n > 1 else float('nan')
    t_stat, p_val = ttest_1samp(delta, 0.0) if n >= 2 else (float('nan'), float('nan'))
    # Wilcoxon signed-rank (non-parametric)
    try:
        w_stat, w_p = wilcoxon(delta) if n >= 1 else (float('nan'), float('nan'))
    except ValueError:
        w_stat, w_p = (float('nan'), float('nan'))
    # Sign test summary
    neg = int((delta < 0).sum())
    pos = int((delta > 0).sum())
    zeros = int((delta == 0).sum())

    # Plot and save histogram next to the CSV
    fig_path = p.parent / 'fig_delta_te_distribution.png'
    try:
        plt.figure(figsize=(6, 4))
        plt.hist(delta, bins=12, color="#4C78A8", alpha=0.9, edgecolor='white')
        plt.axvline(0.0, color="#777", ls=":", lw=1)
        plt.axvline(mu, color="#A05195", ls="-", lw=1.5, label=f"mean={mu:.4f}")
        plt.xlabel("Delta TE (bits)")
        plt.ylabel("Count")
        plt.title(f"Delta TE distribution (N={n}, tau=1)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()
        fig_saved = True
    except Exception:
        fig_saved = False

    return {
        'N_users': n,
        'Bonferroni_factor': bonf,
        'A2S_sig_count': sig_A2S,
        'S2A_sig_count': sig_S2A,
        'Delta_mean': mu,
        'Delta_sd': sd,
        'ttest_t': float(t_stat) if np.isfinite(t_stat) else float('nan'),
        'ttest_p': float(p_val) if np.isfinite(p_val) else float('nan'),
        'wilcoxon_W': float(w_stat) if np.isfinite(w_stat) else float('nan'),
        'wilcoxon_p': float(w_p) if np.isfinite(w_p) else float('nan'),
        'sign_neg': neg,
        'sign_pos': pos,
        'sign_zero': zeros,
        'figure_path': str(fig_path),
        'figure_saved': fig_saved,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Summarize constrained discrete CTE results')
    ap.add_argument('--csv', default='analysis/out/latest/per_user_true_cte_discrete_k2l1_blockperm.csv', help='Path to per_user CSV')
    args = ap.parse_args()
    res = summarize_results(args.csv)
    print('N_users          :', res['N_users'])
    print('Bonferroni factor:', res['Bonferroni_factor'])
    print('A2S sig count    :', res['A2S_sig_count'])
    print('S2A sig count    :', res['S2A_sig_count'])
    print('Delta mean (bits):', f"{res['Delta_mean']:.6f}")
    print('Delta sd (bits)  :', f"{res['Delta_sd']:.6f}")
    print('t-test t         :', f"{res['ttest_t']:.4f}")
    print('t-test p         :', f"{res['ttest_p']:.6g}")
    print('Wilcoxon W       :', f"{res['wilcoxon_W']:.4f}")
    print('Wilcoxon p       :', f"{res['wilcoxon_p']:.6g}")
    print('Sign test counts :', f"neg={res['sign_neg']} pos={res['sign_pos']} zero={res['sign_zero']}")
    print('Histogram saved  :', res['figure_saved'], res['figure_path'])


if __name__ == '__main__':
    main()
