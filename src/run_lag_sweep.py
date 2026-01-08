"""
Lag sweep runner (N=60 users) for discrete Conditional TE with manual offsets.

For each user and tau in [-3,-2,-1,0,1,2,3]:
- Manually slice arrays to implement the lag (no DELAY property)
- Apply safe truncation so that (L - k_dest) is divisible by BLOCK_SIZE=24
- Compute TE(A->S|H) and TE(S->A|H) using _compute_cte_value_only
- Compute p-values via manual block-permutation surrogates (source only)

Outputs: per_user_lag_sweep_FINAL.csv in analysis/out/lag_sweep_<STAMP>/
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src import preprocessing
from src.params import CTEParams
from src.jidt_adapter import _compute_cte_value_only
from src.analysis import start_jvm, shutdown_jvm


BLOCK_SIZE = 24
K_DEST = 2  # target history (S)
K_SRC = 1   # source history (A) implicit in the calculator
TAUS = [-3, -2, -1, 0, 1, 2, 3]


def manual_offset_arrays(A: np.ndarray, S: np.ndarray, H: np.ndarray, tau: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply integer lag tau by manual slicing.
    Positive tau means S/H shift forward (A leads), negative tau means A leads.
    Returns aligned arrays of equal length.
    """
    if tau == 0:
        return A.copy(), S.copy(), H.copy()
    if tau > 0:
        # align A[:-tau] with S[tau:], H[tau:]
        return A[:-tau], S[tau:], H[tau:]
    else:
        t = -tau
        # align A[t:] with S[:-t], H[:-t]
        return A[t:], S[:-t], H[:-t]


def safe_truncate(A: np.ndarray, S: np.ndarray, H: np.ndarray, k_dest: int = K_DEST, block: int = BLOCK_SIZE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Truncate so that (L - k_dest) is a multiple of block.
    """
    L = int(len(A))
    if not (len(S) == L and len(H) == L):
        m = min(L, len(S), len(H))
        A, S, H = A[:m], S[:m], H[:m]
        L = m
    n_eff = L - k_dest
    if n_eff < block:
        # not enough data for one safe block; return as-is and let caller decide
        return A[:L], S[:L], H[:L]
    n_eff_safe = (n_eff // block) * block
    L_safe = n_eff_safe + k_dest
    return A[:L_safe], S[:L_safe], H[:L_safe]


def block_permute_source(src: np.ndarray, k_hist: int, block: int, rng: np.random.Generator) -> np.ndarray:
    if len(src) <= k_hist:
        return src.copy()
    head = src[:k_hist]
    tail = src[k_hist:]
    blocks = [tail[i:i+block] for i in range(0, len(tail), block)]
    order = rng.permutation(len(blocks))
    shuf_tail = np.concatenate([blocks[i] for i in order]) if blocks else tail
    return np.concatenate([head, shuf_tail])


def te_with_perm(A: np.ndarray, S: np.ndarray, H: np.ndarray, base_A: int, base_S: int, base_H: int, M: int = 1000) -> tuple[float, float]:
    params = CTEParams(
        base_source=base_A,
        base_dest=base_S,
        base_cond=base_H,
        k_source=K_SRC,
        k_dest=K_DEST,
        num_cond_bins=1,
        tau=1,
        num_surrogates=M,
    )
    te_actual = _compute_cte_value_only(A, S, H, params)
    rng = np.random.default_rng(42)
    te_sur = np.empty(M, dtype=float)
    for i in range(M):
        shuf_A = block_permute_source(A, k_hist=K_DEST, block=BLOCK_SIZE, rng=rng)
        te_sur[i] = _compute_cte_value_only(shuf_A, S, H, params)
    if np.isnan(te_actual):
        p = float('nan')
    else:
        p = float((1 + np.sum(te_sur >= te_actual)) / (M + 1))
    return te_actual, p


def load_user_ids_from_report() -> List[str]:
    # Prefer the stable 60-user list from report/data
    p = Path('report/data/per_user_true_cte.csv')
    if p.exists():
        df = pd.read_csv(p)
        df = df[df.get('tau', 1) == 1]
        ids = df.drop_duplicates(subset=['user_id'])['user_id'].tolist()
        return ids
    # Fallback: scan data directory
    data_root = Path('data/ExtraSensory.per_uuid_features_labels')
    ids = [f.name.split('.features_labels.csv')[0] for f in data_root.glob('*.features_labels.csv')]
    return ids[:60]


def main():
    ap = argparse.ArgumentParser(description='Run lag sweep across N=60 users (discrete CTE, manual offsets).')
    ap.add_argument('--outdir', default=None, help='Output directory (default: analysis/out/lag_sweep_<stamp>)')
    args = ap.parse_args()

    stamp = datetime.now().strftime('%Y%m%d_%H%M')
    out_dir = Path(args.outdir) if args.outdir else Path(f'analysis/out/lag_sweep_{stamp}')
    out_dir.mkdir(parents=True, exist_ok=True)

    user_ids = load_user_ids_from_report()
    if len(user_ids) != 60:
        print(f"WARN: user count = {len(user_ids)} (expected 60)")

    rows = []
    start_jvm()
    try:
        for uid in user_ids:
            try:
                df = preprocessing.load_subject_data(uid)
                A, S, H_raw, H = preprocessing.create_variables(df, feature_mode='composite', hour_bins=6, a_bins=5, s_mode='binary')
                A = A.astype(int); S = S.astype(int); H = H.astype(int)
                for tau in TAUS:
                    # Manual offset
                    A_tau, S_tau, H_tau = manual_offset_arrays(A, S, H, tau)
                    # Safe truncation using (L - k_dest)
                    A_safe, S_safe, H_safe = safe_truncate(A_tau, S_tau, H_tau, k_dest=K_DEST, block=BLOCK_SIZE)
                    # Base sizes
                    base_A = int(np.max(A_safe)) + 1 if len(A_safe) else 0
                    base_S = int(np.max(S_safe)) + 1 if len(S_safe) else 0
                    base_H = int(np.max(H_safe)) + 1 if len(H_safe) else 0
                    if min(len(A_safe), len(S_safe), len(H_safe)) < (K_DEST + BLOCK_SIZE):
                        rows.append({
                            'user_id': uid, 'tau': tau,
                            'TE_A_to_S': np.nan, 'p_A_to_S': np.nan,
                            'TE_S_to_A': np.nan, 'p_S_to_A': np.nan,
                        })
                        continue
                    # A->S
                    te_a2s, p_a2s = te_with_perm(A_safe, S_safe, H_safe, base_A, base_S, base_H)
                    # S->A
                    te_s2a, p_s2a = te_with_perm(S_safe, A_safe, H_safe, base_S, base_A, base_H)
                    rows.append({
                        'user_id': uid, 'tau': tau,
                        'TE_A_to_S': te_a2s, 'p_A_to_S': p_a2s,
                        'TE_S_to_A': te_s2a, 'p_S_to_A': p_s2a,
                    })
            except Exception as e:
                print(f"ERROR user {uid}: {e}")
                for tau in TAUS:
                    rows.append({'user_id': uid, 'tau': tau, 'TE_A_to_S': np.nan, 'p_A_to_S': np.nan, 'TE_S_to_A': np.nan, 'p_S_to_A': np.nan})
    finally:
        shutdown_jvm()

    out_csv = out_dir / 'per_user_lag_sweep_FINAL.csv'
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


if __name__ == '__main__':
    main()

