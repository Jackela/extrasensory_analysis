"""
Debug A/B test for causal (tau=+1) vs non-causal (tau=-1) CTE using the
discrete JIDT calculator and the project's safe-truncation helper.

Steps:
1) Load one user's data (default: 00EABED2-271D-49D8-B599-1D4A09240601).
2) Build A,S,H (discrete) via preprocessing (A: 5 bins, S: binary, H: 6 bins).
3) Causal test (tau=+1): pass arrays directly into _compute_cte_value_only()
4) Non-causal test (tau=-1 semantics):
   - A_nc = A[2:]; S_nc = S[:-2]; H_nc = H[:-2]
   - pass into _compute_cte_value_only()
5) Print TE(A->S|H), TE(S->A|H) and Delta for both tests.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from src import preprocessing
from src.params import CTEParams
from src.jidt_adapter import _compute_cte_value_only
from src.analysis import start_jvm, shutdown_jvm


def load_user_discrete(uuid: str):
    df = preprocessing.load_subject_data(uuid)
    A, S, H_raw, H_bin = preprocessing.create_variables(
        df,
        feature_mode='composite',
        hour_bins=6,
        a_bins=5,
        s_mode='binary',
    )
    return A.astype(int), S.astype(int), H_bin.astype(int)


def cte_pair(A: np.ndarray, S: np.ndarray, H: np.ndarray) -> tuple[float, float, float]:
    base_A = int(np.max(A)) + 1
    base_S = int(np.max(S)) + 1
    base_H = int(np.max(H)) + 1
    params_A2S = CTEParams(
        base_source=base_A,
        base_dest=base_S,
        base_cond=base_H,
        k_source=1,
        k_dest=2,
        num_cond_bins=1,
        tau=1,
        num_surrogates=1000,
    )
    params_S2A = CTEParams(
        base_source=base_S,
        base_dest=base_A,
        base_cond=base_H,
        k_source=1,
        k_dest=2,
        num_cond_bins=1,
        tau=1,
        num_surrogates=1000,
    )
    te_a2s = _compute_cte_value_only(A, S, H, params_A2S)
    te_s2a = _compute_cte_value_only(S, A, H, params_S2A)
    delta = float(te_a2s - te_s2a) if np.isfinite(te_a2s) and np.isfinite(te_s2a) else float('nan')
    return te_a2s, te_s2a, delta


def main():
    ap = argparse.ArgumentParser(description='Debug A/B test for tau=+1 vs tau=-1 (manual offset).')
    ap.add_argument('--uuid', default='00EABED2-271D-49D8-B599-1D4A09240601', help='User UUID')
    args = ap.parse_args()

    start_jvm()
    try:
        A, S, H = load_user_discrete(args.uuid)
        print(f"Loaded user {args.uuid}: N={len(A)} (A in [0,{int(np.max(A))}], S in [0,{int(np.max(S))}], H bins={int(np.max(H))+1})")

        # Test A: Causal (tau=+1 semantics) — pass arrays as-is
        teA_causal, teB_causal, d_causal = cte_pair(A, S, H)
        print("CAUSAL (tau=+1)")
        print(f"  TE(A->S|H) = {teA_causal:.6f}")
        print(f"  TE(S->A|H) = {teB_causal:.6f}")
        print(f"  Delta_TE    = {d_causal:.6f}")

        # Test B: Non-causal (tau=-1 semantics) — manual offset
        # A_nc = A[2:], S_nc = S[:-2], H_nc = H[:-2]
        if len(A) > 4:
            A_nc = A[2:]
            S_nc = S[:-2]
            H_nc = H[:-2]
        else:
            raise RuntimeError("Insufficient data length for manual non-causal offset")

        teA_nc, teB_nc, d_nc = cte_pair(A_nc, S_nc, H_nc)
        print("NON-CAUSAL (tau=-1 manual)")
        print(f"  TE(A->S|H) = {teA_nc:.6f}")
        print(f"  TE(S->A|H) = {teB_nc:.6f}")
        print(f"  Delta_TE    = {d_nc:.6f}")

    finally:
        shutdown_jvm()


if __name__ == '__main__':
    main()

