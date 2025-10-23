"""Reproduce transfer entropy analyses for the ExtraSensory dataset."""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy
from scipy import stats
import yaml


QuantileEdges = Dict[str, List[float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ExtraSensory transfer entropy reproduction.")
    parser.add_argument("--data-root", type=Path, required=True, help="Directory with per-user CSV files.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory to store outputs.")
    parser.add_argument("--surrogates", type=int, default=1000, help="Number of surrogate shifts.")
    parser.add_argument("--seed", type=int, default=1729, help="Random seed for surrogates.")
    parser.add_argument("--tau", type=int, default=1, help="Lag in minutes.")
    parser.add_argument("--histories", type=int, nargs="+", default=[1, 2, 3, 4], help="History lengths k=l to sweep.")
    return parser.parse_args()


def detect_timestamp_unit(ts: pd.Series) -> str:
    max_ts = ts.max()
    if pd.isna(max_ts):
        return "s"
    return "ms" if max_ts > 1e12 else "s"


def majority_vote(values: pd.Series) -> Optional[int]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    binary = numeric[(numeric == 0) | (numeric == 1)]
    if binary.empty:
        return None
    counts = binary.value_counts()
    return int(counts.idxmax())


def compute_z_scores(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mean) / std


def compute_quintile_edges(z_scores: pd.Series) -> List[float]:
    quantiles = z_scores.quantile(np.linspace(0.0, 1.0, 6), interpolation="linear").tolist()
    return [float(x) if np.isfinite(x) else float("nan") for x in quantiles]


def assign_quintile_bins(z_scores: pd.Series) -> np.ndarray:
    values = z_scores.to_numpy()
    if values.size == 0:
        return np.array([], dtype=np.int32)
    edges = np.array(compute_quintile_edges(z_scores), dtype=np.float64)
    if np.isnan(edges).any():
        edges = np.nan_to_num(edges, nan=np.nanmin(values))
    edges[0] = np.min([edges[0], values.min()])
    edges[-1] = np.max([edges[-1], values.max()])
    for idx in range(1, len(edges)):
        if not np.isfinite(edges[idx]):
            edges[idx] = edges[idx - 1]
        if edges[idx] <= edges[idx - 1]:
            edges[idx] = np.nextafter(edges[idx - 1], np.float64("inf"))
    interior = edges[1:-1]
    bins = np.digitize(values, interior, right=False)
    return bins.astype(np.int32)


def preprocess_user(file_path: Path) -> Tuple[pd.DataFrame, List[float]]:
    df = pd.read_csv(
        file_path,
        usecols=["timestamp", "raw_acc:magnitude_stats:mean", "label:SITTING"],
    )
    ts_unit = detect_timestamp_unit(df["timestamp"])
    df["time"] = (
        pd.to_datetime(df["timestamp"], unit=ts_unit, utc=True)
        .dt.tz_convert(None)
        .dt.floor("min")
    )
    df = df.dropna(subset=["time"])
    df = df.sort_values("time")
    aggregated = (
        df.groupby("time")
        .agg(
            {
                "raw_acc:magnitude_stats:mean": "mean",
                "label:SITTING": majority_vote,
            }
        )
        .rename(columns={"raw_acc:magnitude_stats:mean": "A_raw", "label:SITTING": "S"})
    )
    aggregated = aggregated.dropna(subset=["A_raw", "S"])
    aggregated["S"] = aggregated["S"].astype(int)
    aggregated["A_z"] = compute_z_scores(aggregated["A_raw"])
    quintile_edges = compute_quintile_edges(aggregated["A_z"])
    aggregated["A_bin"] = assign_quintile_bins(aggregated["A_z"])
    aggregated["hour"] = aggregated.index.hour
    aggregated["hour6"] = (aggregated["hour"] // 4).astype(int)
    aggregated["hour24"] = aggregated["hour"].astype(int)
    aggregated = aggregated.reset_index()
    aggregated = aggregated.rename(columns={"time": "minute"})
    return aggregated, quintile_edges


@dataclass
class DirectionSetup:
    dest_future: np.ndarray
    dest_past_codes: np.ndarray
    source_codes: np.ndarray
    cond_values: Optional[np.ndarray]
    time_indices: np.ndarray
    source_series: np.ndarray
    history: int
    tau: int
    dest_base: int
    source_base: int
    cond_base: Optional[int]
    dest_past_states: int
    source_past_states: int
    n_samples: int


def compute_past_codes(
    series: np.ndarray, history: int, time_indices: np.ndarray, base: int
) -> np.ndarray:
    if history <= 0:
        return np.zeros(len(time_indices), dtype=np.int64)
    codes = np.zeros(len(time_indices), dtype=np.int64)
    for offset in range(history):
        shift = history - 1 - offset
        codes = codes * base + series[time_indices - shift]
    return codes


def prepare_direction_setup(
    dest: np.ndarray,
    source: np.ndarray,
    cond: Optional[np.ndarray],
    history: int,
    tau: int,
    dest_base: int,
    source_base: int,
    cond_base: Optional[int],
) -> DirectionSetup:
    n = len(dest)
    if len(source) != n:
        raise ValueError("Source and destination series must share length.")
    if cond is not None and len(cond) != n:
        raise ValueError("Conditioning series must match length.")
    start = max(history, 1) - 1
    end = n - tau
    if end <= start:
        empty = np.array([], dtype=np.int64)
        return DirectionSetup(
            dest_future=empty,
            dest_past_codes=empty,
            source_codes=empty,
            cond_values=empty if cond is not None else None,
            time_indices=empty,
            source_series=source,
            history=history,
            tau=tau,
            dest_base=dest_base,
            source_base=source_base,
            cond_base=cond_base,
            dest_past_states=dest_base**history,
            source_past_states=source_base**history,
            n_samples=0,
        )
    time_idx = np.arange(start, end, dtype=np.int64)
    dest_future = dest[time_idx + tau]
    dest_past = compute_past_codes(dest, history, time_idx, dest_base)
    source_codes = compute_past_codes(source, history, time_idx, source_base)
    cond_values = cond[time_idx] if cond is not None else None
    return DirectionSetup(
        dest_future=dest_future,
        dest_past_codes=dest_past,
        source_codes=source_codes,
        cond_values=cond_values,
        time_indices=time_idx,
        source_series=source,
        history=history,
        tau=tau,
        dest_base=dest_base,
        source_base=source_base,
        cond_base=cond_base,
        dest_past_states=dest_base**history,
        source_past_states=source_base**history,
        n_samples=len(dest_future),
    )


def compute_te_from_setup(setup: DirectionSetup, source_codes: Optional[np.ndarray] = None) -> float:
    if setup.n_samples == 0:
        return float("nan")
    src = setup.source_codes if source_codes is None else source_codes
    dims = (setup.dest_base, setup.dest_past_states, setup.source_past_states)
    joint_index = np.ravel_multi_index(
        (setup.dest_future, setup.dest_past_codes, src), dims=dims
    )
    joint_counts = np.bincount(
        joint_index, minlength=np.prod(dims, dtype=np.int64)
    ).reshape(dims)
    total = float(setup.n_samples)
    dest_source_counts = joint_counts.sum(axis=0)
    dest_future_counts = joint_counts.sum(axis=2)
    dest_past_counts = dest_future_counts.sum(axis=0)
    nz = joint_counts.nonzero()
    if len(nz[0]) == 0:
        return 0.0
    counts = joint_counts[nz].astype(np.float64)
    ds_counts = dest_source_counts[nz[1], nz[2]].astype(np.float64)
    df_counts = dest_future_counts[nz[0], nz[1]].astype(np.float64)
    dp_counts = dest_past_counts[nz[1]].astype(np.float64)
    log_terms = np.log2(counts) + np.log2(dp_counts) - np.log2(ds_counts) - np.log2(df_counts)
    return float(np.sum((counts / total) * log_terms))


def compute_cte_from_setup(
    setup: DirectionSetup,
    source_codes: Optional[np.ndarray] = None,
) -> float:
    if setup.cond_values is None or setup.cond_base is None:
        raise ValueError("Conditioning values are required for conditional TE.")
    if setup.n_samples == 0:
        return float("nan")
    src = setup.source_codes if source_codes is None else source_codes
    dims = (
        setup.dest_base,
        setup.dest_past_states,
        setup.source_past_states,
        setup.cond_base,
    )
    joint_index = np.ravel_multi_index(
        (setup.dest_future, setup.dest_past_codes, src, setup.cond_values),
        dims=dims,
    )
    joint_counts = np.bincount(
        joint_index, minlength=np.prod(dims, dtype=np.int64)
    ).reshape(dims)
    total = float(setup.n_samples)
    dest_source_cond = joint_counts.sum(axis=0)
    dest_future_cond = joint_counts.sum(axis=2)
    dest_cond = dest_future_cond.sum(axis=0)
    nz = joint_counts.nonzero()
    if len(nz[0]) == 0:
        return 0.0
    counts = joint_counts[nz].astype(np.float64)
    dsc_counts = dest_source_cond[nz[1], nz[2], nz[3]].astype(np.float64)
    dfc_counts = dest_future_cond[nz[0], nz[1], nz[3]].astype(np.float64)
    dc_counts = dest_cond[nz[1], nz[3]].astype(np.float64)
    log_terms = np.log2(counts) + np.log2(dc_counts) - np.log2(dsc_counts) - np.log2(dfc_counts)
    return float(np.sum((counts / total) * log_terms))


def generate_offsets(rng: np.random.Generator, series_length: int, tau: int, num: int) -> np.ndarray:
    if series_length <= tau:
        return np.array([], dtype=int)
    return rng.integers(low=tau, high=series_length, size=num, dtype=np.int64)


def compute_surrogate_codes(
    setup: DirectionSetup,
    offsets: Sequence[int],
) -> Iterable[np.ndarray]:
    for offset in offsets:
        shifted = np.roll(setup.source_series, int(offset))
        yield compute_past_codes(shifted, setup.history, setup.time_indices, setup.source_base)


def summarise_delta(values: pd.Series) -> Dict[str, float]:
    cleaned = values.dropna()
    n = len(cleaned)
    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "sd": float("nan"),
            "se": float("nan"),
            "ci95_lower": float("nan"),
            "ci95_upper": float("nan"),
        }
    mean = cleaned.mean()
    sd = cleaned.std(ddof=1)
    se = sd / math.sqrt(n) if n > 0 else float("nan")
    margin = 1.96 * se if np.isfinite(se) else float("nan")
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci95_lower": mean - margin if np.isfinite(margin) else float("nan"),
        "ci95_upper": mean + margin if np.isfinite(margin) else float("nan"),
    }


def hodges_lehmann(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    pair_avgs = []
    for i, xi in enumerate(values):
        for xj in values[i:]:
            pair_avgs.append(0.5 * (xi + xj))
    return float(np.median(pair_avgs))


def bootstrap_ci(
    values: np.ndarray,
    func,
    alpha: float = 0.05,
    n_resamples: int = 5000,
    seed: int = 1729,
) -> Tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    result = stats.bootstrap(
        (values,),
        func,
        vectorized=False,
        paired=False,
        confidence_level=1 - alpha,
        n_resamples=n_resamples,
        random_state=rng,
        method="BCa",
    )
    return (float(result.confidence_interval.low), float(result.confidence_interval.high))


def benjamini_hochberg(p_values: Dict[str, float]) -> Dict[str, float]:
    items = [(key, p) for key, p in p_values.items() if np.isfinite(p)]
    m = len(items)
    if m == 0:
        return {key: float("nan") for key in p_values}
    items.sort(key=lambda kv: kv[1])
    adjusted: Dict[str, float] = {}
    prev = 0.0
    for rank, (key, p) in enumerate(items, start=1):
        adj = min(p * m / rank, 1.0)
        adj = max(adj, prev)
        adjusted[key] = adj
        prev = adj
    for key in p_values:
        adjusted.setdefault(key, float("nan"))
    return adjusted


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(data_root.glob("*.features_labels.csv"))
    if not files:
        raise FileNotFoundError(f"No per-user files found in {data_root}")

    user_frames: Dict[str, pd.DataFrame] = {}
    bin_edges: QuantileEdges = {}
    for path in files:
        user_id = path.name.replace(".features_labels.csv", "")
        frame, edges = preprocess_user(path)
        user_frames[user_id] = frame
        bin_edges[user_id] = edges

    user_sample_counts = {uid: len(df) for uid, df in user_frames.items()}
    avg_per_bin_24 = np.mean([count / 24 for count in user_sample_counts.values()])
    avg_per_bin_6 = np.mean([count / 6 for count in user_sample_counts.values()])
    if avg_per_bin_6 >= 200:
        hour_bins = 6
        hour_key = "hour6"
    elif avg_per_bin_24 >= 200:
        hour_bins = 24
        hour_key = "hour24"
    else:
        hour_bins = 6 if avg_per_bin_6 > avg_per_bin_24 else 24
        hour_key = "hour6" if hour_bins == 6 else "hour24"

    rng = np.random.default_rng(args.seed)

    te_records = []
    cte_records = []

    for user_id, frame in user_frames.items():
        A = frame["A_bin"].to_numpy(dtype=np.int64)
        S = frame["S"].to_numpy(dtype=np.int64)
        H = frame[hour_key].to_numpy(dtype=np.int64)
        base_A = int(A.max()) + 1 if len(A) else 5
        base_S = int(S.max()) + 1 if len(S) else 2

        for history in args.histories:
            setup_a_to_s = prepare_direction_setup(
                dest=S,
                source=A,
                cond=H,
                history=history,
                tau=args.tau,
                dest_base=base_S,
                source_base=base_A,
                cond_base=hour_bins,
            )
            setup_s_to_a = prepare_direction_setup(
                dest=A,
                source=S,
                cond=H,
                history=history,
                tau=args.tau,
                dest_base=base_A,
                source_base=base_S,
                cond_base=hour_bins,
            )

            offsets_a_to_s = generate_offsets(rng, len(A), args.tau, args.surrogates)
            offsets_s_to_a = generate_offsets(rng, len(S), args.tau, args.surrogates)

            te_a2s = compute_te_from_setup(setup_a_to_s)
            te_s2a = compute_te_from_setup(setup_s_to_a)

            te_surr_a2s = np.array(
                [compute_te_from_setup(setup_a_to_s, codes) for codes in compute_surrogate_codes(setup_a_to_s, offsets_a_to_s)]
            )
            te_surr_s2a = np.array(
                [compute_te_from_setup(setup_s_to_a, codes) for codes in compute_surrogate_codes(setup_s_to_a, offsets_s_to_a)]
            )

            p_a2s = (
                (1.0 + np.sum(te_surr_a2s >= te_a2s))
                / (len(te_surr_a2s) + 1.0)
                if te_surr_a2s.size
                else float("nan")
            )
            p_s2a = (
                (1.0 + np.sum(te_surr_s2a >= te_s2a))
                / (len(te_surr_s2a) + 1.0)
                if te_surr_s2a.size
                else float("nan")
            )

            te_records.append(
                {
                    "user_id": user_id,
                    "k": history,
                    "l": history,
                    "tau": args.tau,
                    "TE_A2S": te_a2s,
                    "TE_S2A": te_s2a,
                    "Delta_TE": te_a2s - te_s2a,
                    "p_A2S": p_a2s,
                    "p_S2A": p_s2a,
                    "n_samples": setup_a_to_s.n_samples,
                    "low_n": setup_a_to_s.n_samples < 100,
                }
            )

            if setup_a_to_s.cond_values is not None:
                cte_a2s = compute_cte_from_setup(setup_a_to_s)
                cte_s2a = compute_cte_from_setup(setup_s_to_a)
                cte_surr_a2s = np.array(
                    [
                        compute_cte_from_setup(setup_a_to_s, codes)
                        for codes in compute_surrogate_codes(setup_a_to_s, offsets_a_to_s)
                    ]
                )
                cte_surr_s2a = np.array(
                    [
                        compute_cte_from_setup(setup_s_to_a, codes)
                        for codes in compute_surrogate_codes(setup_s_to_a, offsets_s_to_a)
                    ]
                )
                p_cte_a2s = (
                    (1.0 + np.sum(cte_surr_a2s >= cte_a2s))
                    / (len(cte_surr_a2s) + 1.0)
                    if cte_surr_a2s.size
                    else float("nan")
                )
                p_cte_s2a = (
                    (1.0 + np.sum(cte_surr_s2a >= cte_s2a))
                    / (len(cte_surr_s2a) + 1.0)
                    if cte_surr_s2a.size
                    else float("nan")
                )
                cond_counts = (
                    np.bincount(setup_a_to_s.cond_values, minlength=hour_bins)
                    if setup_a_to_s.cond_values is not None
                    else np.array([])
                )
                min_per_bin = int(cond_counts.min()) if cond_counts.size else 0
                cte_records.append(
                    {
                        "user_id": user_id,
                        "k": history,
                        "l": history,
                        "tau": args.tau,
                        "CTE_A2S_H": cte_a2s,
                        "CTE_S2A_H": cte_s2a,
                        "Delta_CTE": cte_a2s - cte_s2a,
                        "p_A2S_H": p_cte_a2s,
                        "p_S2A_H": p_cte_s2a,
                        "n_samples": setup_a_to_s.n_samples,
                        "n_samples_per_bin_min": min_per_bin,
                        "low_n": setup_a_to_s.n_samples < 100,
                    }
                )

    te_df = pd.DataFrame(te_records)
    cte_df = pd.DataFrame(cte_records)

    te_csv = out_dir / "per_user_te.csv"
    cte_csv = out_dir / "per_user_cte.csv"
    te_df.to_csv(te_csv, index=False)
    cte_df.to_csv(cte_csv, index=False)

    te_summary_records = []
    for k, group in te_df.groupby("k"):
        stats_dict = summarise_delta(group["Delta_TE"])
        te_summary_records.append({"k": k, **stats_dict})
    te_summary_df = pd.DataFrame(te_summary_records).sort_values("k")
    te_summary_path = out_dir / "k_sweep_summary.csv"
    te_summary_df.to_csv(te_summary_path, index=False)

    cte_summary_records = []
    for k, group in cte_df.groupby("k"):
        stats_dict = summarise_delta(group["Delta_CTE"])
        cte_summary_records.append({"k": k, **stats_dict})
    cte_summary_df = pd.DataFrame(cte_summary_records).sort_values("k")
    cte_summary_path = out_dir / "cte_summary.csv"
    cte_summary_df.to_csv(cte_summary_path, index=False)

    plt.figure(figsize=(6, 4))
    plt.errorbar(
        te_summary_df["k"],
        te_summary_df["mean"],
        yerr=te_summary_df["mean"] - te_summary_df["ci95_lower"],
        fmt="o-",
        capsize=4,
    )
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("History length k")
    plt.ylabel("ΔTE (bits)")
    plt.title("Delta TE across history lengths")
    plt.tight_layout()
    fig_delta_te = out_dir / "fig_deltaTE_by_k.png"
    plt.savefig(fig_delta_te, dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.violinplot(data=cte_df, x="k", y="Delta_CTE", inner="box", cut=0)
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("History length k")
    plt.ylabel("ΔCTE (bits)")
    plt.title("Conditional ΔTE by history length")
    plt.tight_layout()
    fig_delta_cte = out_dir / "fig_deltaCTE_violin.png"
    plt.savefig(fig_delta_cte, dpi=300)
    plt.close()

    def paired_plot(df: pd.DataFrame, col_x: str, col_y: str, title: str, path: Path) -> None:
        plt.figure(figsize=(6, 6))
        xs = [0, 1]
        for _, row in df.iterrows():
            plt.plot(xs, [row[col_x], row[col_y]], marker="o", color="tab:blue", alpha=0.6)
        plt.xticks(xs, ["A→S", "S→A"])
        plt.ylabel("Bits")
        plt.title(title)
        plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()

    te_k4 = te_df[te_df["k"] == 4]
    cte_k4 = cte_df[cte_df["k"] == 4]
    fig_te_pair = out_dir / "fig_pair_TE_k4.png"
    fig_cte_pair = out_dir / "fig_pair_CTE_k4.png"
    if not te_k4.empty:
        paired_plot(te_k4, "TE_A2S", "TE_S2A", "Paired TE at k=4", fig_te_pair)
    if not cte_k4.empty:
        paired_plot(cte_k4, "CTE_A2S_H", "CTE_S2A_H", "Paired CTE at k=4", fig_cte_pair)

    delta_te_k4 = te_k4["Delta_TE"].dropna().to_numpy()
    delta_cte_k4 = cte_k4["Delta_CTE"].dropna().to_numpy()

    def wilcoxon_summary(values: np.ndarray) -> Dict[str, float]:
        if values.size == 0:
            return {
                "n": 0,
                "median": float("nan"),
                "hl": float("nan"),
                "hl_ci_lower": float("nan"),
                "hl_ci_upper": float("nan"),
                "W": float("nan"),
                "p_two_sided": float("nan"),
                "p_one_sided": float("nan"),
                "r_effect": float("nan"),
            }
        res_two = stats.wilcoxon(values, alternative="two-sided", zero_method="wilcox", mode="auto")
        res_less = stats.wilcoxon(values, alternative="less", zero_method="wilcox", mode="auto")
        median = float(np.median(values))
        hl = hodges_lehmann(values)
        ci_lower, ci_upper = bootstrap_ci(values, hodges_lehmann)
        n = len(values)
        if np.isfinite(res_two.pvalue) and res_two.pvalue > 0:
            z = stats.norm.isf(res_two.pvalue / 2.0)
            z_signed = -abs(z) if median < 0 else abs(z)
            r_effect = z_signed / math.sqrt(n)
        else:
            r_effect = float("nan")
        return {
            "n": n,
            "median": median,
            "hl": hl,
            "hl_ci_lower": ci_lower,
            "hl_ci_upper": ci_upper,
            "W": float(res_two.statistic),
            "p_two_sided": float(res_two.pvalue),
            "p_one_sided": float(res_less.pvalue),
            "r_effect": r_effect,
        }

    te_stats = wilcoxon_summary(delta_te_k4)
    cte_stats = wilcoxon_summary(delta_cte_k4)
    fdr_adjusted = benjamini_hochberg(
        {"Delta_TE": te_stats["p_two_sided"], "Delta_CTE": cte_stats["p_two_sided"]}
    )

    def fraction_negative(df: pd.DataFrame, column: str) -> Dict[int, float]:
        output = {}
        for k, group in df.groupby("k"):
            valid = group[column].dropna()
            output[int(k)] = float((valid < 0).mean()) if len(valid) else float("nan")
        return output

    frac_te = fraction_negative(te_df, "Delta_TE")
    frac_cte = fraction_negative(cte_df, "Delta_CTE")

    low_n_users = sorted(
        {
            row["user_id"]
            for _, row in te_df[te_df["k"] == 4].iterrows()
            if row["low_n"]
        }
    )

    group_stats_path = out_dir / "group_stats.md"
    with group_stats_path.open("w", encoding="utf-8") as fh:
        fh.write("# Group-level Statistics (k = 4, τ = 1)\n\n")
        fh.write("## Transfer Entropy (A→S vs S→A)\n")
        fh.write(f"- N = {te_stats['n']}\n")
        fh.write(f"- Median ΔTE = {te_stats['median']:.6f} bits\n")
        fh.write(f"- Hodges–Lehmann ΔTE = {te_stats['hl']:.6f} bits\n")
        fh.write(
            f"- 95% CI (bootstrap) = [{te_stats['hl_ci_lower']:.6f}, {te_stats['hl_ci_upper']:.6f}] bits\n"
        )
        fh.write(f"- Wilcoxon W = {te_stats['W']:.4f}\n")
        fh.write(f"- Two-sided p = {te_stats['p_two_sided']:.6g}\n")
        fh.write(f"- One-sided p (ΔTE < 0) = {te_stats['p_one_sided']:.6g}\n")
        fh.write(f"- Effect size r = {te_stats['r_effect']:.4f}\n")
        fh.write(f"- BH-FDR adjusted p = {fdr_adjusted['Delta_TE']:.6g}\n\n")

        fh.write("## Conditional Transfer Entropy (A→S|H vs S→A|H)\n")
        fh.write(f"- N = {cte_stats['n']}\n")
        fh.write(f"- Median ΔCTE = {cte_stats['median']:.6f} bits\n")
        fh.write(f"- Hodges–Lehmann ΔCTE = {cte_stats['hl']:.6f} bits\n")
        fh.write(
            f"- 95% CI (bootstrap) = [{cte_stats['hl_ci_lower']:.6f}, {cte_stats['hl_ci_upper']:.6f}] bits\n"
        )
        fh.write(f"- Wilcoxon W = {cte_stats['W']:.4f}\n")
        fh.write(f"- Two-sided p = {cte_stats['p_two_sided']:.6g}\n")
        fh.write(f"- One-sided p (ΔCTE < 0) = {cte_stats['p_one_sided']:.6g}\n")
        fh.write(f"- Effect size r = {cte_stats['r_effect']:.4f}\n")
        fh.write(f"- BH-FDR adjusted p = {fdr_adjusted['Delta_CTE']:.6g}\n\n")

        fh.write("## Robustness across k (fraction of users with negative delta)\n")
        fh.write("- ΔTE: " + ", ".join(f"k={k}: {frac:.3f}" for k, frac in sorted(frac_te.items())) + "\n")
        fh.write(
            "- ΔCTE: " + ", ".join(f"k={k}: {frac:.3f}" for k, frac in sorted(frac_cte.items())) + "\n\n"
        )
        fh.write("## Users with n_samples < 100 at k=4\n")
        if low_n_users:
            for uid in low_n_users:
                fh.write(f"- {uid}\n")
        else:
            fh.write("- None\n")

    bin_path = out_dir / "bin_edges.json"
    with bin_path.open("w", encoding="utf-8") as fh:
        json.dump(bin_edges, fh, indent=2)

    run_info = {
        "data_root": str(data_root),
        "out_dir": str(out_dir),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "hour_bins": hour_bins,
        "hour_bins_choice": {
            "avg_per_bin_24": float(avg_per_bin_24),
            "avg_per_bin_6": float(avg_per_bin_6),
        },
        "histories": args.histories,
        "tau": args.tau,
        "num_surrogates": args.surrogates,
        "surrogate_seed": args.seed,
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": plt.matplotlib.__version__,
            "seaborn": sns.__version__,
            "pyyaml": yaml.__version__,
            "os": platform.platform(),
        },
    }
    run_info_path = out_dir / "run_info.yaml"
    with run_info_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(run_info, fh, sort_keys=False)


if __name__ == "__main__":
    main()
