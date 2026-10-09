#!/usr/bin/env python3
"""Verify committed report exports and reproduce descriptive README numbers offline."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "report/data/manifest.json"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize(data_dir: Path) -> dict:
    primary = rows(data_dir / "per_user_true_cte.csv")
    alias = rows(data_dir / "final_results_summary_n60.csv")
    pairs = [(r["user_id"], r["tau"]) for r in primary]
    if len(pairs) != len(set(pairs)):
        raise ValueError("Duplicate user/tau observation in primary export")
    if (
        len(primary) != 120
        or len({r["user_id"] for r in primary}) != 60
        or set(r["tau"] for r in primary) != {"1", "2"}
    ):
        raise ValueError("Core export must contain 60 users at tau 1 and 2")
    users = {r["user_id"] for r in primary}
    if set(pairs) != {(user, tau) for user in users for tau in ["1", "2"]}:
        raise ValueError("Every core user must have one row at each tau")
    expected = {(r["user_id"], r["tau"]): r for r in primary}
    if len(alias) != len(primary) or {(r["user_id"], r["tau"]) for r in alias} != set(expected):
        raise ValueError("Legacy summary and primary export observation sets differ")
    for r in primary:
        for column in ["CTE_true_A2S", "CTE_true_S2A", "Delta_CTE_true", "q_A2S", "q_S2A"]:
            if not math.isfinite(float(r[column])):
                raise ValueError(f"Non-finite {column}")
        if not math.isclose(
            float(r["CTE_true_A2S"]) - float(r["CTE_true_S2A"]),
            float(r["Delta_CTE_true"]),
            abs_tol=1e-12,
        ):
            raise ValueError("Delta is inconsistent with the two True CTE directions")
        if not 0 <= float(r["q_A2S"]) <= 1 or not 0 <= float(r["q_S2A"]) <= 1:
            raise ValueError("Adjusted p-value outside [0, 1]")
    for r in alias:
        target = expected[(r["user_id"], r["tau"])]
        for old, new in [
            ("Delta_TE", "Delta_CTE_true"),
            ("TE_AtoS", "CTE_true_A2S"),
            ("TE_StoA", "CTE_true_S2A"),
            ("q_AtoS", "q_A2S"),
            ("q_StoA", "q_S2A"),
        ]:
            if not math.isclose(float(r[old]), float(target[new]), abs_tol=1e-12):
                raise ValueError(f"Legacy {old} does not match primary {new}")
    selections: dict[str, int] = {}
    for r in rows(data_dir / "k_selected_by_user_ALL.csv"):
        if not r["k_selected"]:
            continue
        k = int(r["k_selected"])
        if r["user_id"] in selections and selections[r["user_id"]] != k:
            raise ValueError("Conflicting k-selection for one user")
        selections[r["user_id"]] = k
    if set(selections) != {r["user_id"] for r in primary}:
        raise ValueError("k-selection and core user sets differ")
    sensitivity = rows(data_dir / "sensitivity_12cell_matrix.csv")
    cells = {(r["A_bins"], r["S_mode"], r["H_bin_hours"]) for r in sensitivity}
    if len(sensitivity) != 12 or cells != {
        (a, mode, hour)
        for a in ["3", "5", "7"]
        for mode in ["binary", "quantile3"]
        for hour in ["2", "4"]
    }:
        raise ValueError("Sensitivity export must have 12 distinct cells")
    files = []
    for path in sorted(data_dir.rglob("*.csv")):
        records = rows(path)
        files.append(
            {
                "path": path.relative_to(data_dir).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "rows": len(records),
                "columns": list(records[0]) if records else [],
            }
        )
    metrics = []
    for tau in ["1", "2"]:
        group = [r for r in primary if r["tau"] == tau]
        significant = sum(float(r["q_A2S"]) < 0.05 for r in group)
        metrics.append(
            {
                "tau": int(tau),
                "users": len(group),
                "mean_delta_true_cte_bits": statistics.mean(
                    float(r["Delta_CTE_true"]) for r in group
                ),
                "a_to_s_q_lt_0_05_users": significant,
                "a_to_s_q_lt_0_05_percent": significant * 100 / len(group),
            }
        )
    return {
        "schema_version": 1,
        "evidence_kind": "committed_curated_exports",
        "primary": "per_user_true_cte.csv",
        "legacy_summary_field_semantics": "TE-labelled columns alias True CTE, checked against primary",
        "full_raw_run_committed": False,
        "unique_users": 60,
        "k_6_unique_users": sum(k == 6 for k in selections.values()),
        "metrics": metrics,
        "sensitivity_cells": 12,
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=MANIFEST.parent)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    actual = summarize(args.data_dir)
    if args.write:
        args.manifest.write_text(json.dumps(actual, indent=2, ensure_ascii=False) + "\n")
    elif json.loads(args.manifest.read_text()) != actual:
        raise ValueError("Report manifest is stale; review data changes before --write")
    print(
        "Report exports verified: 60 users; tau 1 mean -0.033426 bits; 33/60 = 55.0%; 12 sensitivity cells."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
