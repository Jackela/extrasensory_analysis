#!/usr/bin/env python
"""
Appendix runner: build OOM user list, create high-RAM TE-only config,
run the appendix pipeline, and compare ΔTE means vs True CTE.

Steps
- Read FINAL_RUN per_user_te.csv → tau==1 & TE_A2S is NaN (OOM users)
- Take first 10 user_ids → write config/appendix_user_list.txt
- Write config/presets/appendix_k6_high_ram.yaml (if missing)
- Execute: python run_production.py --config config/presets/appendix_k6_high_ram.yaml --workers 3 --no-progress
- After completion, gather per_user_te.csv from shard outputs and compare
  ΔTE mean sign with True CTE ΔTE for the same 10 users from FINAL_RUN

Usage
  python tools/appendix_run.py            # do all steps + run + report
  python tools/appendix_run.py --prepare  # only build user list + config

Note: This script runs a real analysis; ensure data and JIDT are installed.
"""
import sys
import time
import subprocess
from pathlib import Path
from typing import List
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
FINAL_RUN = REPO / 'analysis/out/FINAL_RUN_k60_COMPLETE'
FINAL_TE = FINAL_RUN / 'per_user_te.csv'
FINAL_TRUE_CTE = FINAL_RUN / 'per_user_true_cte.csv'
USER_LIST_TXT = REPO / 'config/appendix_user_list.txt'
APPENDIX_YAML = REPO / 'config/presets/appendix_k6_high_ram.yaml'


def build_user_list(limit: int = 10) -> List[str]:
    df = pd.read_csv(FINAL_TE)
    df1 = df[df['tau'] == 1]
    subset = df1[df1['TE_A2S'].isna()]
    users = subset['user_id'].drop_duplicates().tolist()
    if len(users) < limit:
        print(f"WARNING: Found only {len(users)} OOM users at tau==1; expected ~44")
    pick = users[:limit]
    USER_LIST_TXT.parent.mkdir(parents=True, exist_ok=True)
    USER_LIST_TXT.write_text("\n".join(pick) + "\n", encoding='utf-8')
    print(f"✓ Wrote {len(pick)} user_ids → {USER_LIST_TXT}")
    return pick


def ensure_appendix_yaml():
    if APPENDIX_YAML.exists():
        print(f"Config already exists: {APPENDIX_YAML}")
        return
    # Content mirrors preset we ship in the repo; keep in sync if needed
    cfg = {
        'data_root': 'data/ExtraSensory.per_uuid_features_labels',
        'out_dir': 'analysis/out/appendix_k6_high_ram_<STAMP>',
        'analysis_modes': ['global_te'],
        'user_list_file': str(USER_LIST_TXT.relative_to(REPO)),
        'feature_modes': ['composite'],
        'hour_bins': 6,
        'taus': [1, 2],
        'k_selection': {
            'strategy': 'AIS',
            'k_grid': [1, 2, 3, 4, 5, 6],
            'k_max': 6,
        },
        'surrogates': 1000,
        'statistical': { 'adaptive_surrogates': { 'enabled': False }},
        'fdr': { 'families': ['TE'], 'by_tau': True, 'alpha': 0.05 },
        'jvm': { 'xms': '8g', 'xmx': '16g' },
        'runtime': { 'concurrency': 3 },
    }
    APPENDIX_YAML.parent.mkdir(parents=True, exist_ok=True)
    APPENDIX_YAML.write_text(yaml.dump(cfg, sort_keys=False), encoding='utf-8')
    print(f"✓ Wrote config → {APPENDIX_YAML}")


def run_appendix():
    cmd = [
        sys.executable,
        'run_production.py',
        '--config', str(APPENDIX_YAML.relative_to(REPO)),
        '--workers', '3',
        '--no-progress'
    ]
    print("\n→ Running appendix pipeline:\n  ", ' '.join(cmd))
    subprocess.check_call(cmd, cwd=str(REPO))


def find_latest_appendix_dirs() -> List[Path]:
    out_glob = REPO / 'analysis/out'
    candidates = sorted(out_glob.glob('appendix_k6_high_ram_*'))
    if not candidates:
        return []
    # Heuristic: take the most recent timestamp prefix (multiple shards)
    latest_mtime = max(p.stat().st_mtime for p in candidates)
    # Allow a 2-hour window to capture all shards from same batch
    return [p for p in candidates if (latest_mtime - p.stat().st_mtime) < 2*3600]


def load_merged_te(dirs: List[Path]) -> pd.DataFrame:
    frames = []
    for d in dirs:
        f = d / 'per_user_te.csv'
        if f.exists():
            frames.append(pd.read_csv(f))
    if not frames:
        raise FileNotFoundError("No per_user_te.csv found in appendix outputs")
    return pd.concat(frames, ignore_index=True)


def compare_delta_te(user_ids: List[str]):
    appendix_dirs = find_latest_appendix_dirs()
    if not appendix_dirs:
        raise RuntimeError("Could not find appendix output directories")
    df_te = load_merged_te(appendix_dirs)
    df_te = df_te[df_te['user_id'].isin(user_ids)]
    mean_te = float(pd.to_numeric(df_te['Delta_TE'], errors='coerce').dropna().mean())

    df_true = pd.read_csv(FINAL_TRUE_CTE)
    df_true = df_true[df_true['user_id'].isin(user_ids)]
    mean_true = float(pd.to_numeric(df_true['Delta_CTE_true'], errors='coerce').dropna().mean())

    print("\nAppendix comparison (10 users, k≥5 OOM subset):")
    print(f"  Global TE ΔTE mean (high RAM): {mean_te:+.6f}")
    print(f"  True CTE ΔTE mean (FINAL RUN): {mean_true:+.6f}")
    same_direction_negative = (mean_te < 0) and (mean_true < 0)
    print(f"  Direction consistent and negative? {'YES' if same_direction_negative else 'NO'}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepare', action='store_true', help='Only prepare user list and config, do not run')
    args = ap.parse_args()

    if not FINAL_TE.exists():
        raise FileNotFoundError(f"Missing final TE file: {FINAL_TE}")
    if not FINAL_TRUE_CTE.exists():
        print(f"WARNING: Missing final True CTE file: {FINAL_TRUE_CTE}")

    users = build_user_list(limit=10)
    ensure_appendix_yaml()
    if args.prepare:
        return
    run_appendix()
    compare_delta_te(users)


if __name__ == '__main__':
    main()

