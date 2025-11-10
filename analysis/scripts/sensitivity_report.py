"""
Aggregate sensitivity grid results into a 12-row matrix and display.

Finds the latest run under analysis/out/sensitivity/<STAMP>/ and loads
each combo's summary.csv, then composes a tidy table with columns:
  A_bins, H_bin_hours, S_mode, N, mean_Delta_TE, median_Delta_TE,
  ci95_lower, ci95_upper, t_stat, p_value

If some combos are missing, they will appear absent; use this after a
full run finishes to verify all 12 combinations are present.
"""
from pathlib import Path
import pandas as pd
import sys


def parse_combo_name(name: str):
    # Example: A5_Squantile3_H4h
    parts = name.split('_')
    a = int(parts[0][1:])
    s = parts[1][1:]
    h = int(parts[2][1:-1])
    return a, s, h


def latest_run_dir(root: Path) -> Path:
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No runs under {root}")
    # Sort by name which is timestamp-like
    subdirs.sort(key=lambda p: p.name)
    return subdirs[-1]


def main() -> int:
    root = Path("analysis/out/sensitivity")
    run_dir = latest_run_dir(root)
    rows = []
    for combo_dir in sorted([p for p in run_dir.iterdir() if p.is_dir()]):
        s = combo_dir / 'summary.csv'
        if not s.exists():
            continue
        df = pd.read_csv(s)
        if df.empty:
            continue
        r = df.iloc[0].to_dict()
        combo = r['combo']
        A_bins, S_mode, H_bin_hours = parse_combo_name(combo)

        def fget_any(*keys):
            for key in keys:
                v = r.get(key)
                if v is None or v == '' or pd.isna(v):
                    continue
                try:
                    return float(v)
                except Exception:
                    continue
            return None

        def fget(key):
            v = r.get(key)
            if v == '' or pd.isna(v):
                return None
            try:
                return float(v)
            except Exception:
                return None

        rows.append({
            'A_bins': A_bins,
            'H_bin_hours': H_bin_hours,
            'S_mode': S_mode,
            'N': int(r.get('N', 0)) if str(r.get('N','')).isdigit() else 0,
            # Prefer True CTE metrics when present
            'mean_Delta_CTE_true': fget_any('mean_Delta_CTE_true', 'mean_Delta_TE'),
            'median_Delta_CTE_true': fget_any('median_Delta_CTE_true', 'median_Delta_TE'),
            'ci95_lower': fget('ci95_lower'),
            'ci95_upper': fget('ci95_upper'),
            't_stat': fget('t_stat'),
            'p_value': fget('p_value'),
        })
    if not rows:
        print(f"No summaries found in {run_dir}")
        return 1
    out = pd.DataFrame(rows).sort_values(['A_bins','H_bin_hours','S_mode'])
    print("Latest run:", run_dir)
    print()
    print(out.to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
