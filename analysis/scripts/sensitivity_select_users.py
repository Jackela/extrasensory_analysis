"""
Select N=10 users stratified by k (High-k: k>=5; Low-k: k<=4) from
analysis/out/FINAL_RUN_k60_COMPLETE/k_selected_by_user.csv and save to
analysis/config/sensitivity_users_n10.txt. Deterministic sampling.
"""
from pathlib import Path
import pandas as pd
import json

BASE_DIR = Path("analysis/out/FINAL_RUN_k60_COMPLETE")
KSEL_PATH = BASE_DIR / "k_selected_by_user.csv"
OUT_DIR = Path("analysis/config")
OUT_TXT = OUT_DIR / "sensitivity_users_n10.txt"
OUT_JSON = OUT_DIR / "sensitivity_users_n10.json"


def main() -> int:
    df = pd.read_csv(KSEL_PATH)
    if not {"user_id", "k_selected"}.issubset(df.columns):
        raise ValueError("k_selected_by_user.csv missing required columns")

    high = df[df["k_selected"] >= 5].copy()
    low = df[df["k_selected"] <= 4].copy()

    # Deterministic sampling
    high_sample = high.sample(n=min(5, len(high)), random_state=42) if len(high) > 5 else high
    low_sample = low.sample(n=min(5, len(low)), random_state=42) if len(low) > 5 else low

    # If one group has fewer than 5, top up from the other
    if len(high_sample) < 5 and len(low) > len(low_sample):
        needed = 5 - len(high_sample)
        extra = low.drop(index=low_sample.index, errors='ignore').sample(n=min(needed, len(low) - len(low_sample)), random_state=43)
        high_sample = pd.concat([high_sample, extra]).head(5)
    if len(low_sample) < 5 and len(high) > len(high_sample):
        needed = 5 - len(low_sample)
        extra = high.drop(index=high_sample.index, errors='ignore').sample(n=min(needed, len(high) - len(high_sample)), random_state=44)
        low_sample = pd.concat([low_sample, extra]).head(5)

    users = pd.concat([high_sample, low_sample])["user_id"].tolist()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_TXT, "w") as f:
        for u in users:
            f.write(str(u) + "\n")

    meta = {
        "counts": {"high_k": int(len(high_sample)), "low_k": int(len(low_sample))},
        "k_rule": "High: k>=5 (k=5,6), Low: k<=4 (k=1..4)",
        "source": str(KSEL_PATH),
        "users": users,
    }
    OUT_JSON.write_text(json.dumps(meta, indent=2))

    print(f"Saved {len(users)} users to {OUT_TXT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
