"""Compute transfer entropy between activity_level and label:SITTING using JIDT."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import jpype
import pandas as pd
from jpype import JArray, JInt

DATA_PATH = Path("mvp_dataset.csv")
JAR_PATH = Path("infodynamics-dist-1.6.1/infodynamics.jar")
PER_USER_OUTPUT = Path("per_user_transfer_entropy.csv")
REPORT_PATH = Path("transfer_entropy_report.txt")

HISTORY_LENGTH = 1
SOURCE_HISTORY_LENGTH = 1
DELAY = 1


def start_jvm() -> None:
    """Start the JVM with the JIDT jar on the classpath if it is not running."""
    if jpype.isJVMStarted():
        return
    if not JAR_PATH.exists():
        raise FileNotFoundError(f"JIDT jar not found at {JAR_PATH}")
    jpype.startJVM(
        jpype.getDefaultJVMPath(),
        "-ea",
        f"-Djava.class.path={JAR_PATH}"
    )


def to_jint_array(values: Iterable[int]):
    """Convert an iterable of ints to a Java int[] array."""
    return JArray(JInt, 1)([int(v) for v in values])


def compute_te(
    source: Iterable[int],
    dest: Iterable[int],
    base: int,
    history_length: int = HISTORY_LENGTH,
    source_history: int = SOURCE_HISTORY_LENGTH,
    delay: int = DELAY,
) -> float:
    """Compute transfer entropy from source to dest using the discrete calculator."""
    start_jvm()
    TransferEntropyCalculatorDiscrete = jpype.JClass(
        "infodynamics.measures.discrete.TransferEntropyCalculatorDiscrete"
    )
    calculator = TransferEntropyCalculatorDiscrete(base, history_length, source_history)
    try:
        calculator.setProperty("k_TAU", str(delay))
    except Exception:
        # Some discrete calculators do not expose delay as a property; defaults to 1.
        pass
    try:
        calculator.setProperty("l_TAU", str(delay))
    except Exception:
        pass
    calculator.initialise()
    calculator.addObservations(to_jint_array(source), to_jint_array(dest))
    return float(calculator.computeAverageLocalOfObservations())


def prepare_dataframe() -> pd.DataFrame:
    """Load the dataset, create the activity_level column, and ensure integer targets."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    uuid_column = "uuid" if "uuid" in df.columns else None
    if uuid_column is None:
        # Dataset ships without explicit uuid; treat the sequence as a single cohort.
        df["uuid"] = "global"
        uuid_column = "uuid"

    required_columns = [
        uuid_column,
        "timestamp",
        "raw_acc:magnitude_stats:mean",
        "label:SITTING",
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["raw_acc:magnitude_stats:mean", "label:SITTING"])
    df = df.sort_values([uuid_column, "timestamp"]).reset_index(drop=True)

    # Equal-frequency binning into 3 activity levels.
    activity_col = "raw_acc:magnitude_stats:mean"
    try:
        df["activity_level"] = pd.qcut(
            df[activity_col],
            q=3,
            labels=[0, 1, 2],
            duplicates="drop",
        )
    except ValueError as exc:
        raise ValueError("Failed to create equal-frequency bins for activity level") from exc

    if df["activity_level"].isna().any():
        # If duplicates were dropped resulting in fewer bins, fall back to pandas cut on percentiles.
        quantiles = df[activity_col].quantile([0, 1 / 3, 2 / 3, 1]).values
        df["activity_level"] = pd.cut(
            df[activity_col],
            bins=quantiles,
            labels=[0, 1, 2],
            include_lowest=True,
        )
    df["activity_level"] = df["activity_level"].astype(int)
    df["label:SITTING"] = df["label:SITTING"].astype(int)
    return df


def summarize(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean_val = float(sum(values) / len(values))
    if len(values) > 1:
        variance = sum((v - mean_val) ** 2 for v in values) / (len(values) - 1)
        std_val = math.sqrt(variance)
    else:
        std_val = 0.0
    return mean_val, std_val


def main() -> None:
    df = prepare_dataframe()
    uuid_column = "uuid"

    records: List[Dict[str, object]] = []
    for uuid_value, group in df.groupby(uuid_column):
        if len(group) <= HISTORY_LENGTH + DELAY:
            continue
        activity = group["activity_level"].astype(int)
        sitting = group["label:SITTING"].astype(int)
        alphabet_size = max(int(activity.max()) + 1, int(sitting.max()) + 1, 2)
        te_activity_to_sitting = compute_te(activity, sitting, alphabet_size)
        te_sitting_to_activity = compute_te(sitting, activity, alphabet_size)
        records.append(
            {
                "uuid": uuid_value,
                "direction": "activity_level->label:SITTING",
                "transfer_entropy_bits": te_activity_to_sitting,
            }
        )
        records.append(
            {
                "uuid": uuid_value,
                "direction": "label:SITTING->activity_level",
                "transfer_entropy_bits": te_sitting_to_activity,
            }
        )

    if not records:
        raise RuntimeError("No valid groups found for transfer entropy computation")

    results_df = pd.DataFrame(records)
    results_df.to_csv(PER_USER_OUTPUT, index=False)

    activity_to_state = results_df[
        results_df["direction"] == "activity_level->label:SITTING"
    ]["transfer_entropy_bits"].tolist()
    state_to_activity = results_df[
        results_df["direction"] == "label:SITTING->activity_level"
    ]["transfer_entropy_bits"].tolist()

    mean_a2s, std_a2s = summarize(activity_to_state)
    mean_s2a, std_s2a = summarize(state_to_activity)

    comparison = "greater than"
    if math.isclose(mean_a2s, mean_s2a, rel_tol=1e-6, abs_tol=1e-6):
        comparison = "roughly equal to"
    elif mean_a2s < mean_s2a:
        comparison = "less than"

    interpretation_lines = [
        "Transfer Entropy Summary",
        "-------------------------",
        f"Parameters: history length k={HISTORY_LENGTH}, delay tau={DELAY}",
        "",
        "Average TE (bits):",
        f"- activity_level -> label:SITTING: {mean_a2s:.4f} +/- {std_a2s:.4f} (n={len(activity_to_state)})",
        f"- label:SITTING -> activity_level: {mean_s2a:.4f} +/- {std_s2a:.4f} (n={len(state_to_activity)})",
        "",
        "Interpretation:",
        f"The average TE from activity level to sitting state is {comparison} the reverse direction,",
        "suggesting how reliably motion signals anticipate posture changes.",
        "This insight helps product teams reason about moments like a candidate PM spotting a\n"
        " user's upcoming focus period (e.g., sitting still before a meeting) and timing 'do not\n"
        " disturb' nudges or proactive assistance.",
    ]

    # Write report and print to stdout.
    with open(REPORT_PATH, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(interpretation_lines))

    print("\n".join(interpretation_lines))


if __name__ == "__main__":
    main()
