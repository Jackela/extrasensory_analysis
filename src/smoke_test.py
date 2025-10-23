"""Lightweight smoke test for the ExtraSensory TE pipeline."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

import src.settings as settings
import src.preprocessing as preprocessing
import src.analysis as analysis

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def pick_sample_uuid(data_dir: str) -> Optional[str]:
    """Return the first UUID (filename stem) found in the data directory."""
    for entry in sorted(Path(data_dir).glob("*.features_labels.csv")):
        stem = entry.stem
        if stem.endswith(".features_labels"):
            stem = stem.replace(".features_labels", "")
        return stem
    return None


def main() -> int:
    logger.info("Starting smoke test …")

    if not os.path.isdir(settings.DATA_PATH):
        logger.error("Data directory not found at %s", settings.DATA_PATH)
        return 1

    uuid = pick_sample_uuid(settings.DATA_PATH)
    if uuid is None:
        logger.error("No per-user CSV files found under %s", settings.DATA_PATH)
        return 1

    logger.info("Using subject %s for the smoke test.", uuid)

    original_surrogates = settings.NUM_SURROGATES
    try:
        settings.NUM_SURROGATES = 2  # keep the smoke test fast

        # Load and preprocess series
        raw_df = preprocessing.load_subject_data(uuid)
        series_A, series_S, series_H = preprocessing.create_variables(raw_df)

        if len(series_A) < 10:
            logger.warning("Subject %s has very few samples after preprocessing (%d).", uuid, len(series_A))

        base_A = int(np.max(series_A)) + 1
        base_S = int(np.max(series_S)) + 1
        base_H = settings.NUM_HOUR_BINS

        analysis.start_jvm()
        _ = analysis.get_jidt_classes()

        te_results = analysis.run_te_analysis(series_A, series_S, k_A=1, k_S=1, base_A=base_A, base_S=base_S)
        cte_results = analysis.run_cte_analysis(
            series_A, series_S, series_H, k_A=1, k_S=1, base_A=base_A, base_S=base_S, base_H_binned=base_H
        )

        logger.info("TE results snapshot: %s", {k: round(v, 4) for k, v in te_results.items()})
        logger.info("CTE results snapshot: %s", {k: round(v, 4) for k, v in cte_results.items()})
        logger.info("Smoke test completed successfully.")
        return 0
    except Exception as exc:  # pragma: no cover - defensive path
        logger.exception("Smoke test failed: %s", exc)
        return 1
    finally:
        settings.NUM_SURROGATES = original_surrogates
        analysis.shutdown_jvm()


if __name__ == "__main__":
    raise SystemExit(main())
