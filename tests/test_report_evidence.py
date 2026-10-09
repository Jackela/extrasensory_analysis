"""Exercise real exports and failure cases without Java or external data."""

import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "report_evidence", ROOT / "tools/verify_report_data.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReportEvidence(unittest.TestCase):
    def test_committed_exports_match_manifest_and_known_counts(self):
        actual = module.summarize(ROOT / "report/data")
        self.assertEqual(actual, json.loads((ROOT / "report/data/manifest.json").read_text()))
        self.assertEqual(actual["unique_users"], 60)
        self.assertEqual(actual["metrics"][0]["a_to_s_q_lt_0_05_users"], 33)
        self.assertAlmostEqual(actual["metrics"][0]["a_to_s_q_lt_0_05_percent"], 55.0)

    def test_duplicate_primary_observation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data"
            shutil.copytree(ROOT / "report/data", data)
            p = data / "per_user_true_cte.csv"
            lines = p.read_text().splitlines()
            p.write_text("\n".join([*lines, lines[1]]) + "\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                module.summarize(data)

    def test_legacy_direction_disagreement_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data"
            shutil.copytree(ROOT / "report/data", data)
            p = data / "final_results_summary_n60.csv"
            import csv

            with p.open() as f:
                rows = list(csv.DictReader(f))
            rows[0]["TE_AtoS"] = "123"
            with p.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "does not match"):
                module.summarize(data)


if __name__ == "__main__":
    unittest.main()
