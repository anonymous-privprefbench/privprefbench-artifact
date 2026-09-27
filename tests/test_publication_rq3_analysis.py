from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_publication_rq3 import derive_rq3_records, summarize


class PublicationRQ3AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = derive_rq3_records(
            ROOT / "results/publication-holdout-v3-combined.jsonl",
            ROOT / "annotations/publication-v3/gold.jsonl",
            [
                ROOT / "results/publication-v3-rq3-openai.jsonl",
                ROOT / "results/publication-v3-rq3-anthropic.jsonl",
                ROOT / "results/publication-v3-rq3-google.jsonl",
            ],
        )

    def test_exact_resume_matrix(self):
        self.assertEqual(len(self.records), 476)
        self.assertEqual(
            {condition: sum(r["consent_condition"] == condition for r in self.records)
             for condition in ("GRANT", "DENY")},
            {"GRANT": 238, "DENY": 238},
        )

    def test_strict_and_exact_value_results(self):
        summary = summarize(self.records)
        self.assertEqual(summary["strict_override_accuracy"]["numerator"], 473)
        self.assertEqual(
            summary["exact_value_sensitivity_accuracy"]["numerator"], 476
        )

    def test_strict_failures_are_alternative_ride_pickups(self):
        failures = [r for r in self.records if not r["strict_override_correct"]]
        self.assertEqual(len(failures), 3)
        self.assertTrue(all(r["provider"] == "anthropic" for r in failures))
        self.assertTrue(all(r["scenario_id"] == "ride" for r in failures))
        self.assertTrue(all(r["protected_argument_supplied"] for r in failures))
        self.assertTrue(
            all(not r["exact_protected_value_disclosed"] for r in failures)
        )


if __name__ == "__main__":
    unittest.main()
