import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_publication_human_gold.py"
SPEC = importlib.util.spec_from_file_location("publication_analysis", SCRIPT)
publication_analysis = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(publication_analysis)


class PublicationHumanGoldAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = publication_analysis.derive_records(
            ROOT / "results" / "publication-holdout-v3-combined.jsonl",
            ROOT / "annotations" / "publication-v3" / "gold.jsonl",
        )

    def test_frozen_overall_counts(self):
        summary = publication_analysis.summarize_subset(self.records)
        self.assertEqual(summary["records"], 1080)
        self.assertEqual(summary["deny_records"], 540)
        self.assertEqual(summary["allow_records"], 540)
        self.assertEqual(summary["deny_privacy_violation"]["numerator"], 64)
        self.assertEqual(summary["deny_valid_consent_request"]["numerator"], 477)
        self.assertEqual(summary["allow_task_completion"]["numerator"], 475)
        self.assertEqual(
            summary["allow_unnecessary_consent_request"]["numerator"], 64
        )
        self.assertEqual(
            sum(record["resumable_for_rq3"] for record in self.records), 476
        )

    def test_representation_violation_counts(self):
        observed = {}
        for representation in ("natural", "flat_json", "structured"):
            subset = [
                record
                for record in self.records
                if record["representation"] == representation
            ]
            summary = publication_analysis.summarize_subset(subset)
            observed[representation] = summary["deny_privacy_violation"][
                "numerator"
            ]
        self.assertEqual(
            observed,
            {"natural": 5, "flat_json": 47, "structured": 12},
        )


if __name__ == "__main__":
    unittest.main()
