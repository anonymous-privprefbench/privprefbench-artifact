import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import merge_publication_first_turns as merger


SCENARIOS = [
    "flight",
    "job_application",
    "medical",
    "restaurant",
    "ride",
    "shipment",
]
POLICIES = ["ALLOW", "DENY"]
REPRESENTATIONS = ["natural", "flat_json", "structured"]


def provider_rows(provider, repeats=1):
    rows = []
    for scenario in SCENARIOS:
        for policy in POLICIES:
            for representation in REPRESENTATIONS:
                for repeat in range(repeats):
                    rows.append(
                        {
                            "first_turn_id": "%s-%s-%s-%s-%d"
                            % (provider, scenario, policy, representation, repeat),
                            "provider": provider,
                            "model": merger.EXPECTED_MODELS[provider],
                            "status": "completed",
                            "scenario_id": scenario,
                            "policy_mode": policy,
                            "representation": representation,
                            "repeat": repeat,
                        }
                    )
    return rows


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class PublicationMergeTests(unittest.TestCase):
    def test_valid_three_provider_matrix_merges(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for provider in merger.EXPECTED_MODELS:
                paths[provider] = root / (provider + ".jsonl")
                write_jsonl(paths[provider], provider_rows(provider))
            output = root / "combined.jsonl"
            argv = [
                "merge_publication_first_turns.py",
                "--openai", str(paths["openai"]),
                "--anthropic", str(paths["anthropic"]),
                "--google", str(paths["google"]),
                "--repeats", "1",
                "--output", str(output),
            ]
            with patch.object(sys, "argv", argv):
                self.assertEqual(merger.main(), 0)
            self.assertEqual(len(merger.load_jsonl(output)), 108)

    def test_provider_failure_is_rejected(self):
        rows = provider_rows("google")
        rows[0]["status"] = "provider_error"
        with self.assertRaisesRegex(ValueError, "provider failures"):
            merger.validate_provider(rows, "google", 1)

    def test_missing_repeat_is_rejected(self):
        rows = provider_rows("anthropic", repeats=2)
        rows[-1]["repeat"] = 0
        with self.assertRaisesRegex(ValueError, "missing or duplicate repeat"):
            merger.validate_provider(rows, "anthropic", 2)


if __name__ == "__main__":
    unittest.main()
