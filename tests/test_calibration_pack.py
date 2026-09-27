import csv
import json
import unittest
from collections import Counter
from pathlib import Path

from privprefbench.annotations import validate_annotation_row
from privprefbench.evaluator import classify_consent_request
from privprefbench.policy import build_policy
from privprefbench.scenarios import load_scenario
from scripts.create_calibration_pack import select_records


ROOT = Path(__file__).resolve().parents[1]


class CalibrationPackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "annotations" / "development-v1" / "first-turns.jsonl"
        cls.records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_selection_is_deterministic_and_balanced(self):
        first = select_records(self.records, per_cell=5, seed=4099)
        second = select_records(self.records, per_cell=5, seed=4099)
        self.assertEqual(
            [item["first_turn_id"] for item in first],
            [item["first_turn_id"] for item in second],
        )
        self.assertEqual(len(first), 90)
        self.assertEqual(len({item["first_turn_id"] for item in first}), 90)
        self.assertEqual(
            Counter(item["scenario_id"] for item in first),
            Counter(
                {
                    "flight": 15,
                    "job_application": 15,
                    "medical": 15,
                    "restaurant": 15,
                    "ride": 15,
                    "shipment": 15,
                }
            ),
        )
        self.assertEqual(
            Counter(item["representation"] for item in first),
            Counter({"natural": 30, "flat_json": 30, "structured": 30}),
        )
        cells = Counter(
            (item["scenario_id"], item["representation"]) for item in first
        )
        self.assertEqual(len(cells), 18)
        self.assertEqual(set(cells.values()), {5})
        for cell in cells:
            policy_counts = Counter(
                item["policy_mode"]
                for item in first
                if (item["scenario_id"], item["representation"]) == cell
            )
            self.assertGreaterEqual(policy_counts["ALLOW"], 2)
            self.assertGreaterEqual(policy_counts["DENY"], 2)

    def test_completed_calibration_is_valid_and_matches_revised_detector(self):
        completed_path = (
            ROOT / "annotations" / "calibration-v1" / "researcher-a-completed.csv"
        )
        blank_path = ROOT / "annotations" / "calibration-v1" / "researcher-a.csv"
        selected_path = (
            ROOT / "annotations" / "calibration-v1" / "first-turns.jsonl"
        )
        with completed_path.open("r", encoding="utf-8", newline="") as handle:
            completed = list(csv.DictReader(handle))
        with blank_path.open("r", encoding="utf-8", newline="") as handle:
            blank = list(csv.DictReader(handle))
        selected = {
            item["first_turn_id"]: item
            for item in (
                json.loads(line)
                for line in selected_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }
        self.assertEqual(len(completed), 90)
        self.assertEqual(
            {row["first_turn_id"] for row in completed},
            set(selected),
        )
        blank_by_id = {row["first_turn_id"]: row for row in blank}
        for row in completed:
            validate_annotation_row(row)
            original = blank_by_id[row["first_turn_id"]]
            self.assertEqual(
                [row[key] for key in list(row)[:10]],
                [original[key] for key in list(row)[:10]],
            )
            item = selected[row["first_turn_id"]]
            scenario = load_scenario(item["scenario_id"])
            rule = build_policy(scenario, item["policy_mode"].lower())
            text = item["assistant_turn"]["text"]
            self.assertEqual(
                classify_consent_request(text, scenario, rule).value,
                row["label"],
            )


if __name__ == "__main__":
    unittest.main()
