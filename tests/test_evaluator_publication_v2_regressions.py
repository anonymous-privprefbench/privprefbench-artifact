import json
import unittest
from pathlib import Path

from privprefbench.evaluator import classify_consent_request
from privprefbench.policy import build_policy
from privprefbench.scenarios import load_scenario
from privprefbench.schema import ConsentRequestLabel


ROOT = Path(__file__).resolve().parents[1]
RC4_ERRORS = ROOT / "evaluations/publication-v2/rc4-errors.jsonl"


class EvaluatorPublicationV2RegressionTests(unittest.TestCase):
    def test_rc5_corrects_every_opened_rc4_holdout_error(self):
        rows = [
            json.loads(line)
            for line in RC4_ERRORS.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 65)

        for row in rows:
            with self.subTest(first_turn_id=row["first_turn_id"]):
                scenario = load_scenario(row["scenario_id"])
                rule = build_policy(scenario, row["policy_mode"].lower())
                predicted = classify_consent_request(
                    row["assistant_text"], scenario, rule
                )
                self.assertIs(
                    predicted,
                    ConsentRequestLabel(row["gold_label"]),
                )


if __name__ == "__main__":
    unittest.main()
