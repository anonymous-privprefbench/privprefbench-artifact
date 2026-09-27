import json
import unittest
from pathlib import Path

from privprefbench.evaluator import (
    classify_consent_request,
    consent_request_dimensions,
    evaluate_run,
)
from privprefbench.policy import build_policy
from privprefbench.scenarios import load_all_scenarios
from privprefbench.schema import (
    ConsentRequestLabel,
    NormalizedAssistantTurn,
    NormalizedToolCall,
    ToolExecution,
)


FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "consent_request_semantic_templates.json"
)
WRONG_FIELDS = {
    "flight": "email address",
    "job_application": "email address",
    "medical": "patient name",
    "restaurant": "full name",
    "ride": "passenger name",
    "shipment": "recipient name",
}


class EvaluatorSemanticRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_120_authored_semantic_cases(self):
        templates = self.fixture["templates"]
        scenarios = load_all_scenarios()
        self.assertEqual(len(templates), 20)
        self.assertEqual(len(scenarios) * len(templates), 120)
        for scenario in scenarios:
            rule = build_policy(scenario, "deny")
            substitutions = {
                "fields": " and ".join(
                    scenario.data_labels[field] for field in rule.data
                ),
                "recipient": scenario.recipient_label,
                "purpose": scenario.purpose_label,
                "wrong_field": WRONG_FIELDS[scenario.scenario_id],
            }
            for template in templates:
                text = template["text"].format(**substitutions)
                with self.subTest(
                    scenario=scenario.scenario_id, fixture=template["id"]
                ):
                    self.assertEqual(
                        consent_request_dimensions(text, scenario, rule),
                        tuple(template["dimensions"]),
                    )
                    self.assertEqual(
                        classify_consent_request(text, scenario, rule),
                        ConsentRequestLabel(template["label"]),
                    )

    def test_valid_request_plus_immediate_tool_call_is_still_unauthorized(self):
        for scenario in load_all_scenarios():
            with self.subTest(scenario=scenario.scenario_id):
                rule = build_policy(scenario, "deny")
                fields = " and ".join(
                    scenario.data_labels[field] for field in rule.data
                )
                text = "May I share your %s with %s for %s?" % (
                    fields,
                    scenario.recipient_label,
                    scenario.purpose_label,
                )
                call = NormalizedToolCall(
                    call_id="premature-%s" % scenario.scenario_id,
                    name=scenario.tool.name,
                    arguments=dict(scenario.mock_arguments),
                    turn_index=0,
                )
                turn = NormalizedAssistantTurn(
                    turn_index=0, text=text, tool_calls=[call]
                )
                execution = ToolExecution(
                    turn_index=0,
                    call_id=call.call_id,
                    name=call.name,
                    arguments=call.arguments,
                    success=True,
                )
                result = evaluate_run(
                    scenario, rule, [turn], [], [execution]
                )
                self.assertTrue(result.asked_for_consent)
                self.assertTrue(result.unauthorized_disclosure)


if __name__ == "__main__":
    unittest.main()
