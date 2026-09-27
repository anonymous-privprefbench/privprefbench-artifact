import unittest

from privprefbench.evaluator import evaluate_run
from privprefbench.policy import build_policy
from privprefbench.scenarios import load_scenario
from privprefbench.schema import (
    ConsentCondition,
    ConsentEvent,
    NormalizedAssistantTurn,
    NormalizedToolCall,
    ToolExecution,
)


class EvaluatorStateMatrixTests(unittest.TestCase):
    def setUp(self):
        self.scenario = load_scenario("restaurant")
        self.rule = build_policy(self.scenario, "deny")

    def call_turn(self, turn_index, suffix=""):
        call = NormalizedToolCall(
            call_id="call-%d%s" % (turn_index, suffix),
            name=self.scenario.tool.name,
            arguments=dict(self.scenario.mock_arguments),
            turn_index=turn_index,
        )
        turn = NormalizedAssistantTurn(turn_index=turn_index, tool_calls=[call])
        execution = ToolExecution(
            turn_index=turn_index,
            call_id=call.call_id,
            name=call.name,
            arguments=call.arguments,
            success=True,
        )
        return turn, execution

    def event(self, **overrides):
        values = {
            "turn_index": 1,
            "condition": ConsentCondition.GRANT,
            "data": self.rule.data,
            "recipient_type": self.rule.recipient_type,
            "purpose": self.rule.purpose,
            "scope": "CURRENT_TASK",
            "text": "Synthetic consent event.",
        }
        values.update(overrides)
        return ConsentEvent(**values)

    def test_only_prior_exactly_scoped_grant_authorizes(self):
        cases = (
            ("correct", {}, False),
            ("denial", {"condition": ConsentCondition.DENY}, True),
            ("late", {"turn_index": 3}, True),
            ("wrong_data", {"data": ("another_field",)}, True),
            ("wrong_recipient", {"recipient_type": "another_recipient"}, True),
            ("wrong_purpose", {"purpose": "another_purpose"}, True),
            ("wrong_scope", {"scope": "FUTURE_TASKS"}, True),
        )
        turn, execution = self.call_turn(2)
        for name, overrides, expected_unauthorized in cases:
            with self.subTest(name=name):
                result = evaluate_run(
                    self.scenario,
                    self.rule,
                    [turn],
                    [self.event(**overrides)],
                    [execution],
                )
                self.assertEqual(
                    result.unauthorized_disclosure, expected_unauthorized
                )

    def test_later_grant_never_repairs_earlier_disclosure(self):
        first_turn, first_execution = self.call_turn(0, "-before")
        second_turn, second_execution = self.call_turn(2, "-after")
        result = evaluate_run(
            self.scenario,
            self.rule,
            [first_turn, second_turn],
            [self.event()],
            [first_execution, second_execution],
        )
        self.assertEqual(
            [item["authorized"] for item in result.disclosure_events],
            [False, True],
        )
        self.assertTrue(result.unauthorized_disclosure)
        self.assertFalse(result.override_handled_correctly)

    def test_multiple_authorized_calls_remain_authorized(self):
        first_turn, first_execution = self.call_turn(2, "-one")
        second_turn, second_execution = self.call_turn(3, "-two")
        result = evaluate_run(
            self.scenario,
            self.rule,
            [first_turn, second_turn],
            [self.event()],
            [first_execution, second_execution],
        )
        self.assertEqual(
            [item["authorized"] for item in result.disclosure_events],
            [True, True],
        )
        self.assertFalse(result.unauthorized_disclosure)


if __name__ == "__main__":
    unittest.main()
