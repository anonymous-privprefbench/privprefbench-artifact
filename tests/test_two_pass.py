import unittest

from privprefbench.schema import to_jsonable

from privprefbench.providers.base import ScriptedProvider
from privprefbench.scenarios import load_scenario
from privprefbench.schema import (
    ConsentAdjudication,
    ConsentCondition,
    ConsentRequestLabel,
    PolicyEffect,
    Representation,
    NormalizedToolCall,
)
from privprefbench.tools import execute_mock_tool
from privprefbench.two_pass import (
    adjudication_from_mapping,
    collect_first_turn,
    first_turn_record_from_mapping,
    resume_adjudicated_run,
)


class TwoPassTests(unittest.TestCase):
    def setUp(self):
        self.scenario = load_scenario("restaurant")

    def collect(self, effect, condition):
        provider = ScriptedProvider(self.scenario, effect, condition)
        record = collect_first_turn(
            provider=provider,
            scenario=self.scenario,
            policy_mode=effect.value,
            representation=Representation.NATURAL,
            repeat=0,
            consent_condition=condition,
        )
        return provider, record

    def adjudication(self, record, label):
        return ConsentAdjudication(
            first_turn_id=record.first_turn_id,
            label=label,
            adjudicator_id="test-adjudicator",
            evidence="Synthetic fixture evidence.",
            action_explicit=label is not ConsentRequestLabel.NOT_REQUEST,
            protected_fields_complete=label is ConsentRequestLabel.VALID_REQUEST,
            recipient_or_purpose_scoped=label is ConsentRequestLabel.VALID_REQUEST,
            source_annotators=("annotator-a", "annotator-b"),
        )

    def test_collection_never_injects_consent(self):
        _, record = self.collect(PolicyEffect.DENY, ConsentCondition.GRANT)
        self.assertIsNotNone(record.assistant_turn)
        self.assertEqual(record.tool_executions, [])
        self.assertIn("May I", record.assistant_turn.text)

    def test_valid_grant_resumes_and_completes(self):
        provider, first = self.collect(PolicyEffect.DENY, ConsentCondition.GRANT)
        final = resume_adjudicated_run(
            provider,
            self.scenario,
            first,
            self.adjudication(first, ConsentRequestLabel.VALID_REQUEST),
        )
        self.assertEqual(len(final.assistant_turns), 2)
        self.assertEqual(len(final.consent_events), 1)
        self.assertTrue(final.evaluation.task_completed)
        self.assertTrue(final.evaluation.override_handled_correctly)

    def test_valid_denial_resumes_without_disclosure(self):
        provider, first = self.collect(PolicyEffect.DENY, ConsentCondition.DENY)
        final = resume_adjudicated_run(
            provider,
            self.scenario,
            first,
            self.adjudication(first, ConsentRequestLabel.VALID_REQUEST),
        )
        self.assertFalse(final.evaluation.task_completed)
        self.assertFalse(final.evaluation.unauthorized_disclosure)
        self.assertTrue(final.evaluation.override_handled_correctly)

    def test_not_request_does_not_resume(self):
        provider, first = self.collect(PolicyEffect.DENY, ConsentCondition.GRANT)
        final = resume_adjudicated_run(
            provider,
            self.scenario,
            first,
            self.adjudication(first, ConsentRequestLabel.NOT_REQUEST),
        )
        self.assertEqual(len(final.assistant_turns), 1)
        self.assertEqual(final.consent_events, [])
        self.assertFalse(final.evaluation.asked_for_consent)

    def test_valid_request_with_premature_call_is_not_resumed(self):
        provider, first = self.collect(PolicyEffect.DENY, ConsentCondition.GRANT)
        call = NormalizedToolCall(
            call_id="premature",
            name=self.scenario.tool.name,
            arguments=dict(self.scenario.mock_arguments),
            turn_index=0,
        )
        first.assistant_turn.tool_calls.append(call)
        first.tool_executions.append(
            execute_mock_tool(
                scenario=self.scenario,
                call_id=call.call_id,
                name=call.name,
                arguments=call.arguments,
                turn_index=call.turn_index,
            )
        )
        final = resume_adjudicated_run(
            provider,
            self.scenario,
            first,
            self.adjudication(first, ConsentRequestLabel.VALID_REQUEST),
        )
        self.assertEqual(len(final.assistant_turns), 1)
        self.assertEqual(final.consent_events, [])
        self.assertTrue(final.evaluation.asked_for_consent)
        self.assertTrue(final.evaluation.unauthorized_disclosure)

    def test_ambiguous_must_be_resolved(self):
        provider, first = self.collect(PolicyEffect.DENY, ConsentCondition.GRANT)
        with self.assertRaisesRegex(ValueError, "must be resolved"):
            resume_adjudicated_run(
                provider,
                self.scenario,
                first,
                self.adjudication(first, ConsentRequestLabel.AMBIGUOUS),
            )

    def test_adjudication_must_reference_same_record(self):
        provider, first = self.collect(PolicyEffect.DENY, ConsentCondition.GRANT)
        wrong = ConsentAdjudication(
            first_turn_id="another-record",
            label=ConsentRequestLabel.VALID_REQUEST,
            adjudicator_id="test-adjudicator",
            evidence="Synthetic fixture evidence.",
            action_explicit=True,
            protected_fields_complete=True,
            recipient_or_purpose_scoped=True,
        )
        with self.assertRaisesRegex(ValueError, "different first-turn"):
            resume_adjudicated_run(provider, self.scenario, first, wrong)

    def test_persisted_records_round_trip(self):
        _, first = self.collect(PolicyEffect.DENY, ConsentCondition.GRANT)
        adjudication = self.adjudication(
            first, ConsentRequestLabel.VALID_REQUEST
        )
        restored_first = first_turn_record_from_mapping(to_jsonable(first))
        restored_adjudication = adjudication_from_mapping(
            to_jsonable(adjudication)
        )
        self.assertEqual(restored_first, first)
        self.assertEqual(restored_adjudication, adjudication)

    def test_adjudication_json_rejects_string_booleans(self):
        _, first = self.collect(PolicyEffect.DENY, ConsentCondition.GRANT)
        raw = to_jsonable(
            self.adjudication(first, ConsentRequestLabel.VALID_REQUEST)
        )
        raw["action_explicit"] = "false"
        with self.assertRaisesRegex(TypeError, "must be booleans"):
            adjudication_from_mapping(raw)


if __name__ == "__main__":
    unittest.main()
