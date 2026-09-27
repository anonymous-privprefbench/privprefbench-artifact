import unittest

from privprefbench.providers.base import ScriptedProvider
from privprefbench.runner import run_once
from privprefbench.scenarios import load_scenario
from privprefbench.schema import ConsentCondition, PolicyEffect, Representation


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.scenario = load_scenario("restaurant")

    def run_scripted(self, effect, condition):
        provider = ScriptedProvider(self.scenario, effect, condition)
        return run_once(
            provider=provider,
            scenario=self.scenario,
            policy_mode=effect.value,
            representation=Representation.STRUCTURED,
            repeat=0,
            consent_condition=condition,
        )

    def test_allow_completes_directly(self):
        record = self.run_scripted(
            PolicyEffect.ALLOW, ConsentCondition.NOT_APPLICABLE
        )
        self.assertEqual(record.status, "completed")
        self.assertTrue(record.evaluation.task_completed)
        self.assertEqual(len(record.assistant_turns), 1)

    def test_grant_path_completes_safely(self):
        record = self.run_scripted(PolicyEffect.DENY, ConsentCondition.GRANT)
        self.assertTrue(record.evaluation.asked_for_consent)
        self.assertTrue(record.evaluation.override_handled_correctly)
        self.assertTrue(record.evaluation.task_completed)
        self.assertEqual(len(record.assistant_turns), 2)

    def test_deny_path_does_not_disclose(self):
        record = self.run_scripted(PolicyEffect.DENY, ConsentCondition.DENY)
        self.assertTrue(record.evaluation.asked_for_consent)
        self.assertTrue(record.evaluation.override_handled_correctly)
        self.assertFalse(record.evaluation.task_completed)
        self.assertFalse(record.evaluation.unauthorized_disclosure)

    def test_provider_error_is_not_recorded_as_refusal(self):
        class FailingProvider(ScriptedProvider):
            name = "failing"

            def complete(self, messages, tools, turn_index):
                raise RuntimeError("synthetic provider failure")

        provider = FailingProvider(
            self.scenario, PolicyEffect.DENY, ConsentCondition.GRANT
        )
        record = run_once(
            provider=provider,
            scenario=self.scenario,
            policy_mode="deny",
            representation=Representation.NATURAL,
            repeat=0,
            consent_condition=ConsentCondition.GRANT,
        )
        self.assertEqual(record.status, "provider_error")
        self.assertIn("synthetic provider failure", record.error)
        self.assertFalse(record.evaluation.task_completed)
        self.assertFalse(record.evaluation.asked_for_consent)
        self.assertIn("not classified as a refusal", record.evaluation.notes[-1])


if __name__ == "__main__":
    unittest.main()
