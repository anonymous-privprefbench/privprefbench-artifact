import json
import unittest

from privprefbench.policy import CONSENT_EXCEPTION, build_policy
from privprefbench.renderers import flat_object, render_preference, structured_object
from privprefbench.scenarios import load_all_scenarios
from privprefbench.schema import PolicyEffect, Representation


class RendererTests(unittest.TestCase):
    def test_all_representations_preserve_policy_dimensions(self):
        for scenario in load_all_scenarios():
            for mode in ("allow", "deny"):
                rule = build_policy(scenario, mode)
                flat = flat_object(rule)["privacy_preferences"][0]
                structured = structured_object(rule)["rules"][0]

                self.assertEqual(flat["data"], list(rule.data))
                self.assertEqual(flat["recipient_type"], rule.recipient_type)
                self.assertEqual(flat["purpose"], rule.purpose)
                self.assertEqual(flat["decision"], rule.effect.value.lower())
                self.assertEqual(structured["data"], list(rule.data))
                self.assertEqual(structured["recipient"]["type"], rule.recipient_type)
                self.assertEqual(structured["purpose"], rule.purpose)
                self.assertEqual(structured["effect"], rule.effect.value)

                natural = render_preference(rule, Representation.NATURAL, scenario)
                self.assertIn(scenario.recipient_label, natural)
                self.assertIn(scenario.purpose_label, natural)
                for field in rule.data:
                    self.assertIn(scenario.data_labels[field], natural)

                flat_text = render_preference(rule, Representation.FLAT_JSON, scenario)
                structured_text = render_preference(
                    rule, Representation.STRUCTURED, scenario
                )
                json.loads(flat_text)
                json.loads(structured_text)

                if rule.effect is PolicyEffect.DENY:
                    self.assertEqual(flat["exception"], CONSENT_EXCEPTION.lower())
                    self.assertEqual(
                        structured["unless"],
                        {"consent": "EXPLICIT", "scope": "CURRENT_TASK"},
                    )
                else:
                    self.assertNotIn("exception", flat)
                    self.assertNotIn("unless", structured)


if __name__ == "__main__":
    unittest.main()

