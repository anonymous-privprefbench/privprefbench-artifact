import unittest

from privprefbench.scenarios import list_scenario_ids, load_all_scenarios


class ScenarioTests(unittest.TestCase):
    def test_all_six_scenarios_load(self):
        self.assertEqual(len(load_all_scenarios()), 6)
        self.assertEqual(
            set(list_scenario_ids()),
            {"restaurant", "ride", "job_application", "shipment", "medical", "flight"},
        )

    def test_synthetic_values_and_protected_mappings(self):
        for scenario in load_all_scenarios():
            self.assertTrue(scenario.protected_fields)
            self.assertEqual(set(scenario.tool.required), set(scenario.mock_arguments))
            for field, argument in scenario.protected_argument_map.items():
                self.assertIn(field, scenario.user_data)
                self.assertEqual(
                    scenario.user_data[field], scenario.mock_arguments[argument]
                )


if __name__ == "__main__":
    unittest.main()

