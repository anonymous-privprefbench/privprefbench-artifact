import unittest

from privprefbench.scenarios import load_scenario
from privprefbench.tools import build_tool_schema, execute_mock_tool


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.scenario = load_scenario("ride")

    def test_schema_is_strict_and_exposes_recipient_metadata(self):
        schema = build_tool_schema(self.scenario)
        self.assertTrue(schema["strict"])
        self.assertFalse(schema["parameters"]["additionalProperties"])
        self.assertIn("mobility provider", schema["description"])
        self.assertEqual(
            schema["parameters"]["properties"]["pickup_location"]["pattern"],
            ".+",
        )

    def test_mock_tool_validates_required_arguments(self):
        execution = execute_mock_tool(
            self.scenario,
            "call-1",
            self.scenario.tool.name,
            {"pickup_location": "somewhere"},
            0,
        )
        self.assertFalse(execution.success)
        self.assertIn("missing required", execution.error)

    def test_mock_tool_rejects_empty_required_string(self):
        arguments = dict(self.scenario.mock_arguments)
        arguments["pickup_location"] = ""
        execution = execute_mock_tool(
            self.scenario,
            "call-2",
            self.scenario.tool.name,
            arguments,
            0,
        )
        self.assertFalse(execution.success)
        self.assertIn("empty required argument", execution.error)

    def test_mock_tool_rejects_non_disclosing_protected_placeholder(self):
        scenario = load_scenario("job_application")
        arguments = dict(scenario.mock_arguments)
        arguments["phone"] = "Not provided"
        execution = execute_mock_tool(
            scenario=scenario,
            call_id="placeholder-call",
            name=scenario.tool.name,
            arguments=arguments,
            turn_index=0,
        )
        self.assertFalse(execution.success)
        self.assertIn("non-disclosing placeholder", execution.error)


if __name__ == "__main__":
    unittest.main()
