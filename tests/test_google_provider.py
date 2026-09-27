import types as pytypes
import unittest

try:
    from google.genai import types
except ImportError:
    types = None

from privprefbench.providers.google_provider import GoogleProvider, _normalized_usage


class FakeModels:
    def __init__(self, response, error=None):
        self.response = response
        self.error = error
        self.request = None

    def generate_content(self, **kwargs):
        self.request = kwargs
        if self.error is not None:
            raise self.error
        return self.response


def fake_response(malformed=False, second_call=False):
    function_args = {"email": "a@example.test"}
    primary_call = types.FunctionCall(
        id="call-1", name="reserve_restaurant", args=function_args
    )
    if malformed:
        primary_call.args = "not-an-object"
    parts = [
        types.Part(text="hidden", thought=True, thought_signature=b"opaque"),
        types.Part(text="I can help."),
        types.Part(
            function_call=primary_call
        ),
    ]
    if second_call:
        parts.append(
            types.Part(
                function_call=types.FunctionCall(
                    id="call-2", name="send_confirmation", args={"channel": "email"}
                )
            )
        )
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=parts),
                finish_reason="STOP",
            )
        ],
        model_version="gemini-3.8-flash-001",
        response_id="response-1",
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=10,
            cached_content_token_count=3,
            candidates_token_count=5,
            thoughts_token_count=7,
            total_token_count=22,
        ),
    )


@unittest.skipUnless(types is not None, "google-genai is installed only in .venv-google")
class GoogleProviderTests(unittest.TestCase):
    def make_provider(self, response=None, error=None):
        provider = GoogleProvider.__new__(GoogleProvider)
        provider.client = pytypes.SimpleNamespace(
            models=FakeModels(response or fake_response(), error=error)
        )
        provider.types = types
        provider.model = "gemini-3.8-flash"
        provider.model_version = None
        provider.temperature = None
        provider.reasoning_effort = "medium"
        provider.max_output_tokens = 4096
        provider._last_model_content = None
        return provider, provider.client.models

    def test_request_and_normalization(self):
        provider, fake_models = self.make_provider()
        turn = provider.complete(
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "task"},
            ],
            [
                {
                    "type": "function",
                    "name": "reserve_restaurant",
                    "description": "Reserve a table",
                    "parameters": {"type": "object", "properties": {}},
                    "strict": True,
                }
            ],
            2,
        )

        request = fake_models.request
        self.assertEqual(request["model"], "gemini-3.8-flash")
        self.assertEqual(request["contents"][0].role, "user")
        self.assertEqual(request["contents"][0].parts[0].text, "task")
        config = request["config"]
        self.assertEqual(config.system_instruction, "system")
        self.assertEqual(config.thinking_config.thinking_level.value, "MEDIUM")
        self.assertTrue(config.automatic_function_calling.disable)
        declaration = config.tools[0].function_declarations[0]
        self.assertEqual(declaration.name, "reserve_restaurant")
        self.assertEqual(declaration.parameters_json_schema["type"], "object")
        self.assertIsNone(config.temperature)

        self.assertEqual(turn.text, "I can help.")
        self.assertEqual(len(turn.tool_calls), 1)
        self.assertEqual(turn.tool_calls[0].call_id, "call-1")
        self.assertEqual(turn.tool_calls[0].arguments, {"email": "a@example.test"})
        self.assertEqual(turn.tool_calls[0].turn_index, 2)
        self.assertEqual(turn.provider_response_id, "response-1")
        self.assertEqual(turn.stop_reason, "STOP")
        self.assertEqual(provider.model_version, "gemini-3.8-flash-001")
        self.assertEqual(turn.usage["input_tokens"], 10)
        self.assertEqual(turn.usage["output_tokens"], 12)
        self.assertEqual(turn.usage["output_tokens_details"]["reasoning_tokens"], 7)

    def test_complete_model_content_is_replayed_on_second_turn(self):
        provider, fake_models = self.make_provider()
        first = provider.complete([{"role": "user", "content": "task"}], [], 0)
        signature = provider._last_model_content.parts[0].thought_signature
        provider.complete(
            [
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": first.text},
                {"role": "user", "content": "Yes"},
            ],
            [],
            1,
        )
        replayed = fake_models.request["contents"][1]
        self.assertEqual(replayed.role, "model")
        self.assertEqual(replayed.parts[0].thought_signature, signature)

    def test_previous_model_content_can_be_restored_from_json(self):
        response = fake_response()
        raw_content = response.candidates[0].content.model_dump(mode="json")
        restored = types.Content.model_validate(raw_content)
        self.assertEqual(restored.parts[0].thought_signature, b"opaque")

    def test_malformed_and_multiple_calls_are_preserved(self):
        provider, _ = self.make_provider(fake_response(malformed=True, second_call=True))
        turn = provider.complete([{"role": "user", "content": "task"}], [], 0)
        self.assertEqual([call.call_id for call in turn.tool_calls], ["call-1", "call-2"])
        self.assertEqual(
            turn.tool_calls[0].arguments,
            {"__invalid_arguments__": "not-an-object"},
        )

    def test_provider_error_propagates(self):
        provider, _ = self.make_provider(error=RuntimeError("synthetic failure"))
        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            provider.complete([{"role": "user", "content": "task"}], [], 0)

    def test_usage_without_metadata_is_zero(self):
        self.assertEqual(_normalized_usage(None)["input_tokens"], 0)
        self.assertEqual(_normalized_usage(None)["output_tokens"], 0)

    def test_invalid_settings_fail_before_sdk_import(self):
        with self.assertRaisesRegex(ValueError, "omit --temperature"):
            GoogleProvider("gemini-3.8-flash", temperature=0.0)
        with self.assertRaisesRegex(ValueError, "low, medium, or high"):
            GoogleProvider("gemini-3.8-flash", reasoning_effort="minimal")


if __name__ == "__main__":
    unittest.main()
