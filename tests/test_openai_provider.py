import sys
import types
import unittest
from unittest.mock import patch

from privprefbench.providers.openai_provider import OpenAIProvider


class Dumpable:
    def __init__(self, value):
        self.value = value

    def model_dump(self, mode="json"):
        return self.value


class FakeResponses:
    def __init__(self, response):
        self.response = response
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return self.response


class FakeResponse:
    def __init__(self):
        content = types.SimpleNamespace(type="output_text", text="I can help.")
        message = types.SimpleNamespace(type="message", content=[content])
        function = types.SimpleNamespace(
            type="function_call",
            call_id="call-1",
            id="item-1",
            name="reserve_restaurant",
            arguments='{"customer_email":"alice@example.test"}',
        )
        self.output = [message, function]
        self.id = "resp-1"
        self.status = "completed"
        self.usage = Dumpable({"input_tokens": 10, "output_tokens": 5})

    def model_dump(self, mode="json"):
        return {"id": self.id, "status": self.status}


class OpenAIProviderTests(unittest.TestCase):
    def make_provider(self, response, temperature=None, reasoning_effort="medium"):
        fake_responses = FakeResponses(response)
        fake_client = types.SimpleNamespace(responses=fake_responses)
        fake_module = types.ModuleType("openai")
        fake_module.OpenAI = lambda: fake_client

        with patch.dict(sys.modules, {"openai": fake_module}):
            provider = OpenAIProvider(
                "test-model",
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )
        return provider, fake_responses

    def test_responses_api_request_and_normalization(self):
        response = FakeResponse()
        provider, fake_responses = self.make_provider(response)
        turn = provider.complete(
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "task"},
            ],
            [{"type": "function", "name": "reserve_restaurant"}],
            turn_index=0,
        )

        self.assertEqual(turn.text, "I can help.")
        self.assertEqual(turn.tool_calls[0].name, "reserve_restaurant")
        self.assertEqual(
            turn.tool_calls[0].arguments, {"customer_email": "alice@example.test"}
        )
        self.assertEqual(fake_responses.request["instructions"], "system")
        self.assertEqual(fake_responses.request["input"], [{"role": "user", "content": "task"}])
        self.assertEqual(fake_responses.request["tool_choice"], "auto")
        self.assertFalse(fake_responses.request["parallel_tool_calls"])
        self.assertFalse(fake_responses.request["store"])
        self.assertEqual(fake_responses.request["reasoning"], {"effort": "medium"})
        self.assertNotIn("temperature", fake_responses.request)

    def test_explicit_temperature_is_forwarded(self):
        provider, fake_responses = self.make_provider(FakeResponse(), temperature=0.0)

        provider.complete(
            [{"role": "user", "content": "task"}],
            [],
            turn_index=0,
        )

        self.assertEqual(fake_responses.request["temperature"], 0.0)


if __name__ == "__main__":
    unittest.main()
