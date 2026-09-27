import sys
import types
import unittest
from unittest.mock import patch

from privprefbench.providers.anthropic_provider import AnthropicProvider


class Dumpable:
    def __init__(self, value):
        self.value = value

    def model_dump(self, mode="json"):
        return self.value


class FakeMessages:
    def __init__(self, response, error=None):
        self.response = response
        self.error = error
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        if self.error is not None:
            raise self.error
        return self.response


class FakeResponse:
    def __init__(self, malformed=False):
        self.content = [
            types.SimpleNamespace(type="thinking", thinking="hidden"),
            types.SimpleNamespace(type="text", text="I can help."),
            types.SimpleNamespace(
                type="tool_use",
                id="toolu-1",
                name="reserve_restaurant",
                input=(
                    "not-an-object"
                    if malformed
                    else {"customer_email": "alice@example.test"}
                ),
            ),
        ]
        self.id = "msg-1"
        self.model = "claude-sonnet-5"
        self.stop_reason = "tool_use"
        self.usage = Dumpable(
            {
                "input_tokens": 10,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 2,
                "output_tokens": 5,
            }
        )

    def model_dump(self, mode="json"):
        return {"id": self.id, "model": self.model}


class AnthropicProviderTests(unittest.TestCase):
    def make_provider(
        self, response=None, temperature=None, reasoning_effort="medium"
    ):
        fake_messages = FakeMessages(response or FakeResponse())
        fake_client = types.SimpleNamespace(messages=fake_messages)
        fake_module = types.ModuleType("anthropic")
        fake_module.Anthropic = lambda: fake_client
        with patch.dict(sys.modules, {"anthropic": fake_module}):
            provider = AnthropicProvider(
                "claude-sonnet-5",
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )
        return provider, fake_messages

    def test_messages_request_and_normalization(self):
        provider, fake_messages = self.make_provider()
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
            turn_index=2,
        )

        request = fake_messages.request
        self.assertEqual(request["system"], "system")
        self.assertEqual(request["messages"], [{"role": "user", "content": "task"}])
        self.assertEqual(request["output_config"], {"effort": "medium"})
        self.assertEqual(
            request["tool_choice"],
            {"type": "auto", "disable_parallel_tool_use": True},
        )
        self.assertEqual(
            request["tools"],
            [
                {
                    "name": "reserve_restaurant",
                    "description": "Reserve a table",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
        )
        self.assertNotIn("temperature", request)
        self.assertEqual(turn.text, "I can help.")
        self.assertEqual(len(turn.tool_calls), 1)
        self.assertEqual(turn.tool_calls[0].call_id, "toolu-1")
        self.assertEqual(turn.tool_calls[0].turn_index, 2)
        self.assertEqual(
            turn.tool_calls[0].arguments,
            {"customer_email": "alice@example.test"},
        )
        self.assertEqual(turn.stop_reason, "tool_use")
        self.assertEqual(turn.provider_response_id, "msg-1")
        self.assertEqual(provider.model_version, "claude-sonnet-5")
        self.assertEqual(turn.usage["input_tokens"], 15)
        self.assertEqual(turn.usage["output_tokens"], 5)
        self.assertEqual(turn.usage["input_tokens_details"]["cached_tokens"], 3)

    def test_none_effort_disables_thinking(self):
        provider, fake_messages = self.make_provider(reasoning_effort="none")
        provider.complete([{"role": "user", "content": "task"}], [], 0)
        self.assertEqual(fake_messages.request["thinking"], {"type": "disabled"})
        self.assertNotIn("output_config", fake_messages.request)

    def test_structured_tool_result_history_is_preserved(self):
        provider, fake_messages = self.make_provider()
        history = [
            {"role": "user", "content": "task"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu-prior",
                        "name": "lookup",
                        "input": {"query": "value"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu-prior",
                        "content": "done",
                    }
                ],
            },
        ]
        provider.complete(history, [], 2)
        self.assertEqual(fake_messages.request["messages"], history)
        self.assertNotIn("tools", fake_messages.request)
        self.assertNotIn("tool_choice", fake_messages.request)

    def test_malformed_tool_input_is_preserved_for_validation(self):
        provider, _ = self.make_provider(FakeResponse(malformed=True))
        turn = provider.complete([{"role": "user", "content": "task"}], [], 0)
        self.assertEqual(
            turn.tool_calls[0].arguments,
            {"__invalid_arguments__": "not-an-object"},
        )

    def test_multiple_tool_calls_are_normalized(self):
        response = FakeResponse()
        response.content.append(
            types.SimpleNamespace(
                type="tool_use",
                id="toolu-2",
                name="send_confirmation",
                input={"channel": "email"},
            )
        )
        provider, _ = self.make_provider(response)
        turn = provider.complete([{"role": "user", "content": "task"}], [], 0)
        self.assertEqual([call.call_id for call in turn.tool_calls], ["toolu-1", "toolu-2"])

    def test_provider_error_propagates_to_runner_boundary(self):
        failure = RuntimeError("synthetic provider failure")
        fake_messages = FakeMessages(FakeResponse(), error=failure)
        fake_client = types.SimpleNamespace(messages=fake_messages)
        fake_module = types.ModuleType("anthropic")
        fake_module.Anthropic = lambda: fake_client
        with patch.dict(sys.modules, {"anthropic": fake_module}):
            provider = AnthropicProvider("claude-sonnet-5")
        with self.assertRaisesRegex(RuntimeError, "synthetic provider failure"):
            provider.complete([{"role": "user", "content": "task"}], [], 0)

    def test_sonnet_5_rejects_explicit_temperature_before_sdk_call(self):
        with self.assertRaisesRegex(ValueError, "omit --temperature"):
            AnthropicProvider("claude-sonnet-5", temperature=0.0)


if __name__ == "__main__":
    unittest.main()
