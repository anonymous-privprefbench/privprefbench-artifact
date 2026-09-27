"""Google Gemini GenerateContent adapter.

The optional ``google-genai`` dependency is imported only when this adapter is
used.  Gemini thought signatures are retained between turns because Gemini 3
requires the complete model content to be replayed for function calling.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..schema import NormalizedAssistantTurn, NormalizedToolCall
from .base import Provider


SUPPORTED_THINKING_LEVELS = {"low", "medium", "high"}


def _enum_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return "" if raw is None else str(raw)


def _normalized_usage(value: Any) -> Dict[str, Any]:
    raw = value.model_dump(mode="json") if value is not None else {}
    candidates = int(raw.get("candidates_token_count", 0) or 0)
    thoughts = int(raw.get("thoughts_token_count", 0) or 0)
    return {
        "input_tokens": int(raw.get("prompt_token_count", 0) or 0),
        "output_tokens": candidates + thoughts,
        "input_tokens_details": {
            "cached_tokens": int(raw.get("cached_content_token_count", 0) or 0),
            "cache_write_tokens": 0,
        },
        "output_tokens_details": {"reasoning_tokens": thoughts},
        "provider_usage": raw,
    }


class GoogleProvider(Provider):
    name = "google"

    def __init__(
        self,
        model: str,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = "medium",
        max_output_tokens: int = 4096,
        previous_model_content: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if model == "gemini-3.8-flash" and temperature is not None:
            raise ValueError(
                "Gemini 3.8 Flash should not receive explicit temperature; "
                "omit --temperature"
            )
        if reasoning_effort not in SUPPORTED_THINKING_LEVELS:
            raise ValueError(
                "Gemini 3.8 Flash thinking level must be low, medium, or high"
            )
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "Google Gen AI SDK is not installed. Use the project's "
                ".venv-google environment."
            ) from exc
        self.client = genai.Client(
            http_options=types.HttpOptions(
                timeout=120_000,
                retry_options=types.HttpRetryOptions(
                    attempts=3,
                    initial_delay=1.0,
                    max_delay=8.0,
                ),
            )
        )
        self.types = types
        self.model = model
        self.model_version = None
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self._last_model_content = (
            types.Content.model_validate(previous_model_content)
            if previous_model_content is not None
            else None
        )

    def _tool(self, tool: Mapping[str, Any]):
        if tool.get("type") != "function":
            raise ValueError("Google adapter supports only function tools")
        name = str(tool.get("name", ""))
        parameters = tool.get("parameters")
        if not name or not isinstance(parameters, Mapping):
            raise ValueError("Function tools require a name and parameters schema")
        return self.types.FunctionDeclaration(
            name=name,
            description=str(tool.get("description", "")) or None,
            parameters_json_schema=dict(parameters),
        )

    def _contents(self, messages: Sequence[Mapping[str, Any]]) -> List[Any]:
        conversation = [item for item in messages if item.get("role") != "system"]
        contents: List[Any] = []
        assistant_indexes = [
            index for index, item in enumerate(conversation)
            if item.get("role") == "assistant"
        ]
        last_assistant = assistant_indexes[-1] if assistant_indexes else None
        for index, item in enumerate(conversation):
            role = item.get("role")
            if index == last_assistant and self._last_model_content is not None:
                contents.append(self._last_model_content)
                continue
            if role not in {"user", "assistant"}:
                raise ValueError("Google conversation roles must be user or assistant")
            contents.append(
                self.types.Content(
                    role="model" if role == "assistant" else "user",
                    parts=[self.types.Part(text=str(item.get("content", "")))],
                )
            )
        return contents

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        turn_index: int,
    ) -> NormalizedAssistantTurn:
        system = "\n\n".join(
            str(item.get("content", ""))
            for item in messages
            if item.get("role") == "system"
        )
        config: Dict[str, Any] = {
            "system_instruction": system or None,
            "max_output_tokens": self.max_output_tokens,
            "thinking_config": self.types.ThinkingConfig(
                thinking_level=self.reasoning_effort
            ),
        }
        if tools:
            config["tools"] = [
                self.types.Tool(function_declarations=[self._tool(tool) for tool in tools])
            ]
            config["automatic_function_calling"] = (
                self.types.AutomaticFunctionCallingConfig(disable=True)
            )

        response = self.client.models.generate_content(
            model=self.model,
            contents=self._contents(messages),
            config=self.types.GenerateContentConfig(**config),
        )
        candidates = list(response.candidates or [])
        candidate = candidates[0] if candidates else None
        content = getattr(candidate, "content", None)
        if content is not None:
            self._last_model_content = content

        text_parts: List[str] = []
        tool_calls: List[NormalizedToolCall] = []
        for index, part in enumerate(getattr(content, "parts", None) or []):
            if getattr(part, "text", None) and not getattr(part, "thought", False):
                text_parts.append(str(part.text))
            function_call = getattr(part, "function_call", None)
            if function_call is None:
                continue
            raw_args = getattr(function_call, "args", {})
            arguments = (
                dict(raw_args)
                if isinstance(raw_args, Mapping)
                else {"__invalid_arguments__": raw_args}
            )
            call_id = getattr(function_call, "id", None)
            tool_calls.append(
                NormalizedToolCall(
                    call_id=str(call_id or "gemini-call-%d-%d" % (turn_index, index)),
                    name=str(getattr(function_call, "name", "") or ""),
                    arguments=arguments,
                    turn_index=turn_index,
                )
            )

        raw = response.model_dump(mode="json")
        served_model = str(getattr(response, "model_version", "") or "")
        self.model_version = served_model or None
        return NormalizedAssistantTurn(
            turn_index=turn_index,
            text="\n".join(part for part in text_parts if part),
            tool_calls=tool_calls,
            raw_response=raw,
            provider_response_id=str(getattr(response, "response_id", "") or ""),
            stop_reason=_enum_value(getattr(candidate, "finish_reason", None)),
            usage=_normalized_usage(getattr(response, "usage_metadata", None)),
        )
