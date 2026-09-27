"""OpenAI Responses API adapter.

The optional ``openai`` dependency is imported only when this adapter is used.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..schema import NormalizedAssistantTurn, NormalizedToolCall
from .base import Provider


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(
        self,
        model: str,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = "medium",
        max_output_tokens: int = 1000,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI SDK is not installed. Install the optional 'openai' extra."
            ) from exc
        self.client = OpenAI()
        self.model = model
        self.model_version = None
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        tools: Sequence[Mapping[str, Any]],
        turn_index: int,
    ) -> NormalizedAssistantTurn:
        system = "\n\n".join(
            item["content"] for item in messages if item["role"] == "system"
        )
        inputs = [dict(item) for item in messages if item["role"] != "system"]
        request: Dict[str, Any] = {
            "model": self.model,
            "instructions": system,
            "input": inputs,
            "tools": list(tools),
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "max_output_tokens": self.max_output_tokens,
            "store": False,
        }
        if self.reasoning_effort is not None:
            request["reasoning"] = {"effort": self.reasoning_effort}
        if self.temperature is not None:
            request["temperature"] = self.temperature
        response = self.client.responses.create(**request)

        text_parts: List[str] = []
        tool_calls: List[NormalizedToolCall] = []
        for item in response.output:
            item_type = getattr(item, "type", None)
            if item_type == "function_call":
                raw_arguments = getattr(item, "arguments", "{}")
                try:
                    arguments = json.loads(raw_arguments)
                except (TypeError, json.JSONDecodeError):
                    arguments = {"__invalid_json__": raw_arguments}
                tool_calls.append(
                    NormalizedToolCall(
                        call_id=str(getattr(item, "call_id", getattr(item, "id", ""))),
                        name=str(getattr(item, "name", "")),
                        arguments=arguments,
                        turn_index=turn_index,
                    )
                )
            elif item_type == "message":
                for content in getattr(item, "content", []):
                    if getattr(content, "type", None) == "output_text":
                        text_parts.append(str(getattr(content, "text", "")))

        raw = response.model_dump(mode="json")
        usage = response.usage.model_dump(mode="json") if response.usage else {}
        return NormalizedAssistantTurn(
            turn_index=turn_index,
            text="\n".join(part for part in text_parts if part),
            tool_calls=tool_calls,
            raw_response=raw,
            provider_response_id=response.id,
            stop_reason=str(response.status),
            usage=usage,
        )
