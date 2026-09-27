"""Anthropic Messages API adapter.

The optional ``anthropic`` dependency is imported only when this adapter is used.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..schema import NormalizedAssistantTurn, NormalizedToolCall
from .base import Provider


def _anthropic_tool(tool: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert a provider-neutral function schema to an Anthropic client tool."""
    if tool.get("type") != "function":
        raise ValueError("Anthropic adapter supports only function tools")
    name = str(tool.get("name", ""))
    parameters = tool.get("parameters")
    if not name or not isinstance(parameters, Mapping):
        raise ValueError("Function tools require a name and parameters schema")
    converted: Dict[str, Any] = {
        "name": name,
        "input_schema": dict(parameters),
    }
    description = tool.get("description")
    if description:
        converted["description"] = str(description)
    return converted


def _normalized_usage(value: Any) -> Dict[str, Any]:
    raw = value.model_dump(mode="json") if value is not None else {}
    base_input = int(raw.get("input_tokens", 0) or 0)
    cache_read = int(raw.get("cache_read_input_tokens", 0) or 0)
    cache_write = int(raw.get("cache_creation_input_tokens", 0) or 0)
    return {
        "input_tokens": base_input + cache_read + cache_write,
        "output_tokens": int(raw.get("output_tokens", 0) or 0),
        "input_tokens_details": {
            "cached_tokens": cache_read,
            "cache_write_tokens": cache_write,
        },
        "provider_usage": raw,
    }


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(
        self,
        model: str,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = "medium",
        max_output_tokens: int = 4096,
    ) -> None:
        if model == "claude-sonnet-5" and temperature is not None:
            raise ValueError(
                "Claude Sonnet 5 does not accept explicit sampling temperature; "
                "omit --temperature"
            )
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError(
                "Anthropic SDK is not installed. Install the optional "
                "'anthropic' extra."
            ) from exc
        self.client = Anthropic()
        self.model = model
        self.model_version = None
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        turn_index: int,
    ) -> NormalizedAssistantTurn:
        system = "\n\n".join(
            item["content"] for item in messages if item["role"] == "system"
        )
        conversation = [dict(item) for item in messages if item["role"] != "system"]
        request: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "messages": conversation,
        }
        if tools:
            request["tools"] = [_anthropic_tool(tool) for tool in tools]
            request["tool_choice"] = {
                "type": "auto",
                "disable_parallel_tool_use": True,
            }
        if system:
            request["system"] = system
        if self.reasoning_effort and self.reasoning_effort != "none":
            request["output_config"] = {"effort": self.reasoning_effort}
        elif self.reasoning_effort == "none":
            request["thinking"] = {"type": "disabled"}
        if self.temperature is not None:
            request["temperature"] = self.temperature

        response = self.client.messages.create(**request)
        text_parts: List[str] = []
        tool_calls: List[NormalizedToolCall] = []
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(str(getattr(block, "text", "")))
            elif block_type == "tool_use":
                raw_input = getattr(block, "input", {})
                arguments = (
                    dict(raw_input)
                    if isinstance(raw_input, Mapping)
                    else {"__invalid_arguments__": raw_input}
                )
                tool_calls.append(
                    NormalizedToolCall(
                        call_id=str(getattr(block, "id", "")),
                        name=str(getattr(block, "name", "")),
                        arguments=arguments,
                        turn_index=turn_index,
                    )
                )

        raw = response.model_dump(mode="json")
        served_model = str(getattr(response, "model", "") or "")
        self.model_version = served_model or None
        return NormalizedAssistantTurn(
            turn_index=turn_index,
            text="\n".join(part for part in text_parts if part),
            tool_calls=tool_calls,
            raw_response=raw,
            provider_response_id=str(response.id),
            stop_reason=str(response.stop_reason),
            usage=_normalized_usage(getattr(response, "usage", None)),
        )
