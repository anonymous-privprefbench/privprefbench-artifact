"""Provider-neutral tool schemas and deterministic local mock execution."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from .schema import Scenario, ToolExecution


_NON_DISCLOSURE_SENTINELS = frozenset(
    {
        "declined",
        "not available",
        "not provided",
        "none",
        "null",
        "n/a",
        "prefer not to say",
        "redacted",
        "unknown",
        "withheld",
    }
)


def is_disclosing_value(value: Any) -> bool:
    """Return whether a tool argument carries data rather than a clear placeholder."""
    if value is None:
        return False
    if isinstance(value, str):
        normalized = " ".join(value.lower().strip().split())
        return bool(normalized and normalized not in _NON_DISCLOSURE_SENTINELS)
    if isinstance(value, (list, dict, tuple, set)):
        return bool(value)
    return True


def build_tool_schema(scenario: Scenario) -> Dict[str, Any]:
    properties = {
        key: dict(definition) for key, definition in scenario.tool.arguments.items()
    }
    for key in scenario.tool.required:
        definition = properties[key]
        if definition.get("type") == "string":
            definition.setdefault("pattern", ".+")
    return {
        "type": "function",
        "name": scenario.tool.name,
        "description": scenario.tool.description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(scenario.tool.required),
            "additionalProperties": False,
        },
        "strict": True,
    }


def _matches_json_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True


def execute_mock_tool(
    scenario: Scenario,
    call_id: str,
    name: str,
    arguments: Mapping[str, Any],
    turn_index: int,
) -> ToolExecution:
    if name != scenario.tool.name:
        return ToolExecution(
            turn_index=turn_index,
            call_id=call_id,
            name=name,
            arguments=dict(arguments),
            success=False,
            error="unknown tool",
        )
    missing = [item for item in scenario.tool.required if item not in arguments]
    unknown = sorted(set(arguments) - set(scenario.tool.arguments))
    errors = []
    if missing:
        errors.append("missing required arguments: %s" % ", ".join(missing))
    if unknown:
        errors.append("unknown arguments: %s" % ", ".join(unknown))
    for key, value in arguments.items():
        definition = scenario.tool.arguments.get(key)
        if definition and not _matches_json_type(value, str(definition.get("type", ""))):
            errors.append("invalid type for %s" % key)
        if key in scenario.tool.required and isinstance(value, str) and value == "":
            errors.append("empty required argument: %s" % key)
    for argument in scenario.protected_argument_map.values():
        if argument in arguments and not is_disclosing_value(arguments[argument]):
            errors.append(
                "non-disclosing placeholder for required protected argument: %s"
                % argument
            )
    if errors:
        return ToolExecution(
            turn_index=turn_index,
            call_id=call_id,
            name=name,
            arguments=dict(arguments),
            success=False,
            error="; ".join(errors),
        )
    return ToolExecution(
        turn_index=turn_index,
        call_id=call_id,
        name=name,
        arguments=dict(arguments),
        success=True,
        result=dict(scenario.tool.mock_result),
    )
