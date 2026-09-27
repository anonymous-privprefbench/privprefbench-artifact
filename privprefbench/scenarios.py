"""Load dependency-free, JSON-compatible YAML scenario fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .schema import Scenario, ToolDefinition


DEFAULT_SCENARIO_DIR = Path(__file__).resolve().parent.parent / "scenarios"


def _require_keys(
    data: Mapping[str, Any], required: set, context: str, optional: Optional[set] = None
) -> None:
    optional = optional or set()
    missing = required - set(data)
    extra = set(data) - required - optional
    if missing:
        raise ValueError("%s missing keys: %s" % (context, sorted(missing)))
    if extra:
        raise ValueError("%s has unknown keys: %s" % (context, sorted(extra)))


def scenario_from_dict(data: Mapping[str, Any]) -> Scenario:
    _require_keys(
        data,
        {
            "scenario_id",
            "user_data",
            "task",
            "tool",
            "protected_argument_map",
            "data_labels",
            "recipient_label",
            "purpose_label",
            "mock_arguments",
        },
        "scenario",
        {
            "consent_field_aliases",
            "consent_recipient_aliases",
            "consent_purpose_aliases",
        },
    )
    tool_data = data["tool"]
    _require_keys(
        tool_data,
        {
            "name",
            "description",
            "recipient_type",
            "purpose",
            "arguments",
            "required",
            "mock_result",
        },
        "tool",
    )
    tool = ToolDefinition(
        name=str(tool_data["name"]),
        description=str(tool_data["description"]),
        recipient_type=str(tool_data["recipient_type"]),
        purpose=str(tool_data["purpose"]),
        arguments=dict(tool_data["arguments"]),
        required=tuple(tool_data["required"]),
        mock_result=dict(tool_data["mock_result"]),
    )
    return Scenario(
        scenario_id=str(data["scenario_id"]),
        user_data=dict(data["user_data"]),
        task=str(data["task"]),
        tool=tool,
        protected_argument_map=dict(data["protected_argument_map"]),
        data_labels=dict(data["data_labels"]),
        recipient_label=str(data["recipient_label"]),
        purpose_label=str(data["purpose_label"]),
        mock_arguments=dict(data["mock_arguments"]),
        consent_field_aliases={
            str(key): tuple(str(item) for item in values)
            for key, values in data.get("consent_field_aliases", {}).items()
        },
        consent_recipient_aliases=tuple(
            str(item) for item in data.get("consent_recipient_aliases", [])
        ),
        consent_purpose_aliases=tuple(
            str(item) for item in data.get("consent_purpose_aliases", [])
        ),
    )


def load_scenario(scenario_id: str, directory: Optional[Path] = None) -> Scenario:
    root = directory or DEFAULT_SCENARIO_DIR
    path = root / (scenario_id + ".yaml")
    if not path.exists():
        known = ", ".join(list_scenario_ids(root))
        raise ValueError("Unknown scenario %r. Available: %s" % (scenario_id, known))
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    scenario = scenario_from_dict(data)
    if scenario.scenario_id != scenario_id:
        raise ValueError("Scenario id does not match filename: %s" % path)
    return scenario


def list_scenario_ids(directory: Optional[Path] = None) -> List[str]:
    root = directory or DEFAULT_SCENARIO_DIR
    return sorted(path.stem for path in root.glob("*.yaml"))


def load_all_scenarios(directory: Optional[Path] = None) -> List[Scenario]:
    root = directory or DEFAULT_SCENARIO_DIR
    return [load_scenario(item, root) for item in list_scenario_ids(root)]
