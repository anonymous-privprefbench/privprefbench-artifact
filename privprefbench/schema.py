"""Typed benchmark contracts and validation helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


class PolicyEffect(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"


class Operation(str, Enum):
    DISCLOSE = "DISCLOSE"


class Representation(str, Enum):
    NATURAL = "natural"
    FLAT_JSON = "flat_json"
    STRUCTURED = "structured"


class ConsentCondition(str, Enum):
    GRANT = "GRANT"
    DENY = "DENY"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ConsentRequestLabel(str, Enum):
    VALID_REQUEST = "VALID_REQUEST"
    NOT_REQUEST = "NOT_REQUEST"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class PolicyRule:
    effect: PolicyEffect
    operation: Operation
    data: Tuple[str, ...]
    recipient_type: str
    purpose: str
    exception: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.data or any(not item for item in self.data):
            raise ValueError("PolicyRule.data must contain at least one field")
        if not self.recipient_type or not self.purpose:
            raise ValueError("PolicyRule requires recipient_type and purpose")
        if self.effect is PolicyEffect.DENY and not self.exception:
            raise ValueError("DENY rules require an explicit exception contract")
        if self.effect is PolicyEffect.ALLOW and self.exception is not None:
            raise ValueError("ALLOW rules cannot contain a consent exception")


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    recipient_type: str
    purpose: str
    arguments: Mapping[str, Mapping[str, Any]]
    required: Tuple[str, ...]
    mock_result: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.name or not self.description:
            raise ValueError("ToolDefinition requires a name and description")
        unknown = set(self.required) - set(self.arguments)
        if unknown:
            raise ValueError("Required arguments missing schemas: %s" % sorted(unknown))


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    user_data: Mapping[str, Any]
    task: str
    tool: ToolDefinition
    protected_argument_map: Mapping[str, str]
    data_labels: Mapping[str, str]
    recipient_label: str
    purpose_label: str
    mock_arguments: Mapping[str, Any]
    consent_field_aliases: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    consent_recipient_aliases: Tuple[str, ...] = ()
    consent_purpose_aliases: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        protected = set(self.protected_argument_map)
        if not self.scenario_id or not self.task:
            raise ValueError("Scenario requires scenario_id and task")
        if not protected:
            raise ValueError("Scenario requires protected fields")
        if protected - set(self.user_data):
            raise ValueError("Protected fields must exist in user_data")
        if protected - set(self.data_labels):
            raise ValueError("Each protected field requires a display label")
        if set(self.consent_field_aliases) - protected:
            raise ValueError("Consent field aliases reference unknown protected fields")
        mapped_args = set(self.protected_argument_map.values())
        if mapped_args - set(self.tool.arguments):
            raise ValueError("Protected argument mapping references unknown tool arguments")
        if set(self.tool.required) - set(self.mock_arguments):
            raise ValueError("mock_arguments must provide every required tool argument")
        if self.tool.recipient_type == "" or self.tool.purpose == "":
            raise ValueError("Tool recipient and purpose must be explicit")

    @property
    def protected_fields(self) -> Tuple[str, ...]:
        return tuple(self.protected_argument_map.keys())


@dataclass
class NormalizedToolCall:
    call_id: str
    name: str
    arguments: Dict[str, Any]
    turn_index: int


@dataclass
class NormalizedAssistantTurn:
    turn_index: int
    text: str = ""
    tool_calls: List[NormalizedToolCall] = field(default_factory=list)
    raw_response: Mapping[str, Any] = field(default_factory=dict)
    provider_response_id: Optional[str] = None
    stop_reason: Optional[str] = None
    usage: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ConsentEvent:
    turn_index: int
    condition: ConsentCondition
    data: Tuple[str, ...]
    recipient_type: str
    purpose: str
    scope: str = "CURRENT_TASK"
    text: str = ""


@dataclass
class ToolExecution:
    turn_index: int
    call_id: str
    name: str
    arguments: Mapping[str, Any]
    success: bool
    result: Mapping[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class EvaluationResult:
    unauthorized_disclosure: bool
    asked_for_consent: bool
    task_completed: bool
    unnecessary_consent_request: bool
    override_handled_correctly: Optional[bool]
    needs_human_review: bool
    disclosure_events: List[Mapping[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class RunRecord:
    run_id: str
    timestamp: str
    benchmark_version: str
    evaluator_version: str
    provider: str
    model: str
    model_version: Optional[str]
    temperature: Optional[float]
    reasoning_effort: Optional[str]
    max_output_tokens: Optional[int]
    scenario_id: str
    policy_mode: str
    representation: str
    repeat: int
    consent_condition: str
    system_prompt: str
    user_prompt: str
    privacy_preference: str
    tool_schema: Mapping[str, Any]
    assistant_turns: List[NormalizedAssistantTurn]
    consent_events: List[ConsentEvent]
    tool_executions: List[ToolExecution]
    evaluation: EvaluationResult
    status: str = "completed"
    error: Optional[str] = None


@dataclass
class FirstTurnRecord:
    first_turn_id: str
    timestamp: str
    benchmark_version: str
    provider: str
    model: str
    model_version: Optional[str]
    temperature: Optional[float]
    reasoning_effort: Optional[str]
    max_output_tokens: Optional[int]
    scenario_id: str
    policy_mode: str
    representation: str
    repeat: int
    consent_condition: str
    system_prompt: str
    user_prompt: str
    privacy_preference: str
    tool_schema: Mapping[str, Any]
    assistant_turn: Optional[NormalizedAssistantTurn]
    tool_executions: List[ToolExecution]
    status: str = "completed"
    error: Optional[str] = None


@dataclass(frozen=True)
class ConsentAdjudication:
    first_turn_id: str
    label: ConsentRequestLabel
    adjudicator_id: str
    evidence: str
    action_explicit: bool
    protected_fields_complete: bool
    recipient_or_purpose_scoped: bool
    source_annotators: Tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.first_turn_id or not self.adjudicator_id:
            raise ValueError("ConsentAdjudication requires record and adjudicator IDs")
        if not self.evidence.strip():
            raise ValueError("ConsentAdjudication requires transcript evidence")
        dimensions = (
            self.action_explicit,
            self.protected_fields_complete,
            self.recipient_or_purpose_scoped,
        )
        if any(type(item) is not bool for item in dimensions):
            raise TypeError("Consent adjudication dimensions must be booleans")
        if self.label is ConsentRequestLabel.VALID_REQUEST and not all(
            dimensions
        ):
            raise ValueError("VALID_REQUEST requires all consent dimensions")
        if self.label is not ConsentRequestLabel.VALID_REQUEST and all(dimensions):
            raise ValueError("All consent dimensions imply VALID_REQUEST")


@dataclass
class AdjudicatedRunRecord(RunRecord):
    source_first_turn_id: str = ""
    consent_adjudication: Optional[ConsentAdjudication] = None


def to_jsonable(value: Any) -> Any:
    """Convert dataclasses, enums, and tuples into JSON-safe primitives."""
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value
