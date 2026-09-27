"""Two-pass collection and consent resumption for adjudicated experiments."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, List, Mapping
from uuid import uuid4

from .consent_simulator import consent_response
from .evaluator import evaluate_run
from .policy import build_policy
from .prompts import build_messages, build_prompts
from .providers.base import Provider
from .schema import (
    AdjudicatedRunRecord,
    ConsentAdjudication,
    ConsentCondition,
    ConsentRequestLabel,
    FirstTurnRecord,
    NormalizedAssistantTurn,
    NormalizedToolCall,
    PolicyEffect,
    Representation,
    Scenario,
    ToolExecution,
)
from .tools import build_tool_schema, execute_mock_tool


TWO_PASS_BENCHMARK_VERSION = "0.2.0-dev1"
ADJUDICATED_EVALUATOR_VERSION = "0.3.0-rc6"


def _tool_call_from_mapping(value: Mapping[str, Any]) -> NormalizedToolCall:
    return NormalizedToolCall(
        call_id=str(value["call_id"]),
        name=str(value["name"]),
        arguments=dict(value.get("arguments", {})),
        turn_index=int(value["turn_index"]),
    )


def _assistant_turn_from_mapping(value: Mapping[str, Any]) -> NormalizedAssistantTurn:
    return NormalizedAssistantTurn(
        turn_index=int(value["turn_index"]),
        text=str(value.get("text", "")),
        tool_calls=[
            _tool_call_from_mapping(item) for item in value.get("tool_calls", [])
        ],
        raw_response=dict(value.get("raw_response", {})),
        provider_response_id=value.get("provider_response_id"),
        stop_reason=value.get("stop_reason"),
        usage=dict(value.get("usage", {})),
    )


def _tool_execution_from_mapping(value: Mapping[str, Any]) -> ToolExecution:
    return ToolExecution(
        turn_index=int(value["turn_index"]),
        call_id=str(value["call_id"]),
        name=str(value["name"]),
        arguments=dict(value.get("arguments", {})),
        success=bool(value["success"]),
        result=dict(value.get("result", {})),
        error=value.get("error"),
    )


def first_turn_record_from_mapping(value: Mapping[str, Any]) -> FirstTurnRecord:
    raw_turn = value.get("assistant_turn")
    return FirstTurnRecord(
        first_turn_id=str(value["first_turn_id"]),
        timestamp=str(value["timestamp"]),
        benchmark_version=str(value["benchmark_version"]),
        provider=str(value["provider"]),
        model=str(value["model"]),
        model_version=value.get("model_version"),
        temperature=value.get("temperature"),
        reasoning_effort=value.get("reasoning_effort"),
        max_output_tokens=value.get("max_output_tokens"),
        scenario_id=str(value["scenario_id"]),
        policy_mode=str(value["policy_mode"]),
        representation=str(value["representation"]),
        repeat=int(value["repeat"]),
        consent_condition=str(value["consent_condition"]),
        system_prompt=str(value["system_prompt"]),
        user_prompt=str(value["user_prompt"]),
        privacy_preference=str(value["privacy_preference"]),
        tool_schema=dict(value["tool_schema"]),
        assistant_turn=(
            _assistant_turn_from_mapping(raw_turn) if raw_turn is not None else None
        ),
        tool_executions=[
            _tool_execution_from_mapping(item)
            for item in value.get("tool_executions", [])
        ],
        status=str(value.get("status", "completed")),
        error=value.get("error"),
    )


def adjudication_from_mapping(value: Mapping[str, Any]) -> ConsentAdjudication:
    dimensions = {
        name: value[name]
        for name in (
            "action_explicit",
            "protected_fields_complete",
            "recipient_or_purpose_scoped",
        )
    }
    if any(type(item) is not bool for item in dimensions.values()):
        raise TypeError("Adjudication JSON dimensions must be booleans")
    return ConsentAdjudication(
        first_turn_id=str(value["first_turn_id"]),
        label=ConsentRequestLabel(str(value["label"])),
        adjudicator_id=str(value["adjudicator_id"]),
        evidence=str(value["evidence"]),
        action_explicit=dimensions["action_explicit"],
        protected_fields_complete=dimensions["protected_fields_complete"],
        recipient_or_purpose_scoped=dimensions[
            "recipient_or_purpose_scoped"
        ],
        source_annotators=tuple(str(item) for item in value.get("source_annotators", [])),
        notes=str(value.get("notes", "")),
    )


def _execute_turn_tools(scenario: Scenario, turn) -> List[ToolExecution]:
    return [
        execute_mock_tool(
            scenario=scenario,
            call_id=call.call_id,
            name=call.name,
            arguments=call.arguments,
            turn_index=call.turn_index,
        )
        for call in turn.tool_calls
    ]


def collect_first_turn(
    provider: Provider,
    scenario: Scenario,
    policy_mode: str,
    representation: Representation,
    repeat: int,
    consent_condition: ConsentCondition,
) -> FirstTurnRecord:
    """Collect one initial model turn without running consent detection."""
    rule = build_policy(scenario, policy_mode)
    system_prompt, user_prompt, preference = build_prompts(
        scenario, rule, representation
    )
    messages = build_messages(system_prompt, user_prompt)
    tool_schema = build_tool_schema(scenario)
    assistant_turn = None
    executions: List[ToolExecution] = []
    status = "completed"
    error = None
    try:
        assistant_turn = provider.complete(messages, [tool_schema], turn_index=0)
        executions.extend(_execute_turn_tools(scenario, assistant_turn))
    except Exception as exc:
        status = "provider_error"
        error = "%s: %s" % (exc.__class__.__name__, str(exc))

    effective_condition = (
        consent_condition.value
        if rule.effect is PolicyEffect.DENY
        else ConsentCondition.NOT_APPLICABLE.value
    )
    return FirstTurnRecord(
        first_turn_id=str(uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        benchmark_version=TWO_PASS_BENCHMARK_VERSION,
        provider=provider.name,
        model=provider.model,
        model_version=provider.model_version,
        temperature=provider.temperature,
        reasoning_effort=provider.reasoning_effort,
        max_output_tokens=provider.max_output_tokens,
        scenario_id=scenario.scenario_id,
        policy_mode=rule.effect.value,
        representation=representation.value,
        repeat=repeat,
        consent_condition=effective_condition,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        privacy_preference=preference,
        tool_schema=tool_schema,
        assistant_turn=assistant_turn,
        tool_executions=executions,
        status=status,
        error=error,
    )


def resume_adjudicated_run(
    provider: Provider,
    scenario: Scenario,
    first_turn: FirstTurnRecord,
    adjudication: ConsentAdjudication,
) -> AdjudicatedRunRecord:
    """Finalize a first turn and resume only a resolved valid DENY request."""
    if first_turn.status != "completed" or first_turn.assistant_turn is None:
        raise ValueError("Cannot resume an incomplete first-turn record")
    if adjudication.first_turn_id != first_turn.first_turn_id:
        raise ValueError("Adjudication references a different first-turn record")
    if adjudication.label is ConsentRequestLabel.AMBIGUOUS:
        raise ValueError("Ambiguous adjudication must be resolved before resumption")
    if scenario.scenario_id != first_turn.scenario_id:
        raise ValueError("Scenario does not match the first-turn record")
    if provider.name != first_turn.provider or provider.model != first_turn.model:
        raise ValueError("Resume provider/model must match first-turn collection")

    rule = build_policy(scenario, first_turn.policy_mode.lower())
    representation = Representation(first_turn.representation)
    assistant_turns = [first_turn.assistant_turn]
    executions = list(first_turn.tool_executions)
    consent_events = []
    status = "completed"
    error = None

    should_resume = bool(
        rule.effect is PolicyEffect.DENY
        and adjudication.label is ConsentRequestLabel.VALID_REQUEST
        and not first_turn.assistant_turn.tool_calls
    )
    if should_resume:
        condition = ConsentCondition(first_turn.consent_condition)
        event = consent_response(condition, scenario, rule, turn_index=1)
        consent_events.append(event)
        messages = build_messages(first_turn.system_prompt, first_turn.user_prompt)
        messages.append(
            {"role": "assistant", "content": first_turn.assistant_turn.text}
        )
        messages.append({"role": "user", "content": event.text})
        try:
            second_turn = provider.complete(
                messages, [first_turn.tool_schema], turn_index=2
            )
            assistant_turns.append(second_turn)
            executions.extend(_execute_turn_tools(scenario, second_turn))
        except Exception as exc:
            status = "provider_error"
            error = "%s: %s" % (exc.__class__.__name__, str(exc))

    evaluation = evaluate_run(
        scenario=scenario,
        rule=rule,
        assistant_turns=assistant_turns,
        consent_events=consent_events,
        tool_executions=executions,
        consent_request_label=adjudication.label,
        consent_request_like_override=adjudication.action_explicit,
    )
    if error:
        evaluation.notes.append("Provider error during adjudicated resumption")

    return AdjudicatedRunRecord(
        run_id=str(uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        benchmark_version=TWO_PASS_BENCHMARK_VERSION,
        evaluator_version=ADJUDICATED_EVALUATOR_VERSION,
        provider=first_turn.provider,
        model=first_turn.model,
        model_version=first_turn.model_version,
        temperature=first_turn.temperature,
        reasoning_effort=first_turn.reasoning_effort,
        max_output_tokens=first_turn.max_output_tokens,
        scenario_id=first_turn.scenario_id,
        policy_mode=first_turn.policy_mode,
        representation=representation.value,
        repeat=first_turn.repeat,
        consent_condition=first_turn.consent_condition,
        system_prompt=first_turn.system_prompt,
        user_prompt=first_turn.user_prompt,
        privacy_preference=first_turn.privacy_preference,
        tool_schema=first_turn.tool_schema,
        assistant_turns=assistant_turns,
        consent_events=consent_events,
        tool_executions=executions,
        evaluation=evaluation,
        status=status,
        error=error,
        source_first_turn_id=first_turn.first_turn_id,
        consent_adjudication=adjudication,
    )
