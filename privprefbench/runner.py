"""Single-run orchestration and event logging."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List
from uuid import uuid4

from .consent_simulator import consent_response
from .evaluator import EVALUATOR_VERSION, evaluate_run, is_valid_consent_request
from .policy import build_policy
from .prompts import build_messages, build_prompts
from .providers.base import Provider
from .schema import (
    ConsentCondition,
    ConsentEvent,
    EvaluationResult,
    PolicyEffect,
    Representation,
    RunRecord,
    Scenario,
    ToolExecution,
)
from .tools import build_tool_schema, execute_mock_tool


BENCHMARK_VERSION = "0.1.4"


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


def run_once(
    provider: Provider,
    scenario: Scenario,
    policy_mode: str,
    representation: Representation,
    repeat: int,
    consent_condition: ConsentCondition,
) -> RunRecord:
    rule = build_policy(scenario, policy_mode)
    system_prompt, user_prompt, preference = build_prompts(
        scenario, rule, representation
    )
    messages = build_messages(system_prompt, user_prompt)
    tool_schema = build_tool_schema(scenario)
    assistant_turns = []
    consent_events: List[ConsentEvent] = []
    executions: List[ToolExecution] = []
    status = "completed"
    error = None

    try:
        first_turn = provider.complete(messages, [tool_schema], turn_index=0)
        assistant_turns.append(first_turn)
        executions.extend(_execute_turn_tools(scenario, first_turn))

        should_follow_up = bool(
            rule.effect is PolicyEffect.DENY
            and not first_turn.tool_calls
            and is_valid_consent_request(first_turn.text, scenario, rule)
        )
        if should_follow_up:
            event = consent_response(
                consent_condition, scenario, rule, turn_index=1
            )
            consent_events.append(event)
            messages.append({"role": "assistant", "content": first_turn.text})
            messages.append({"role": "user", "content": event.text})
            second_turn = provider.complete(messages, [tool_schema], turn_index=2)
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
    )
    if error:
        evaluation.notes.append("Provider error; not classified as a refusal")

    effective_condition = (
        consent_condition.value
        if rule.effect is PolicyEffect.DENY
        else ConsentCondition.NOT_APPLICABLE.value
    )
    return RunRecord(
        run_id=str(uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        benchmark_version=BENCHMARK_VERSION,
        evaluator_version=EVALUATOR_VERSION,
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
        assistant_turns=assistant_turns,
        consent_events=consent_events,
        tool_executions=executions,
        evaluation=evaluation,
        status=status,
        error=error,
    )
