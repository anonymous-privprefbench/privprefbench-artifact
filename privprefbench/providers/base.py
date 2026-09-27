"""Provider protocol and deterministic pipeline-test provider."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Sequence

from ..schema import (
    ConsentCondition,
    NormalizedAssistantTurn,
    NormalizedToolCall,
    PolicyEffect,
    Scenario,
)


class Provider(ABC):
    name = "unknown"
    model = "unknown"
    model_version = None
    temperature = None
    reasoning_effort = None
    max_output_tokens = None

    @abstractmethod
    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        tools: Sequence[Mapping[str, Any]],
        turn_index: int,
    ) -> NormalizedAssistantTurn:
        raise NotImplementedError


class ScriptedProvider(Provider):
    """Deterministic provider used only to test the benchmark machinery."""

    name = "mock"
    model = "scripted-compliant-baseline"
    model_version = "0.1"
    temperature = 0.0

    def __init__(
        self,
        scenario: Scenario,
        policy_effect: PolicyEffect,
        consent_condition: ConsentCondition,
    ) -> None:
        self.scenario = scenario
        self.policy_effect = policy_effect
        self.consent_condition = consent_condition
        self._calls = 0

    def _tool_turn(self, turn_index: int) -> NormalizedAssistantTurn:
        call = NormalizedToolCall(
            call_id="mock-call-%d" % self._calls,
            name=self.scenario.tool.name,
            arguments=dict(self.scenario.mock_arguments),
            turn_index=turn_index,
        )
        return NormalizedAssistantTurn(
            turn_index=turn_index,
            text="",
            tool_calls=[call],
            raw_response={"kind": "scripted_tool_call"},
            stop_reason="tool_call",
        )

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        tools: Sequence[Mapping[str, Any]],
        turn_index: int,
    ) -> NormalizedAssistantTurn:
        self._calls += 1
        if turn_index == 0 and self.policy_effect is PolicyEffect.ALLOW:
            return self._tool_turn(turn_index)
        if turn_index == 0:
            fields = " and ".join(
                self.scenario.data_labels[item]
                for item in self.scenario.protected_fields
            )
            text = (
                "This task requires disclosing your {fields} to {recipient} for "
                "{purpose}. May I make that disclosure for this task?"
            ).format(
                fields=fields,
                recipient=self.scenario.recipient_label,
                purpose=self.scenario.purpose_label,
            )
            return NormalizedAssistantTurn(
                turn_index=turn_index,
                text=text,
                raw_response={"kind": "scripted_consent_request"},
                stop_reason="completed",
            )
        if self.consent_condition is ConsentCondition.GRANT:
            return self._tool_turn(turn_index)
        return NormalizedAssistantTurn(
            turn_index=turn_index,
            text="I will not disclose the protected data, so I cannot complete the tool action.",
            raw_response={"kind": "scripted_refusal"},
            stop_reason="completed",
        )
