"""Canonical policy construction."""

from .schema import Operation, PolicyEffect, PolicyRule, Scenario


CONSENT_EXCEPTION = "EXPLICIT_CURRENT_TASK_CONSENT"


def build_policy(scenario: Scenario, mode: str) -> PolicyRule:
    normalized = mode.strip().upper()
    if normalized == PolicyEffect.ALLOW.value:
        return PolicyRule(
            effect=PolicyEffect.ALLOW,
            operation=Operation.DISCLOSE,
            data=scenario.protected_fields,
            recipient_type=scenario.tool.recipient_type,
            purpose=scenario.tool.purpose,
        )
    if normalized == PolicyEffect.DENY.value:
        return PolicyRule(
            effect=PolicyEffect.DENY,
            operation=Operation.DISCLOSE,
            data=scenario.protected_fields,
            recipient_type=scenario.tool.recipient_type,
            purpose=scenario.tool.purpose,
            exception=CONSENT_EXCEPTION,
        )
    raise ValueError("Unknown policy mode: %s" % mode)

