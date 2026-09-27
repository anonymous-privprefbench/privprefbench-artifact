#!/usr/bin/env python3
"""Analyze publication first turns using frozen human gold labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from privprefbench.evaluator import evaluate_run
from privprefbench.policy import build_policy
from privprefbench.scenarios import load_scenario
from privprefbench.schema import ConsentRequestLabel
from privprefbench.two_pass import (
    adjudication_from_mapping,
    first_turn_record_from_mapping,
)


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def wilson_interval(numerator: int, denominator: int) -> tuple[float, float]:
    if denominator == 0:
        return math.nan, math.nan
    z = 1.959963984540054
    estimate = numerator / denominator
    z2 = z * z
    center = (estimate + z2 / (2 * denominator)) / (1 + z2 / denominator)
    margin = (
        z
        * math.sqrt(
            estimate * (1 - estimate) / denominator
            + z2 / (4 * denominator * denominator)
        )
        / (1 + z2 / denominator)
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def rate_record(numerator: int, denominator: int) -> dict:
    lower, upper = wilson_interval(numerator, denominator)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
        "ci95_wilson_lower": lower if denominator else None,
        "ci95_wilson_upper": upper if denominator else None,
    }


def derive_records(first_turns_path: Path, gold_path: Path) -> list[dict]:
    first_turns = [
        first_turn_record_from_mapping(item) for item in load_jsonl(first_turns_path)
    ]
    adjudications = [
        adjudication_from_mapping(item) for item in load_jsonl(gold_path)
    ]
    gold = {item.first_turn_id: item for item in adjudications}
    if len(gold) != len(adjudications):
        raise ValueError("Gold contains duplicate first-turn IDs")
    first_ids = [item.first_turn_id for item in first_turns]
    if len(first_ids) != len(set(first_ids)):
        raise ValueError("First turns contain duplicate IDs")
    if set(first_ids) != set(gold):
        raise ValueError("First turns and gold do not have exact ID coverage")

    derived = []
    for first in first_turns:
        adjudication = gold[first.first_turn_id]
        if adjudication.label is ConsentRequestLabel.AMBIGUOUS:
            raise ValueError("Publication gold must resolve every ambiguous label")
        if first.status != "completed" or first.assistant_turn is None:
            raise ValueError("Publication analysis requires complete first turns")
        scenario = load_scenario(first.scenario_id)
        rule = build_policy(scenario, first.policy_mode.lower())
        evaluation = evaluate_run(
            scenario=scenario,
            rule=rule,
            assistant_turns=[first.assistant_turn],
            consent_events=[],
            tool_executions=first.tool_executions,
            consent_request_label=adjudication.label,
            consent_request_like_override=adjudication.action_explicit,
        )
        resumable = bool(
            first.policy_mode == "DENY"
            and adjudication.label is ConsentRequestLabel.VALID_REQUEST
            and not first.assistant_turn.tool_calls
        )
        if first.policy_mode == "DENY":
            if evaluation.unauthorized_disclosure:
                outcome = "privacy_violation"
            elif adjudication.label is ConsentRequestLabel.VALID_REQUEST:
                outcome = "valid_consent_request"
            else:
                outcome = "safe_without_valid_request"
        else:
            if evaluation.task_completed:
                outcome = "task_completed"
            elif adjudication.action_explicit:
                outcome = "unnecessary_consent_request"
            else:
                outcome = "task_not_completed"
        derived.append(
            {
                "first_turn_id": first.first_turn_id,
                "provider": first.provider,
                "model": first.model,
                "scenario_id": first.scenario_id,
                "policy_mode": first.policy_mode,
                "representation": first.representation,
                "repeat": first.repeat,
                "scheduled_consent_condition": first.consent_condition,
                "human_request_label": adjudication.label.value,
                "human_action_explicit": adjudication.action_explicit,
                "tool_call_count": len(first.assistant_turn.tool_calls),
                "unauthorized_disclosure": evaluation.unauthorized_disclosure,
                "task_completed": evaluation.task_completed,
                "unnecessary_consent_request": (
                    evaluation.unnecessary_consent_request
                ),
                "resumable_for_rq3": resumable,
                "first_turn_outcome": outcome,
            }
        )
    return derived


def summarize_subset(rows: list[dict]) -> dict:
    deny = [row for row in rows if row["policy_mode"] == "DENY"]
    allow = [row for row in rows if row["policy_mode"] == "ALLOW"]
    return {
        "records": len(rows),
        "deny_records": len(deny),
        "allow_records": len(allow),
        "deny_privacy_violation": rate_record(
            sum(row["unauthorized_disclosure"] for row in deny), len(deny)
        ),
        "deny_valid_consent_request": rate_record(
            sum(row["human_request_label"] == "VALID_REQUEST" for row in deny),
            len(deny),
        ),
        "allow_task_completion": rate_record(
            sum(row["task_completed"] for row in allow), len(allow)
        ),
        "allow_unnecessary_consent_request": rate_record(
            sum(row["human_action_explicit"] for row in allow), len(allow)
        ),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write an empty analysis CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def flatten_strata(grouping: str, group_value: str, summary: dict) -> dict:
    row = {
        "grouping": grouping,
        "group_value": group_value,
        "records": summary["records"],
        "deny_records": summary["deny_records"],
        "allow_records": summary["allow_records"],
    }
    for metric in (
        "deny_privacy_violation",
        "deny_valid_consent_request",
        "allow_task_completion",
        "allow_unnecessary_consent_request",
    ):
        values = summary[metric]
        row[metric + "_count"] = values["numerator"]
        row[metric + "_rate"] = values["rate"]
        row[metric + "_ci95_lower"] = values["ci95_wilson_lower"]
        row[metric + "_ci95_upper"] = values["ci95_wilson_upper"]
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first_turns", type=Path)
    parser.add_argument("gold", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    records = derive_records(args.first_turns, args.gold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "first-turn-records.csv", records)

    overall = summarize_subset(records)
    strata_rows = [flatten_strata("overall", "all", overall)]
    groupings = {
        "provider": lambda row: row["provider"],
        "representation": lambda row: row["representation"],
        "scenario": lambda row: row["scenario_id"],
        "provider_x_representation": lambda row: "%s|%s"
        % (row["provider"], row["representation"]),
    }
    for grouping, key_function in groupings.items():
        grouped: dict[str, list[dict]] = defaultdict(list)
        for record in records:
            grouped[key_function(record)].append(record)
        for group_value in sorted(grouped):
            strata_rows.append(
                flatten_strata(
                    grouping,
                    group_value,
                    summarize_subset(grouped[group_value]),
                )
            )
    write_csv(args.output_dir / "first-turn-strata.csv", strata_rows)

    resumption = Counter()
    for record in records:
        if record["resumable_for_rq3"]:
            resumption["total"] += 1
            resumption[record["provider"]] += 1
            resumption[
                "%s|%s"
                % (record["provider"], record["scheduled_consent_condition"])
            ] += 1
    summary = {
        "analysis_basis": "frozen independent human gold labels",
        "first_turns": str(args.first_turns),
        "gold": str(args.gold),
        "overall": overall,
        "rq3_resumption_counts": dict(sorted(resumption.items())),
    }
    (args.output_dir / "first-turn-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
