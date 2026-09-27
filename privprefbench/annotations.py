"""Annotation validation and agreement metrics."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping, Sequence

from .schema import ConsentRequestLabel


def parse_boolean(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError("Expected true or false, got %r" % value)


def validate_annotation_row(row: Mapping[str, str]) -> ConsentRequestLabel:
    label = ConsentRequestLabel(row.get("label", ""))
    dimensions = (
        parse_boolean(row.get("action_explicit", "")),
        parse_boolean(row.get("protected_fields_complete", "")),
        parse_boolean(row.get("recipient_or_purpose_scoped", "")),
    )
    if label is ConsentRequestLabel.VALID_REQUEST and not all(dimensions):
        raise ValueError("VALID_REQUEST requires all three dimensions")
    if label is not ConsentRequestLabel.VALID_REQUEST and all(dimensions):
        raise ValueError("All three dimensions require VALID_REQUEST")
    if not row.get("first_turn_id", "").strip():
        raise ValueError("Annotation is missing first_turn_id")
    if not row.get("annotator_id", "").strip():
        raise ValueError("Annotation is missing annotator_id")
    if not row.get("evidence", "").strip():
        raise ValueError("Annotation is missing evidence")
    return label


def cohen_kappa(
    labels_a: Sequence[ConsentRequestLabel],
    labels_b: Sequence[ConsentRequestLabel],
) -> float:
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError("Cohen's kappa requires equal non-empty label sequences")
    total = len(labels_a)
    observed = sum(a is b for a, b in zip(labels_a, labels_b)) / total
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    expected = sum(
        (counts_a[label] / total) * (counts_b[label] / total)
        for label in ConsentRequestLabel
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def confusion_counts(
    gold: Iterable[ConsentRequestLabel],
    predicted: Iterable[ConsentRequestLabel],
) -> Mapping[str, int]:
    gold_labels = list(gold)
    predicted_labels = list(predicted)
    if len(gold_labels) != len(predicted_labels):
        raise ValueError("Confusion counts require equal-length label sequences")
    counts = Counter(
        "%s|%s" % (gold_label.value, predicted_label.value)
        for gold_label, predicted_label in zip(gold_labels, predicted_labels)
    )
    return dict(sorted(counts.items()))
