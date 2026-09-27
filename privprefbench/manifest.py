"""Reproducibility manifests for immutable JSONL experiment results."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _counter_dict(values: Iterable[Any]) -> Dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def build_result_manifest(
    result_path: Path,
    config_path: Path | None,
    git_commit: str,
) -> Mapping[str, Any]:
    records = []
    with result_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Invalid JSON at %s:%d" % (result_path, line_number)
                ) from exc

    run_ids = [item.get("run_id") for item in records]
    missing_run_ids = sum(not item for item in run_ids)
    cell_counts = Counter(
        (
            str(item.get("scenario_id")),
            str(item.get("policy_mode")),
            str(item.get("representation")),
        )
        for item in records
    )
    cells = {
        "|".join(key): count for key, count in sorted(cell_counts.items())
    }
    manifest: Dict[str, Any] = {
        "manifest_version": "1.0",
        "git_commit": git_commit,
        "result": {
            "path": str(result_path),
            "sha256": sha256_file(result_path),
            "bytes": result_path.stat().st_size,
            "record_count": len(records),
            "unique_run_ids": len(set(run_ids)) - (1 if None in run_ids else 0),
            "missing_run_ids": missing_run_ids,
        },
        "matrix": {
            "unique_cells": len(cell_counts),
            "cell_counts": cells,
            "repeat_count_distribution": _counter_dict(cell_counts.values()),
        },
        "records": {
            "benchmark_versions": _counter_dict(
                item.get("benchmark_version") for item in records
            ),
            "evaluator_versions": _counter_dict(
                item.get("evaluator_version") for item in records
            ),
            "providers": _counter_dict(item.get("provider") for item in records),
            "models": _counter_dict(item.get("model") for item in records),
            "statuses": _counter_dict(item.get("status") for item in records),
        },
    }
    if config_path is not None:
        manifest["config"] = {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "bytes": config_path.stat().st_size,
        }
    return manifest
