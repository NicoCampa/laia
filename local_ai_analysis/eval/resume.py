from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


ProgressCallback = Callable[[str, dict[str, Any]], None]


def load_resume_records(
    path: Path,
    *,
    key: Callable[[dict[str, Any]], str | None],
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_key = key(record)
            if record_key is not None:
                records[record_key] = record
    return records


def record_matches(record: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(record.get(key) == value for key, value in expected.items())


def sample_file_mode(resume_records: dict[str, dict[str, Any]]) -> str:
    return "a" if resume_records else "w"


def maybe_announce_resume(
    *,
    progress_callback: ProgressCallback | None,
    announced: bool,
    task: str,
    variant: str,
    completed_samples: int,
    total_samples: int,
    correct_samples: float | int | None = None,
    invalid_samples: int | None = None,
    runtime_seconds: float | None = None,
) -> bool:
    if announced or completed_samples <= 0:
        return announced
    if progress_callback:
        payload: dict[str, Any] = {
            "task": task,
            "variant": variant,
            "completed_samples": completed_samples,
            "total_samples": total_samples,
        }
        if correct_samples is not None:
            payload["correct_samples"] = correct_samples
        if invalid_samples is not None:
            payload["invalid_samples"] = invalid_samples
        if runtime_seconds is not None:
            payload["runtime_seconds"] = runtime_seconds
        progress_callback("task_resume", payload)
    return True


def cache_key(*parts: Any) -> str:
    return "|".join("" if part is None else str(part) for part in parts)
