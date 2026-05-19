from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any

from local_ai_analysis.metrics import MetricResult


def efficiency_metrics_from_summary(summary: dict[str, Any]) -> list[MetricResult]:
    samples_path = summary.get("samples_path")
    if not samples_path and isinstance(summary.get("suite_cases"), list):
        records = []
        for suite_case in summary["suite_cases"]:
            if isinstance(suite_case, dict) and suite_case.get("samples_path"):
                records.extend(_read_records(Path(str(suite_case["samples_path"]))))
    elif not samples_path:
        return _null_streaming_metrics(summary)
    else:
        records = _read_records(Path(str(samples_path)))
    if not records:
        return _null_streaming_metrics(summary)

    runtime_values = [_float(record.get("runtime_seconds")) for record in records]
    runtimes = [value for value in runtime_values if value is not None]
    total_runtime = sum(runtimes)
    samples = len(records)
    correct = sum(1 for record in records if _is_correct(record))
    truncated = sum(1 for record in records if _is_truncated(record, summary))

    usage_totals = _usage_totals(records)
    prompt_tokens = usage_totals.get("prompt_tokens")
    completion_tokens = usage_totals.get("completion_tokens")
    total_tokens = usage_totals.get("total_tokens")
    reasoning_tokens = usage_totals.get("reasoning_tokens")

    raw = {"summary": summary, "source": "sample_jsonl_efficiency_v1"}
    metrics = [
        MetricResult("benchmark_samples", float(samples), "samples", raw),
        MetricResult("benchmark_correct_count", float(correct), "count", raw),
        MetricResult("benchmark_prompt_tokens", _as_float(prompt_tokens), "tokens", raw),
        MetricResult(
            "benchmark_completion_tokens", _as_float(completion_tokens), "tokens", raw
        ),
        MetricResult("benchmark_total_tokens", _as_float(total_tokens), "tokens", raw),
        MetricResult(
            "benchmark_reasoning_tokens", _as_float(reasoning_tokens), "tokens", raw
        ),
        MetricResult("benchmark_truncated_count", float(truncated), "count", raw),
        MetricResult(
            "benchmark_truncated_rate",
            truncated / samples if samples else None,
            "fraction",
            raw,
        ),
        MetricResult(
            "benchmark_output_tokens_per_second",
            _safe_div(completion_tokens, total_runtime),
            "tokens_per_second",
            raw,
        ),
        MetricResult(
            "benchmark_total_tokens_per_second",
            _safe_div(total_tokens, total_runtime),
            "tokens_per_second",
            raw,
        ),
        MetricResult(
            "benchmark_avg_latency_seconds",
            total_runtime / len(runtimes) if runtimes else None,
            "seconds",
            raw,
        ),
        MetricResult("benchmark_p50_latency_seconds", _percentile(runtimes, 50), "seconds", raw),
        MetricResult("benchmark_p95_latency_seconds", _percentile(runtimes, 95), "seconds", raw),
        MetricResult(
            "benchmark_tokens_per_correct_answer",
            _safe_div(total_tokens, correct),
            "tokens_per_correct",
            raw,
        ),
        MetricResult(
            "benchmark_seconds_per_correct_answer",
            _safe_div(total_runtime, correct),
            "seconds_per_correct",
            raw,
        ),
    ]
    metrics.extend(_null_streaming_metrics(summary))
    return metrics


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
    return records


def _usage_totals(records: list[dict[str, Any]]) -> dict[str, int | None]:
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
    }
    seen = set()
    for record in records:
        for key in ("usage", "grader_usage", "judge_usage"):
            usage = record.get(key)
            if not isinstance(usage, dict):
                continue
            _add_usage(totals, seen, usage)
    return {key: value if key in seen else None for key, value in totals.items()}


def _add_usage(totals: dict[str, int], seen: set[str], usage: dict[str, Any]) -> None:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = _int(usage.get(key))
        if value is not None:
            totals[key] += value
            seen.add(key)
    details = usage.get("completion_tokens_details")
    if isinstance(details, dict):
        reasoning = _int(details.get("reasoning_tokens"))
        if reasoning is not None:
            totals["reasoning_tokens"] += reasoning
            seen.add("reasoning_tokens")


def _is_correct(record: dict[str, Any]) -> bool:
    if "correct" in record:
        return bool(record.get("correct"))
    if "passed" in record:
        return bool(record.get("passed"))
    if "loose_follow_all" in record:
        return bool(record.get("loose_follow_all"))
    if record.get("grade") == "correct":
        return True
    if "attack_success" in record:
        return not bool(record.get("attack_success"))
    return False


def _is_truncated(record: dict[str, Any], summary: dict[str, Any]) -> bool:
    for key in ("raw_response", "grader_raw_response", "judge_raw_response"):
        if _finish_reason(record.get(key)) == "length":
            return True
    if str(record.get("finish_reason") or "").lower() == "length":
        return True

    max_tokens = _int(summary.get("max_tokens"))
    if max_tokens is not None and _usage_reached_limit(record.get("usage"), max_tokens):
        return True

    grader_max_tokens = _int(summary.get("grader_max_tokens"))
    if grader_max_tokens is not None and _usage_reached_limit(
        record.get("grader_usage"), grader_max_tokens
    ):
        return True

    judge_max_tokens = _int(summary.get("judge_max_tokens"))
    if judge_max_tokens is not None and _usage_reached_limit(
        record.get("judge_usage"), judge_max_tokens
    ):
        return True

    return False


def _usage_reached_limit(usage: Any, max_tokens: int) -> bool:
    if not isinstance(usage, dict) or max_tokens <= 0:
        return False
    completion_tokens = _int(usage.get("completion_tokens"))
    return completion_tokens is not None and completion_tokens >= max_tokens


def _finish_reason(raw: Any) -> str | None:
    if not isinstance(raw, dict):
        return None
    choices = raw.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict) and first.get("finish_reason") is not None:
            return str(first.get("finish_reason")).lower()
    if raw.get("done_reason") is not None:
        return str(raw.get("done_reason")).lower()
    if raw.get("stop_reason") is not None:
        return str(raw.get("stop_reason")).lower()
    return None


def _null_streaming_metrics(summary: dict[str, Any]) -> list[MetricResult]:
    raw = {
        "summary": summary,
        "note": "Streaming latency metrics require a streaming adapter run.",
    }
    return [
        MetricResult("benchmark_time_to_first_token_seconds", None, "seconds", raw),
        MetricResult("benchmark_inter_token_latency_seconds", None, "seconds", raw),
        MetricResult("benchmark_end_to_end_latency_seconds", None, "seconds", raw),
        MetricResult(
            "benchmark_system_output_throughput_tokens_per_second",
            None,
            "tokens_per_second",
            raw,
        ),
        MetricResult("benchmark_input_cost_usd", None, "usd", raw),
        MetricResult("benchmark_output_cost_usd", None, "usd", raw),
        MetricResult("benchmark_total_cost_usd", None, "usd", raw),
        MetricResult("benchmark_cost_per_correct_answer_usd", None, "usd_per_correct", raw),
    ]


def _percentile(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if percentile == 50:
        return float(median(sorted_values))
    index = int(round((percentile / 100) * (len(sorted_values) - 1)))
    return float(sorted_values[index])


def _safe_div(numerator: int | float | None, denominator: int | float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _as_float(value: int | float | None) -> float | None:
    return float(value) if value is not None else None


def _float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
