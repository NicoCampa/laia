from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from local_ai_analysis.db import LocalAIAnalysisDB


def leaderboard_payload(db_path: str | Path) -> dict[str, Any]:
    db = LocalAIAnalysisDB(db_path)
    try:
        db.init_schema()
        language_breakdowns = _global_mmlu_language_breakdowns(db)
        rgb_language_breakdowns = _rgb_language_breakdowns(db)
        rows = []
        for row in db.leaderboard_rows():
            if not (
                row.get("global_mmlu_lite_pass_at_1") is not None
                or row.get("ifbench_prompt_level_loose") is not None
                or row.get("bfcl_v4_selected_accuracy") is not None
                or row.get("ocrbench_v2_score") is not None
                or row.get("mmmu_accuracy") is not None
                or row.get("mbpp_pass_at_1") is not None
                or row.get("rgb_all_rate") is not None
                or row.get("simpleqa_f1") is not None
                or row.get("harmbench_refusal_rate") is not None
            ):
                continue
            public = _public_row(row)
            language_breakdown = language_breakdowns.get(
                (str(row.get("variant_id")), str(row.get("run_uuid")))
            )
            if language_breakdown:
                public["global_mmlu_lite_language_scores"] = language_breakdown
            rgb_language_breakdown = rgb_language_breakdowns.get(
                (str(row.get("variant_id")), str(row.get("run_uuid")))
            )
            if rgb_language_breakdown:
                public["rgb_language_scores"] = rgb_language_breakdown
            rows.append(public)
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "tagline": "Reproducible local benchmark results for API-served AI models.",
            "leaderboard": rows,
            "filters": _filters(rows),
        }
    finally:
        db.close()


def export_leaderboard(db_path: str | Path, out: str | Path, fmt: str) -> dict[str, Any]:
    fmt = fmt.lower()
    payload = leaderboard_payload(db_path)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True, default=str)
    elif fmt == "csv":
        rows = payload["leaderboard"]
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
    else:
        raise ValueError("format must be one of: json, csv")
    return {"out": str(out_path), "format": fmt, "rows": len(payload["leaderboard"])}


def _filters(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    keys = [
        "family",
        "parameter_size_b",
        "quantization",
        "backend_name",
        "hardware_accelerator",
    ]
    filters: dict[str, list[Any]] = {}
    for key in keys:
        values = sorted({row.get(key) for row in rows if row.get(key) is not None})
        filters[key] = values
    return filters


def _public_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "normalized_result_id",
        "variant_id",
        "base_model_id",
        "family",
        "base_model_name",
        "parameter_size_b",
        "variant_name",
        "quantization",
        "precision",
        "model_repo",
        "file_name",
        "checksum_sha256",
        "file_size_bytes",
        "is_baseline",
        "backend_name",
        "backend_version",
        "backend_commit",
        "hardware_hash",
        "hardware_accelerator",
        "cpu_model",
        "gpu_name",
        "run_uuid",
        "started_at",
        "global_mmlu_lite_pass_at_1",
        "global_mmlu_lite_micro_pass_at_1",
        "global_mmlu_lite_invalid_rate",
        "ifbench_prompt_level_loose",
        "ifbench_instruction_level_loose",
        "ifbench_prompt_level_strict",
        "ifbench_instruction_level_strict",
        "bfcl_v4_selected_accuracy",
        "bfcl_v4_invalid_rate",
        "bfcl_v4_non_live_accuracy",
        "bfcl_v4_live_accuracy",
        "bfcl_v4_multi_turn_accuracy",
        "bfcl_v4_agentic_accuracy",
        "ocrbench_v2_score",
        "ocrbench_v2_micro_score",
        "ocrbench_v2_en_score",
        "ocrbench_v2_cn_score",
        "mmmu_accuracy",
        "mmmu_invalid_rate",
        "mmmu_multiple_choice_accuracy",
        "mmmu_open_accuracy",
        "mbpp_pass_at_1",
        "mbpp_invalid_rate",
        "mbpp_compile_rate",
        "mbpp_runtime_error_rate",
        "rgb_all_rate",
        "rgb_rejection_rate",
        "rgb_fact_check_rate",
        "rgb_error_correction_rate",
        "simpleqa_f1",
        "simpleqa_correct_rate",
        "simpleqa_incorrect_rate",
        "simpleqa_hallucination_rate",
        "simpleqa_not_attempted_rate",
        "simpleqa_accuracy_given_attempted",
        "harmbench_attack_success_rate",
        "harmbench_refusal_rate",
        "model_intelligence_score",
        "model_intelligence_coverage",
        "model_intelligence_available_score",
        "benchmark_runtime_seconds",
        "benchmark_samples",
        "benchmark_correct_count",
        "benchmark_prompt_tokens",
        "benchmark_completion_tokens",
        "benchmark_total_tokens",
        "benchmark_reasoning_tokens",
        "benchmark_output_tokens_per_second",
        "benchmark_total_tokens_per_second",
        "benchmark_avg_latency_seconds",
        "benchmark_p50_latency_seconds",
        "benchmark_p95_latency_seconds",
        "benchmark_truncated_count",
        "benchmark_truncated_rate",
        "benchmark_tokens_per_correct_answer",
        "benchmark_seconds_per_correct_answer",
        "benchmark_time_to_first_token_seconds",
        "benchmark_inter_token_latency_seconds",
        "benchmark_end_to_end_latency_seconds",
        "benchmark_system_output_throughput_tokens_per_second",
        "benchmark_input_cost_usd",
        "benchmark_output_cost_usd",
        "benchmark_total_cost_usd",
        "benchmark_cost_per_correct_answer_usd",
        "metadata_json",
    ]
    return {key: row.get(key) for key in keys}


def _global_mmlu_language_breakdowns(
    db: LocalAIAnalysisDB,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    rows = db.conn.execute(
        """
        SELECT r.variant_id, br.run_uuid, r.raw_json
        FROM benchmark_result r
        JOIN benchmark_run br ON r.run_id = br.id
        WHERE r.metric_name = 'global_mmlu_lite_pass_at_1'
        """
    ).fetchall()
    breakdowns: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for variant_id, run_uuid, raw_json in rows:
        try:
            raw = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
        except json.JSONDecodeError:
            continue
        summary = raw.get("summary") if isinstance(raw, dict) else None
        if not isinstance(summary, dict):
            continue
        languages = [str(language) for language in summary.get("languages") or []]
        language_accuracy = summary.get("language_accuracy") or {}
        if not languages or not isinstance(language_accuracy, dict):
            continue
        sample_counts = _global_mmlu_sample_counts(
            summary.get("samples_path"),
            project_root=db.db_path.parent.parent,
        )
        language_scores: list[dict[str, Any]] = []
        for language in languages:
            value = language_accuracy.get(language)
            if value is None:
                continue
            counts = sample_counts.get(language, {})
            language_scores.append(
                {
                    "language": language,
                    "accuracy": value,
                    "correct": counts.get("correct"),
                    "total": counts.get("total"),
                    "invalid": counts.get("invalid"),
                    "invalid_rate": (
                        counts["invalid"] / counts["total"]
                        if counts.get("total")
                        else None
                    ),
                }
            )
        if language_scores:
            breakdowns[(str(variant_id), str(run_uuid))] = language_scores
    return breakdowns


def _global_mmlu_sample_counts(
    samples_path_value: Any,
    *,
    project_root: Path,
) -> dict[str, dict[str, int]]:
    if not samples_path_value:
        return {}
    samples_path = Path(str(samples_path_value))
    candidates = [samples_path]
    if not samples_path.is_absolute():
        candidates = [
            project_root / samples_path,
            Path.cwd() / samples_path,
            samples_path,
        ]
    existing = next((candidate for candidate in candidates if candidate.exists()), None)
    if existing is None:
        return {}

    counts: dict[str, dict[str, int]] = {}
    with existing.open("r", encoding="utf-8") as sample_file:
        for line in sample_file:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            language = str(record.get("language") or "")
            if not language:
                continue
            bucket = counts.setdefault(language, {"total": 0, "correct": 0, "invalid": 0})
            bucket["total"] += 1
            bucket["correct"] += int(bool(record.get("correct")))
            bucket["invalid"] += int(bool(record.get("invalid")))
    return counts


def _rgb_language_breakdowns(
    db: LocalAIAnalysisDB,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    rows = db.conn.execute(
        """
        SELECT r.variant_id, br.run_uuid, r.raw_json
        FROM benchmark_result r
        JOIN benchmark_run br ON r.run_id = br.id
        WHERE r.metric_name = 'rgb_all_rate'
        """
    ).fetchall()
    breakdowns: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for variant_id, run_uuid, raw_json in rows:
        try:
            raw = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
        except json.JSONDecodeError:
            continue
        summary = raw.get("summary") if isinstance(raw, dict) else None
        if not isinstance(summary, dict):
            continue

        language_scores = _rgb_suite_language_scores(summary)
        if not language_scores:
            language_scores = _rgb_single_language_scores(summary)
        if language_scores:
            breakdowns[(str(variant_id), str(run_uuid))] = language_scores
    return breakdowns


RGB_COMPONENT_WEIGHTS = {
    "noise_robustness": 0.30,
    "negative_rejection": 0.25,
    "information_integration": 0.25,
    "error_detection": 0.20,
}


def _rgb_suite_language_scores(summary: dict[str, Any]) -> list[dict[str, Any]]:
    suite_cases = summary.get("suite_cases")
    if not isinstance(suite_cases, list):
        return []

    by_language: dict[str, dict[str, Any]] = {}
    for case in suite_cases:
        if not isinstance(case, dict):
            continue
        language = _rgb_language(case.get("language") or case.get("dataset"))
        component = str(case.get("component") or "")
        score = _to_float(case.get("score"))
        if not language or not component or score is None:
            continue
        total = int(_to_float(case.get("total")) or 0)
        bucket = by_language.setdefault(
            language,
            {
                "total": 0,
                "component_scores": {},
                "component_totals": {},
            },
        )
        bucket["total"] += total
        component_scores = bucket["component_scores"].setdefault(
            component,
            {"weighted_score": 0.0, "total": 0},
        )
        weight = total or 1
        component_scores["weighted_score"] += score * weight
        component_scores["total"] += weight
        bucket["component_totals"][component] = bucket["component_totals"].get(component, 0) + total

    language_scores: list[dict[str, Any]] = []
    for language, bucket in by_language.items():
        components: dict[str, float] = {}
        weighted_sum = 0.0
        covered_weight = 0.0
        for component, values in bucket["component_scores"].items():
            total = values["total"]
            if not total:
                continue
            component_score = values["weighted_score"] / total
            components[component] = component_score
            component_weight = RGB_COMPONENT_WEIGHTS.get(component, 0.0)
            weighted_sum += component_score * component_weight
            covered_weight += component_weight
        if not components or covered_weight == 0:
            continue
        language_scores.append(
            {
                "language": language,
                "accuracy": weighted_sum / covered_weight,
                "total": bucket["total"],
                "components": components,
                "component_totals": bucket["component_totals"],
            }
        )

    return sorted(language_scores, key=lambda score: str(score["language"]))


def _rgb_single_language_scores(summary: dict[str, Any]) -> list[dict[str, Any]]:
    language = _rgb_language(summary.get("rgb_language") or summary.get("rgb_dataset"))
    score = _to_float(_first_present(summary.get("rgb_all_rate"), summary.get("rgb_accuracy")))
    if not language or score is None:
        return []
    return [
        {
            "language": language,
            "accuracy": score,
            "total": _to_float(summary.get("total")),
        }
    ]


def _rgb_language(value: Any) -> str | None:
    text = str(value or "").lower()
    if text in {"en", "english"} or text.startswith("en_") or "_en" in text:
        return "en"
    if text in {"zh", "chinese", "cn"} or text.startswith("zh_") or "_zh" in text:
        return "zh"
    return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_present(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)
