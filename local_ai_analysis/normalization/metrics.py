from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

from local_ai_analysis.db.repository import LocalAIAnalysisDB


@dataclass
class VariantMetrics:
    variant_id: str
    run_id: str | None
    base_model_id: str
    is_baseline: bool
    global_mmlu_lite_pass_at_1: float | None = None
    global_mmlu_lite_micro_pass_at_1: float | None = None
    global_mmlu_lite_invalid_rate: float | None = None
    ifbench_prompt_level_loose: float | None = None
    ifbench_instruction_level_loose: float | None = None
    ifbench_prompt_level_strict: float | None = None
    ifbench_instruction_level_strict: float | None = None
    bfcl_v4_selected_accuracy: float | None = None
    bfcl_v4_invalid_rate: float | None = None
    bfcl_v4_non_live_accuracy: float | None = None
    bfcl_v4_live_accuracy: float | None = None
    bfcl_v4_multi_turn_accuracy: float | None = None
    bfcl_v4_agentic_accuracy: float | None = None
    benchmark_runtime_seconds: float | None = None
    metadata: dict[str, Any] | None = None


def refresh_normalized_results(db: LocalAIAnalysisDB) -> int:
    variants = _load_variant_metrics(db)
    db.conn.execute("DELETE FROM normalized_result")
    inserted = 0
    for base_model_id, base_variants in _group_by_base(variants).items():
        baseline = _select_baseline(base_variants)
        for variant in base_variants:
            db.conn.execute(
                """
                INSERT INTO normalized_result
                (id, variant_id, run_id, base_model_id, baseline_variant_id,
                 global_mmlu_lite_pass_at_1,
                 global_mmlu_lite_micro_pass_at_1, global_mmlu_lite_invalid_rate,
                 ifbench_prompt_level_loose, ifbench_instruction_level_loose,
                 ifbench_prompt_level_strict, ifbench_instruction_level_strict,
                 bfcl_v4_selected_accuracy, bfcl_v4_invalid_rate,
                 bfcl_v4_non_live_accuracy, bfcl_v4_live_accuracy,
                 bfcl_v4_multi_turn_accuracy, bfcl_v4_agentic_accuracy,
                 benchmark_runtime_seconds, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    str(uuid.uuid4()),
                    variant.variant_id,
                    variant.run_id,
                    base_model_id,
                    baseline.variant_id if baseline else variant.variant_id,
                    variant.global_mmlu_lite_pass_at_1,
                    variant.global_mmlu_lite_micro_pass_at_1,
                    variant.global_mmlu_lite_invalid_rate,
                    variant.ifbench_prompt_level_loose,
                    variant.ifbench_instruction_level_loose,
                    variant.ifbench_prompt_level_strict,
                    variant.ifbench_instruction_level_strict,
                    variant.bfcl_v4_selected_accuracy,
                    variant.bfcl_v4_invalid_rate,
                    variant.bfcl_v4_non_live_accuracy,
                    variant.bfcl_v4_live_accuracy,
                    variant.bfcl_v4_multi_turn_accuracy,
                    variant.bfcl_v4_agentic_accuracy,
                    variant.benchmark_runtime_seconds,
                    json.dumps(variant.metadata or {}, sort_keys=True, default=str),
                ],
            )
            inserted += 1
    return inserted


def _load_variant_metrics(db: LocalAIAnalysisDB) -> list[VariantMetrics]:
    rows = db.conn.execute(
        """
        SELECT
            mv.id AS variant_id,
            br.id AS run_id,
            mv.base_model_id,
            mv.is_baseline,
            mv.metadata_json,
            max(CASE WHEN r.metric_name = 'global_mmlu_lite_pass_at_1' THEN r.metric_value END) AS global_mmlu_lite_pass_at_1,
            max(CASE WHEN r.metric_name = 'global_mmlu_lite_micro_pass_at_1' THEN r.metric_value END) AS global_mmlu_lite_micro_pass_at_1,
            max(CASE WHEN r.metric_name = 'global_mmlu_lite_invalid_rate' THEN r.metric_value END) AS global_mmlu_lite_invalid_rate,
            max(CASE WHEN r.metric_name = 'ifbench_prompt_level_loose' THEN r.metric_value END) AS ifbench_prompt_level_loose,
            max(CASE WHEN r.metric_name = 'ifbench_instruction_level_loose' THEN r.metric_value END) AS ifbench_instruction_level_loose,
            max(CASE WHEN r.metric_name = 'ifbench_prompt_level_strict' THEN r.metric_value END) AS ifbench_prompt_level_strict,
            max(CASE WHEN r.metric_name = 'ifbench_instruction_level_strict' THEN r.metric_value END) AS ifbench_instruction_level_strict,
            max(CASE WHEN r.metric_name = 'bfcl_v4_selected_accuracy' THEN r.metric_value END) AS bfcl_v4_selected_accuracy,
            max(CASE WHEN r.metric_name = 'bfcl_v4_invalid_rate' THEN r.metric_value END) AS bfcl_v4_invalid_rate,
            max(CASE WHEN r.metric_name = 'bfcl_v4_non_live_accuracy' THEN r.metric_value END) AS bfcl_v4_non_live_accuracy,
            max(CASE WHEN r.metric_name = 'bfcl_v4_live_accuracy' THEN r.metric_value END) AS bfcl_v4_live_accuracy,
            max(CASE WHEN r.metric_name = 'bfcl_v4_multi_turn_accuracy' THEN r.metric_value END) AS bfcl_v4_multi_turn_accuracy,
            max(CASE WHEN r.metric_name = 'bfcl_v4_agentic_accuracy' THEN r.metric_value END) AS bfcl_v4_agentic_accuracy,
            max(CASE WHEN r.metric_name IN ('global_mmlu_lite_runtime_seconds', 'ifbench_runtime_seconds', 'bfcl_v4_runtime_seconds') THEN r.metric_value END) AS benchmark_runtime_seconds
        FROM model_variant mv
        LEFT JOIN benchmark_result r ON r.variant_id = mv.id
        LEFT JOIN benchmark_run br ON r.run_id = br.id
        GROUP BY mv.id, br.id, mv.base_model_id, mv.is_baseline, mv.metadata_json
        """
    ).fetchall()
    columns = [column[0] for column in db.conn.description]
    variants: list[VariantMetrics] = []
    for row in rows:
        payload = dict(zip(columns, row, strict=True))
        variants.append(
            VariantMetrics(
                variant_id=payload["variant_id"],
                run_id=payload["run_id"],
                base_model_id=payload["base_model_id"],
                is_baseline=bool(payload["is_baseline"]),
                global_mmlu_lite_pass_at_1=_to_float(payload["global_mmlu_lite_pass_at_1"]),
                global_mmlu_lite_micro_pass_at_1=_to_float(
                    payload["global_mmlu_lite_micro_pass_at_1"]
                ),
                global_mmlu_lite_invalid_rate=_to_float(
                    payload["global_mmlu_lite_invalid_rate"]
                ),
                ifbench_prompt_level_loose=_to_float(payload["ifbench_prompt_level_loose"]),
                ifbench_instruction_level_loose=_to_float(
                    payload["ifbench_instruction_level_loose"]
                ),
                ifbench_prompt_level_strict=_to_float(payload["ifbench_prompt_level_strict"]),
                ifbench_instruction_level_strict=_to_float(
                    payload["ifbench_instruction_level_strict"]
                ),
                bfcl_v4_selected_accuracy=_to_float(payload["bfcl_v4_selected_accuracy"]),
                bfcl_v4_invalid_rate=_to_float(payload["bfcl_v4_invalid_rate"]),
                bfcl_v4_non_live_accuracy=_to_float(payload["bfcl_v4_non_live_accuracy"]),
                bfcl_v4_live_accuracy=_to_float(payload["bfcl_v4_live_accuracy"]),
                bfcl_v4_multi_turn_accuracy=_to_float(payload["bfcl_v4_multi_turn_accuracy"]),
                bfcl_v4_agentic_accuracy=_to_float(payload["bfcl_v4_agentic_accuracy"]),
                benchmark_runtime_seconds=_to_float(payload["benchmark_runtime_seconds"]),
                metadata=json.loads(payload["metadata_json"] or "{}"),
            )
        )
    return variants


def _select_baseline(variants: list[VariantMetrics]) -> VariantMetrics | None:
    explicit = [variant for variant in variants if variant.is_baseline]
    return explicit[0] if explicit else (variants[0] if variants else None)


def _group_by_base(variants: list[VariantMetrics]) -> dict[str, list[VariantMetrics]]:
    grouped: dict[str, list[VariantMetrics]] = {}
    for variant in variants:
        grouped.setdefault(variant.base_model_id, []).append(variant)
    return grouped


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
