from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

from local_ai_analysis.db.repository import LocalAIAnalysisDB


MODEL_INTELLIGENCE_WEIGHTS = {
    "global_mmlu_lite_pass_at_1": 0.25,
    "ifbench_prompt_level_loose": 0.20,
    "bfcl_v4_selected_accuracy": 0.20,
    "mbpp_pass_at_1": 0.20,
    "rgb_all_rate": 0.15,
}


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
    ocrbench_v2_score: float | None = None
    ocrbench_v2_micro_score: float | None = None
    ocrbench_v2_en_score: float | None = None
    ocrbench_v2_cn_score: float | None = None
    mmmu_accuracy: float | None = None
    mmmu_invalid_rate: float | None = None
    mmmu_multiple_choice_accuracy: float | None = None
    mmmu_open_accuracy: float | None = None
    mbpp_pass_at_1: float | None = None
    mbpp_invalid_rate: float | None = None
    mbpp_compile_rate: float | None = None
    mbpp_runtime_error_rate: float | None = None
    rgb_all_rate: float | None = None
    rgb_rejection_rate: float | None = None
    rgb_fact_check_rate: float | None = None
    rgb_error_correction_rate: float | None = None
    simpleqa_f1: float | None = None
    simpleqa_correct_rate: float | None = None
    simpleqa_incorrect_rate: float | None = None
    simpleqa_hallucination_rate: float | None = None
    simpleqa_not_attempted_rate: float | None = None
    simpleqa_accuracy_given_attempted: float | None = None
    harmbench_attack_success_rate: float | None = None
    harmbench_refusal_rate: float | None = None
    model_intelligence_score: float | None = None
    model_intelligence_coverage: float | None = None
    model_intelligence_available_score: float | None = None
    benchmark_runtime_seconds: float | None = None
    benchmark_samples: float | None = None
    benchmark_correct_count: float | None = None
    benchmark_prompt_tokens: float | None = None
    benchmark_completion_tokens: float | None = None
    benchmark_total_tokens: float | None = None
    benchmark_reasoning_tokens: float | None = None
    benchmark_output_tokens_per_second: float | None = None
    benchmark_total_tokens_per_second: float | None = None
    benchmark_avg_latency_seconds: float | None = None
    benchmark_p50_latency_seconds: float | None = None
    benchmark_p95_latency_seconds: float | None = None
    benchmark_truncated_count: float | None = None
    benchmark_truncated_rate: float | None = None
    benchmark_tokens_per_correct_answer: float | None = None
    benchmark_seconds_per_correct_answer: float | None = None
    benchmark_time_to_first_token_seconds: float | None = None
    benchmark_inter_token_latency_seconds: float | None = None
    benchmark_end_to_end_latency_seconds: float | None = None
    benchmark_system_output_throughput_tokens_per_second: float | None = None
    benchmark_input_cost_usd: float | None = None
    benchmark_output_cost_usd: float | None = None
    benchmark_total_cost_usd: float | None = None
    benchmark_cost_per_correct_answer_usd: float | None = None
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
                 ocrbench_v2_score, ocrbench_v2_micro_score,
                 ocrbench_v2_en_score, ocrbench_v2_cn_score,
                 mmmu_accuracy, mmmu_invalid_rate,
                 mmmu_multiple_choice_accuracy, mmmu_open_accuracy,
                 mbpp_pass_at_1, mbpp_invalid_rate,
                 mbpp_compile_rate, mbpp_runtime_error_rate,
                 rgb_all_rate, rgb_rejection_rate,
                 rgb_fact_check_rate, rgb_error_correction_rate,
                 simpleqa_f1, simpleqa_correct_rate,
                 simpleqa_incorrect_rate, simpleqa_hallucination_rate,
                 simpleqa_not_attempted_rate, simpleqa_accuracy_given_attempted,
                 harmbench_attack_success_rate, harmbench_refusal_rate,
                 model_intelligence_score, model_intelligence_coverage,
                 model_intelligence_available_score,
                 benchmark_runtime_seconds,
                 benchmark_samples, benchmark_correct_count,
                 benchmark_prompt_tokens, benchmark_completion_tokens,
                 benchmark_total_tokens, benchmark_reasoning_tokens,
                 benchmark_output_tokens_per_second,
                 benchmark_total_tokens_per_second,
                 benchmark_avg_latency_seconds,
                 benchmark_p50_latency_seconds,
                 benchmark_p95_latency_seconds,
                 benchmark_truncated_count, benchmark_truncated_rate,
                 benchmark_tokens_per_correct_answer,
                 benchmark_seconds_per_correct_answer,
                 benchmark_time_to_first_token_seconds,
                 benchmark_inter_token_latency_seconds,
                 benchmark_end_to_end_latency_seconds,
                 benchmark_system_output_throughput_tokens_per_second,
                 benchmark_input_cost_usd, benchmark_output_cost_usd,
                 benchmark_total_cost_usd,
                 benchmark_cost_per_correct_answer_usd,
                 metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    variant.ocrbench_v2_score,
                    variant.ocrbench_v2_micro_score,
                    variant.ocrbench_v2_en_score,
                    variant.ocrbench_v2_cn_score,
                    variant.mmmu_accuracy,
                    variant.mmmu_invalid_rate,
                    variant.mmmu_multiple_choice_accuracy,
                    variant.mmmu_open_accuracy,
                    variant.mbpp_pass_at_1,
                    variant.mbpp_invalid_rate,
                    variant.mbpp_compile_rate,
                    variant.mbpp_runtime_error_rate,
                    variant.rgb_all_rate,
                    variant.rgb_rejection_rate,
                    variant.rgb_fact_check_rate,
                    variant.rgb_error_correction_rate,
                    variant.simpleqa_f1,
                    variant.simpleqa_correct_rate,
                    variant.simpleqa_incorrect_rate,
                    variant.simpleqa_hallucination_rate,
                    variant.simpleqa_not_attempted_rate,
                    variant.simpleqa_accuracy_given_attempted,
                    variant.harmbench_attack_success_rate,
                    variant.harmbench_refusal_rate,
                    variant.model_intelligence_score,
                    variant.model_intelligence_coverage,
                    variant.model_intelligence_available_score,
                    variant.benchmark_runtime_seconds,
                    variant.benchmark_samples,
                    variant.benchmark_correct_count,
                    variant.benchmark_prompt_tokens,
                    variant.benchmark_completion_tokens,
                    variant.benchmark_total_tokens,
                    variant.benchmark_reasoning_tokens,
                    variant.benchmark_output_tokens_per_second,
                    variant.benchmark_total_tokens_per_second,
                    variant.benchmark_avg_latency_seconds,
                    variant.benchmark_p50_latency_seconds,
                    variant.benchmark_p95_latency_seconds,
                    variant.benchmark_truncated_count,
                    variant.benchmark_truncated_rate,
                    variant.benchmark_tokens_per_correct_answer,
                    variant.benchmark_seconds_per_correct_answer,
                    variant.benchmark_time_to_first_token_seconds,
                    variant.benchmark_inter_token_latency_seconds,
                    variant.benchmark_end_to_end_latency_seconds,
                    variant.benchmark_system_output_throughput_tokens_per_second,
                    variant.benchmark_input_cost_usd,
                    variant.benchmark_output_cost_usd,
                    variant.benchmark_total_cost_usd,
                    variant.benchmark_cost_per_correct_answer_usd,
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
            max(CASE WHEN r.metric_name = 'ocrbench_v2_score' THEN r.metric_value END) AS ocrbench_v2_score,
            max(CASE WHEN r.metric_name = 'ocrbench_v2_micro_score' THEN r.metric_value END) AS ocrbench_v2_micro_score,
            max(CASE WHEN r.metric_name = 'ocrbench_v2_en_score' THEN r.metric_value END) AS ocrbench_v2_en_score,
            max(CASE WHEN r.metric_name = 'ocrbench_v2_cn_score' THEN r.metric_value END) AS ocrbench_v2_cn_score,
            max(CASE WHEN r.metric_name = 'mmmu_accuracy' THEN r.metric_value END) AS mmmu_accuracy,
            max(CASE WHEN r.metric_name = 'mmmu_invalid_rate' THEN r.metric_value END) AS mmmu_invalid_rate,
            max(CASE WHEN r.metric_name = 'mmmu_multiple_choice_accuracy' THEN r.metric_value END) AS mmmu_multiple_choice_accuracy,
            max(CASE WHEN r.metric_name = 'mmmu_open_accuracy' THEN r.metric_value END) AS mmmu_open_accuracy,
            max(CASE WHEN r.metric_name = 'mbpp_pass_at_1' THEN r.metric_value END) AS mbpp_pass_at_1,
            max(CASE WHEN r.metric_name = 'mbpp_invalid_rate' THEN r.metric_value END) AS mbpp_invalid_rate,
            max(CASE WHEN r.metric_name = 'mbpp_compile_rate' THEN r.metric_value END) AS mbpp_compile_rate,
            max(CASE WHEN r.metric_name = 'mbpp_runtime_error_rate' THEN r.metric_value END) AS mbpp_runtime_error_rate,
            max(CASE WHEN r.metric_name = 'rgb_all_rate' THEN r.metric_value END) AS rgb_all_rate,
            max(CASE WHEN r.metric_name = 'rgb_rejection_rate' THEN r.metric_value END) AS rgb_rejection_rate,
            max(CASE WHEN r.metric_name = 'rgb_fact_check_rate' THEN r.metric_value END) AS rgb_fact_check_rate,
            max(CASE WHEN r.metric_name = 'rgb_error_correction_rate' THEN r.metric_value END) AS rgb_error_correction_rate,
            max(CASE WHEN r.metric_name = 'simpleqa_f1' THEN r.metric_value END) AS simpleqa_f1,
            max(CASE WHEN r.metric_name = 'simpleqa_correct_rate' THEN r.metric_value END) AS simpleqa_correct_rate,
            max(CASE WHEN r.metric_name = 'simpleqa_incorrect_rate' THEN r.metric_value END) AS simpleqa_incorrect_rate,
            max(CASE WHEN r.metric_name = 'simpleqa_hallucination_rate' THEN r.metric_value END) AS simpleqa_hallucination_rate,
            max(CASE WHEN r.metric_name = 'simpleqa_not_attempted_rate' THEN r.metric_value END) AS simpleqa_not_attempted_rate,
            max(CASE WHEN r.metric_name = 'simpleqa_accuracy_given_attempted' THEN r.metric_value END) AS simpleqa_accuracy_given_attempted,
            max(CASE WHEN r.metric_name = 'harmbench_attack_success_rate' THEN r.metric_value END) AS harmbench_attack_success_rate,
            max(CASE WHEN r.metric_name = 'harmbench_refusal_rate' THEN r.metric_value END) AS harmbench_refusal_rate,
            sum(CASE WHEN r.metric_name IN ('global_mmlu_lite_runtime_seconds', 'ifbench_runtime_seconds', 'bfcl_v4_runtime_seconds', 'ocrbench_v2_runtime_seconds', 'mmmu_runtime_seconds', 'mbpp_runtime_seconds', 'rgb_runtime_seconds', 'simpleqa_runtime_seconds', 'harmbench_runtime_seconds') THEN r.metric_value ELSE 0 END) AS benchmark_runtime_seconds,
            sum(CASE WHEN r.metric_name = 'benchmark_samples' THEN r.metric_value ELSE 0 END) AS benchmark_samples,
            sum(CASE WHEN r.metric_name = 'benchmark_correct_count' THEN r.metric_value ELSE 0 END) AS benchmark_correct_count,
            sum(CASE WHEN r.metric_name = 'benchmark_prompt_tokens' THEN r.metric_value ELSE 0 END) AS benchmark_prompt_tokens,
            sum(CASE WHEN r.metric_name = 'benchmark_completion_tokens' THEN r.metric_value ELSE 0 END) AS benchmark_completion_tokens,
            sum(CASE WHEN r.metric_name = 'benchmark_total_tokens' THEN r.metric_value ELSE 0 END) AS benchmark_total_tokens,
            sum(CASE WHEN r.metric_name = 'benchmark_reasoning_tokens' THEN r.metric_value ELSE 0 END) AS benchmark_reasoning_tokens,
            sum(CASE WHEN r.metric_name = 'benchmark_completion_tokens' THEN r.metric_value ELSE 0 END) / nullif(sum(CASE WHEN r.metric_name IN ('global_mmlu_lite_runtime_seconds', 'ifbench_runtime_seconds', 'bfcl_v4_runtime_seconds', 'ocrbench_v2_runtime_seconds', 'mmmu_runtime_seconds', 'mbpp_runtime_seconds', 'rgb_runtime_seconds', 'simpleqa_runtime_seconds', 'harmbench_runtime_seconds') THEN r.metric_value ELSE 0 END), 0) AS benchmark_output_tokens_per_second,
            sum(CASE WHEN r.metric_name = 'benchmark_total_tokens' THEN r.metric_value ELSE 0 END) / nullif(sum(CASE WHEN r.metric_name IN ('global_mmlu_lite_runtime_seconds', 'ifbench_runtime_seconds', 'bfcl_v4_runtime_seconds', 'ocrbench_v2_runtime_seconds', 'mmmu_runtime_seconds', 'mbpp_runtime_seconds', 'rgb_runtime_seconds', 'simpleqa_runtime_seconds', 'harmbench_runtime_seconds') THEN r.metric_value ELSE 0 END), 0) AS benchmark_total_tokens_per_second,
            sum(CASE WHEN r.metric_name IN ('global_mmlu_lite_runtime_seconds', 'ifbench_runtime_seconds', 'bfcl_v4_runtime_seconds', 'ocrbench_v2_runtime_seconds', 'mmmu_runtime_seconds', 'mbpp_runtime_seconds', 'rgb_runtime_seconds', 'simpleqa_runtime_seconds', 'harmbench_runtime_seconds') THEN r.metric_value ELSE 0 END) / nullif(sum(CASE WHEN r.metric_name = 'benchmark_samples' THEN r.metric_value ELSE 0 END), 0) AS benchmark_avg_latency_seconds,
            max(CASE WHEN r.metric_name = 'benchmark_p50_latency_seconds' THEN r.metric_value END) AS benchmark_p50_latency_seconds,
            max(CASE WHEN r.metric_name = 'benchmark_p95_latency_seconds' THEN r.metric_value END) AS benchmark_p95_latency_seconds,
            sum(CASE WHEN r.metric_name = 'benchmark_truncated_count' THEN r.metric_value ELSE 0 END) AS benchmark_truncated_count,
            sum(CASE WHEN r.metric_name = 'benchmark_truncated_count' THEN r.metric_value ELSE 0 END) / nullif(sum(CASE WHEN r.metric_name = 'benchmark_samples' THEN r.metric_value ELSE 0 END), 0) AS benchmark_truncated_rate,
            sum(CASE WHEN r.metric_name = 'benchmark_total_tokens' THEN r.metric_value ELSE 0 END) / nullif(sum(CASE WHEN r.metric_name = 'benchmark_correct_count' THEN r.metric_value ELSE 0 END), 0) AS benchmark_tokens_per_correct_answer,
            sum(CASE WHEN r.metric_name IN ('global_mmlu_lite_runtime_seconds', 'ifbench_runtime_seconds', 'bfcl_v4_runtime_seconds', 'ocrbench_v2_runtime_seconds', 'mmmu_runtime_seconds', 'mbpp_runtime_seconds', 'rgb_runtime_seconds', 'simpleqa_runtime_seconds', 'harmbench_runtime_seconds') THEN r.metric_value ELSE 0 END) / nullif(sum(CASE WHEN r.metric_name = 'benchmark_correct_count' THEN r.metric_value ELSE 0 END), 0) AS benchmark_seconds_per_correct_answer,
            max(CASE WHEN r.metric_name = 'benchmark_time_to_first_token_seconds' THEN r.metric_value END) AS benchmark_time_to_first_token_seconds,
            max(CASE WHEN r.metric_name = 'benchmark_inter_token_latency_seconds' THEN r.metric_value END) AS benchmark_inter_token_latency_seconds,
            max(CASE WHEN r.metric_name = 'benchmark_end_to_end_latency_seconds' THEN r.metric_value END) AS benchmark_end_to_end_latency_seconds,
            max(CASE WHEN r.metric_name = 'benchmark_system_output_throughput_tokens_per_second' THEN r.metric_value END) AS benchmark_system_output_throughput_tokens_per_second,
            sum(CASE WHEN r.metric_name = 'benchmark_input_cost_usd' THEN r.metric_value ELSE 0 END) AS benchmark_input_cost_usd,
            sum(CASE WHEN r.metric_name = 'benchmark_output_cost_usd' THEN r.metric_value ELSE 0 END) AS benchmark_output_cost_usd,
            sum(CASE WHEN r.metric_name = 'benchmark_total_cost_usd' THEN r.metric_value ELSE 0 END) AS benchmark_total_cost_usd,
            sum(CASE WHEN r.metric_name = 'benchmark_cost_per_correct_answer_usd' THEN r.metric_value ELSE 0 END) AS benchmark_cost_per_correct_answer_usd
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
        intelligence = _model_intelligence_values(payload)
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
                ocrbench_v2_score=_to_float(payload["ocrbench_v2_score"]),
                ocrbench_v2_micro_score=_to_float(payload["ocrbench_v2_micro_score"]),
                ocrbench_v2_en_score=_to_float(payload["ocrbench_v2_en_score"]),
                ocrbench_v2_cn_score=_to_float(payload["ocrbench_v2_cn_score"]),
                mmmu_accuracy=_to_float(payload["mmmu_accuracy"]),
                mmmu_invalid_rate=_to_float(payload["mmmu_invalid_rate"]),
                mmmu_multiple_choice_accuracy=_to_float(
                    payload["mmmu_multiple_choice_accuracy"]
                ),
                mmmu_open_accuracy=_to_float(payload["mmmu_open_accuracy"]),
                mbpp_pass_at_1=_to_float(payload["mbpp_pass_at_1"]),
                mbpp_invalid_rate=_to_float(payload["mbpp_invalid_rate"]),
                mbpp_compile_rate=_to_float(payload["mbpp_compile_rate"]),
                mbpp_runtime_error_rate=_to_float(payload["mbpp_runtime_error_rate"]),
                rgb_all_rate=_to_float(payload["rgb_all_rate"]),
                rgb_rejection_rate=_to_float(payload["rgb_rejection_rate"]),
                rgb_fact_check_rate=_to_float(payload["rgb_fact_check_rate"]),
                rgb_error_correction_rate=_to_float(payload["rgb_error_correction_rate"]),
                simpleqa_f1=_to_float(payload["simpleqa_f1"]),
                simpleqa_correct_rate=_to_float(payload["simpleqa_correct_rate"]),
                simpleqa_incorrect_rate=_to_float(payload["simpleqa_incorrect_rate"]),
                simpleqa_hallucination_rate=_to_float(payload["simpleqa_hallucination_rate"]),
                simpleqa_not_attempted_rate=_to_float(payload["simpleqa_not_attempted_rate"]),
                simpleqa_accuracy_given_attempted=_to_float(
                    payload["simpleqa_accuracy_given_attempted"]
                ),
                harmbench_attack_success_rate=_to_float(
                    payload["harmbench_attack_success_rate"]
                ),
                harmbench_refusal_rate=_to_float(payload["harmbench_refusal_rate"]),
                model_intelligence_score=intelligence["score"],
                model_intelligence_coverage=intelligence["coverage"],
                model_intelligence_available_score=intelligence["available_score"],
                benchmark_runtime_seconds=_to_float(payload["benchmark_runtime_seconds"]),
                benchmark_samples=_to_float(payload["benchmark_samples"]),
                benchmark_correct_count=_to_float(payload["benchmark_correct_count"]),
                benchmark_prompt_tokens=_none_if_zero(payload["benchmark_prompt_tokens"]),
                benchmark_completion_tokens=_none_if_zero(
                    payload["benchmark_completion_tokens"]
                ),
                benchmark_total_tokens=_none_if_zero(payload["benchmark_total_tokens"]),
                benchmark_reasoning_tokens=_none_if_zero(payload["benchmark_reasoning_tokens"]),
                benchmark_output_tokens_per_second=_to_float(
                    payload["benchmark_output_tokens_per_second"]
                ),
                benchmark_total_tokens_per_second=_to_float(
                    payload["benchmark_total_tokens_per_second"]
                ),
                benchmark_avg_latency_seconds=_to_float(
                    payload["benchmark_avg_latency_seconds"]
                ),
                benchmark_p50_latency_seconds=_to_float(
                    payload["benchmark_p50_latency_seconds"]
                ),
                benchmark_p95_latency_seconds=_to_float(
                    payload["benchmark_p95_latency_seconds"]
                ),
                benchmark_truncated_count=_to_float(payload["benchmark_truncated_count"]),
                benchmark_truncated_rate=_to_float(payload["benchmark_truncated_rate"]),
                benchmark_tokens_per_correct_answer=_to_float(
                    payload["benchmark_tokens_per_correct_answer"]
                ),
                benchmark_seconds_per_correct_answer=_to_float(
                    payload["benchmark_seconds_per_correct_answer"]
                ),
                benchmark_time_to_first_token_seconds=_to_float(
                    payload["benchmark_time_to_first_token_seconds"]
                ),
                benchmark_inter_token_latency_seconds=_to_float(
                    payload["benchmark_inter_token_latency_seconds"]
                ),
                benchmark_end_to_end_latency_seconds=_to_float(
                    payload["benchmark_end_to_end_latency_seconds"]
                ),
                benchmark_system_output_throughput_tokens_per_second=_to_float(
                    payload["benchmark_system_output_throughput_tokens_per_second"]
                ),
                benchmark_input_cost_usd=_none_if_zero(payload["benchmark_input_cost_usd"]),
                benchmark_output_cost_usd=_none_if_zero(payload["benchmark_output_cost_usd"]),
                benchmark_total_cost_usd=_none_if_zero(payload["benchmark_total_cost_usd"]),
                benchmark_cost_per_correct_answer_usd=_none_if_zero(
                    payload["benchmark_cost_per_correct_answer_usd"]
                ),
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


def _none_if_zero(value: Any) -> float | None:
    numeric = _to_float(value)
    if numeric == 0:
        return None
    return numeric


def _model_intelligence_values(payload: dict[str, Any]) -> dict[str, float | None]:
    weighted_sum = 0.0
    covered_weight = 0.0
    for metric_name, weight in MODEL_INTELLIGENCE_WEIGHTS.items():
        value = _to_float(payload.get(metric_name))
        if value is None:
            continue
        weighted_sum += max(0.0, min(1.0, value)) * weight
        covered_weight += weight
    if covered_weight <= 0:
        return {"score": None, "coverage": None, "available_score": None}
    return {
        "score": weighted_sum,
        "coverage": covered_weight,
        "available_score": weighted_sum / covered_weight,
    }
