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
        rows = [
            _public_row(row)
            for row in db.leaderboard_rows()
            if row.get("global_mmlu_lite_pass_at_1") is not None
            or row.get("ifbench_prompt_level_loose") is not None
            or row.get("bfcl_v4_selected_accuracy") is not None
            or row.get("ocrbench_v2_score") is not None
            or row.get("mmmu_accuracy") is not None
            or row.get("mbpp_pass_at_1") is not None
            or row.get("rgb_all_rate") is not None
            or row.get("simpleqa_f1") is not None
            or row.get("harmbench_refusal_rate") is not None
        ]
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
        "metadata_json",
    ]
    return {key: row.get(key) for key in keys}
