from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from local_ai_analysis.db.schema import SCHEMA_SQL

try:
    import duckdb
except Exception as exc:  # pragma: no cover - import error message path
    duckdb = None
    DUCKDB_IMPORT_ERROR = exc
else:
    DUCKDB_IMPORT_ERROR = None


def _json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


def _new_id() -> str:
    return str(uuid.uuid4())


class LocalAIAnalysisDB:
    def __init__(self, db_path: str | Path):
        if duckdb is None:  # pragma: no cover
            raise RuntimeError(
                "duckdb is required. Install with `pip install -e .` or `pip install duckdb`."
            ) from DUCKDB_IMPORT_ERROR
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))

    def close(self) -> None:
        self.conn.close()

    def init_schema(self) -> None:
        self.conn.execute(SCHEMA_SQL)
        self._ensure_normalized_result_columns()

    def _ensure_normalized_result_columns(self) -> None:
        columns = {
            "global_mmlu_lite_pass_at_1": "DOUBLE",
            "global_mmlu_lite_micro_pass_at_1": "DOUBLE",
            "global_mmlu_lite_invalid_rate": "DOUBLE",
            "ifbench_prompt_level_loose": "DOUBLE",
            "ifbench_instruction_level_loose": "DOUBLE",
            "ifbench_prompt_level_strict": "DOUBLE",
            "ifbench_instruction_level_strict": "DOUBLE",
            "bfcl_v4_selected_accuracy": "DOUBLE",
            "bfcl_v4_invalid_rate": "DOUBLE",
            "bfcl_v4_non_live_accuracy": "DOUBLE",
            "bfcl_v4_live_accuracy": "DOUBLE",
            "bfcl_v4_multi_turn_accuracy": "DOUBLE",
            "bfcl_v4_agentic_accuracy": "DOUBLE",
            "ocrbench_v2_score": "DOUBLE",
            "ocrbench_v2_micro_score": "DOUBLE",
            "ocrbench_v2_en_score": "DOUBLE",
            "ocrbench_v2_cn_score": "DOUBLE",
            "mmmu_accuracy": "DOUBLE",
            "mmmu_invalid_rate": "DOUBLE",
            "mmmu_multiple_choice_accuracy": "DOUBLE",
            "mmmu_open_accuracy": "DOUBLE",
            "mbpp_pass_at_1": "DOUBLE",
            "mbpp_invalid_rate": "DOUBLE",
            "mbpp_compile_rate": "DOUBLE",
            "mbpp_runtime_error_rate": "DOUBLE",
            "rgb_all_rate": "DOUBLE",
            "rgb_rejection_rate": "DOUBLE",
            "rgb_fact_check_rate": "DOUBLE",
            "rgb_error_correction_rate": "DOUBLE",
            "simpleqa_f1": "DOUBLE",
            "simpleqa_correct_rate": "DOUBLE",
            "simpleqa_incorrect_rate": "DOUBLE",
            "simpleqa_hallucination_rate": "DOUBLE",
            "simpleqa_not_attempted_rate": "DOUBLE",
            "simpleqa_accuracy_given_attempted": "DOUBLE",
            "harmbench_attack_success_rate": "DOUBLE",
            "harmbench_refusal_rate": "DOUBLE",
            "model_intelligence_score": "DOUBLE",
            "model_intelligence_coverage": "DOUBLE",
            "model_intelligence_available_score": "DOUBLE",
        }
        for column_name, column_type in columns.items():
            existing = self.conn.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'normalized_result' AND column_name = ?
                """,
                [column_name],
            ).fetchone()
            if existing is None:
                self.conn.execute(
                    f"ALTER TABLE normalized_result ADD COLUMN {column_name} {column_type}"
                )

    def get_or_create_base_model(self, payload: dict[str, Any]) -> str:
        existing = self.conn.execute(
            "SELECT id FROM base_model WHERE name = ?", [payload["name"]]
        ).fetchone()
        if existing:
            base_id = existing[0]
            self.conn.execute(
                """
                UPDATE base_model
                SET family = ?, parameter_size_b = ?, architecture = ?, license = ?, source_url = ?
                WHERE id = ?
                """,
                [
                    payload["family"],
                    payload.get("parameter_size_b"),
                    payload.get("architecture"),
                    payload.get("license"),
                    payload.get("source_url"),
                    base_id,
                ],
            )
            return base_id

        base_id = _new_id()
        self.conn.execute(
            """
            INSERT INTO base_model
            (id, family, name, parameter_size_b, architecture, license, source_url)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                base_id,
                payload["family"],
                payload["name"],
                payload.get("parameter_size_b"),
                payload.get("architecture"),
                payload.get("license"),
                payload.get("source_url"),
            ],
        )
        return base_id

    def get_or_create_quantization(self, quantization_type: str) -> str:
        normalized = quantization_type.upper()
        existing = self.conn.execute(
            "SELECT id FROM quantization WHERE quantization_type = ?", [normalized]
        ).fetchone()
        if existing:
            return existing[0]
        quant_id = _new_id()
        self.conn.execute(
            """
            INSERT INTO quantization (id, quantization_type, bits, scheme)
            VALUES (?, ?, ?, ?)
            """,
            [quant_id, normalized, _infer_quant_bits(normalized), _infer_quant_scheme(normalized)],
        )
        return quant_id

    def upsert_variant(self, payload: dict[str, Any]) -> str:
        existing = self.conn.execute(
            """
            SELECT id FROM model_variant
            WHERE base_model_id = ? AND variant_name = ?
            """,
            [payload["base_model_id"], payload["variant_name"]],
        ).fetchone()
        metadata = _json(payload.get("metadata", {}))
        if existing:
            variant_id = existing[0]
            # Normalized rows are derived data and are recomputed after imports/runs,
            # so clear stale rows before updating variant metadata.
            self.conn.execute(
                """
                DELETE FROM normalized_result
                WHERE variant_id = ? OR baseline_variant_id = ?
                """,
                [variant_id, variant_id],
            )
            self.conn.execute(
                """
                UPDATE model_variant
                SET model_repo = ?, local_path = ?, file_name = ?, quantization_id = ?,
                    precision = ?, parameter_size_b = ?, checksum_sha256 = ?,
                    file_size_bytes = ?, is_baseline = ?, metadata_json = ?
                WHERE id = ?
                """,
                [
                    payload.get("model_repo"),
                    payload.get("local_path"),
                    payload.get("file_name"),
                    payload["quantization_id"],
                    payload.get("precision"),
                    payload.get("parameter_size_b"),
                    payload.get("checksum_sha256"),
                    payload.get("file_size_bytes"),
                    payload.get("is_baseline", False),
                    metadata,
                    variant_id,
                ],
            )
            return variant_id

        variant_id = _new_id()
        self.conn.execute(
            """
            INSERT INTO model_variant
            (id, base_model_id, variant_name, model_repo, local_path, file_name,
             quantization_id, precision, parameter_size_b, checksum_sha256,
             file_size_bytes, is_baseline, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                variant_id,
                payload["base_model_id"],
                payload["variant_name"],
                payload.get("model_repo"),
                payload.get("local_path"),
                payload.get("file_name"),
                payload["quantization_id"],
                payload.get("precision"),
                payload.get("parameter_size_b"),
                payload.get("checksum_sha256"),
                payload.get("file_size_bytes"),
                payload.get("is_baseline", False),
                metadata,
            ],
        )
        return variant_id

    def get_or_create_backend(self, payload: dict[str, Any]) -> str:
        signature = [
            payload.get("backend_name"),
            payload.get("backend_type"),
            payload.get("backend_version"),
            payload.get("backend_commit"),
            payload.get("command"),
            _json(payload.get("extra", {})),
        ]
        existing = self.conn.execute(
            """
            SELECT id FROM backend_profile
            WHERE backend_name IS NOT DISTINCT FROM ?
              AND backend_type IS NOT DISTINCT FROM ?
              AND backend_version IS NOT DISTINCT FROM ?
              AND backend_commit IS NOT DISTINCT FROM ?
              AND command IS NOT DISTINCT FROM ?
              AND extra_json IS NOT DISTINCT FROM ?
            """,
            signature,
        ).fetchone()
        if existing:
            return existing[0]

        backend_id = _new_id()
        self.conn.execute(
            """
            INSERT INTO backend_profile
            (id, backend_name, backend_type, backend_version, backend_commit, command, extra_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [backend_id, *signature],
        )
        return backend_id

    def get_or_create_hardware(self, payload: dict[str, Any]) -> str:
        existing = self.conn.execute(
            "SELECT id FROM hardware_profile WHERE hardware_hash = ?",
            [payload["hardware_hash"]],
        ).fetchone()
        if existing:
            return existing[0]

        hardware_id = _new_id()
        self.conn.execute(
            """
            INSERT INTO hardware_profile
            (id, hardware_hash, os_name, os_version, python_version, cpu_model, cpu_count,
             ram_total_bytes, gpu_name, gpu_memory_bytes, accelerator, extra_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                hardware_id,
                payload["hardware_hash"],
                payload.get("os_name"),
                payload.get("os_version"),
                payload.get("python_version"),
                payload.get("cpu_model"),
                payload.get("cpu_count"),
                payload.get("ram_total_bytes"),
                payload.get("gpu_name"),
                payload.get("gpu_memory_bytes"),
                payload.get("accelerator"),
                _json(payload.get("extra", {})),
            ],
        )
        return hardware_id

    def get_or_create_task(self, payload: dict[str, Any]) -> str:
        decoding_json = _json(payload.get("decoding", {}))
        params = [
            payload["name"],
            payload["task_type"],
            payload.get("task_version"),
            payload.get("few_shot"),
            payload["metric_name"],
        ]
        existing = self.conn.execute(
            """
            SELECT id FROM benchmark_task
            WHERE name = ?
              AND task_type = ?
              AND task_version IS NOT DISTINCT FROM ?
              AND few_shot IS NOT DISTINCT FROM ?
              AND metric_name = ?
            """,
            params,
        ).fetchone()
        if existing:
            return existing[0]

        task_id = _new_id()
        self.conn.execute(
            """
            INSERT INTO benchmark_task
            (id, name, task_type, task_version, few_shot, prompt_template, decoding_json, metric_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                task_id,
                payload["name"],
                payload["task_type"],
                payload.get("task_version"),
                payload.get("few_shot"),
                payload.get("prompt_template"),
                decoding_json,
                payload["metric_name"],
            ],
        )
        return task_id

    def create_run(
        self,
        *,
        config_path: str,
        seed: int,
        hardware_profile_id: str,
        backend_profile_id: str,
        command_args: dict[str, Any],
        metadata: dict[str, Any],
    ) -> str:
        run_id = _new_id()
        self.conn.execute(
            """
            INSERT INTO benchmark_run
            (id, run_uuid, started_at, config_path, seed, command_args_json,
             hardware_profile_id, backend_profile_id, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                str(uuid.uuid4()),
                datetime.now(timezone.utc),
                config_path,
                seed,
                _json(command_args),
                hardware_profile_id,
                backend_profile_id,
                _json(metadata),
            ],
        )
        return run_id

    def complete_run(self, run_id: str, metadata: dict[str, Any] | None = None) -> None:
        current = self.conn.execute(
            "SELECT metadata_json FROM benchmark_run WHERE id = ?", [run_id]
        ).fetchone()
        merged: dict[str, Any] = {}
        if current and current[0]:
            merged.update(json.loads(current[0]))
        if metadata:
            merged.update(metadata)
        self.conn.execute(
            """
            UPDATE benchmark_run
            SET completed_at = ?, metadata_json = ?
            WHERE id = ?
            """,
            [datetime.now(timezone.utc), _json(merged), run_id],
        )

    def insert_result(
        self,
        *,
        run_id: str,
        variant_id: str,
        task_id: str,
        metric_name: str,
        metric_value: float | None,
        unit: str | None,
        raw: dict[str, Any],
    ) -> str:
        result_id = _new_id()
        self.conn.execute(
            """
            INSERT INTO benchmark_result
            (id, run_id, variant_id, task_id, metric_name, metric_value, unit, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [result_id, run_id, variant_id, task_id, metric_name, metric_value, unit, _json(raw)],
        )
        return result_id

    def leaderboard_rows(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT
                nr.id AS normalized_result_id,
                mv.id AS variant_id,
                bm.id AS base_model_id,
                bm.family,
                bm.name AS base_model_name,
                bm.parameter_size_b,
                mv.variant_name,
                q.quantization_type AS quantization,
                mv.precision,
                mv.model_repo,
                mv.file_name,
                mv.checksum_sha256,
                mv.file_size_bytes,
                mv.is_baseline,
                bp.backend_name,
                bp.backend_version,
                bp.backend_commit,
                hp.hardware_hash,
                hp.accelerator AS hardware_accelerator,
                hp.cpu_model,
                hp.gpu_name,
                br.run_uuid,
                br.started_at,
                nr.global_mmlu_lite_pass_at_1,
                nr.global_mmlu_lite_micro_pass_at_1,
                nr.global_mmlu_lite_invalid_rate,
                nr.ifbench_prompt_level_loose,
                nr.ifbench_instruction_level_loose,
                nr.ifbench_prompt_level_strict,
                nr.ifbench_instruction_level_strict,
                nr.bfcl_v4_selected_accuracy,
                nr.bfcl_v4_invalid_rate,
                nr.bfcl_v4_non_live_accuracy,
                nr.bfcl_v4_live_accuracy,
                nr.bfcl_v4_multi_turn_accuracy,
                nr.bfcl_v4_agentic_accuracy,
                nr.ocrbench_v2_score,
                nr.ocrbench_v2_micro_score,
                nr.ocrbench_v2_en_score,
                nr.ocrbench_v2_cn_score,
                nr.mmmu_accuracy,
                nr.mmmu_invalid_rate,
                nr.mmmu_multiple_choice_accuracy,
                nr.mmmu_open_accuracy,
                nr.mbpp_pass_at_1,
                nr.mbpp_invalid_rate,
                nr.mbpp_compile_rate,
                nr.mbpp_runtime_error_rate,
                nr.rgb_all_rate,
                nr.rgb_rejection_rate,
                nr.rgb_fact_check_rate,
                nr.rgb_error_correction_rate,
                nr.simpleqa_f1,
                nr.simpleqa_correct_rate,
                nr.simpleqa_incorrect_rate,
                nr.simpleqa_hallucination_rate,
                nr.simpleqa_not_attempted_rate,
                nr.simpleqa_accuracy_given_attempted,
                nr.harmbench_attack_success_rate,
                nr.harmbench_refusal_rate,
                nr.model_intelligence_score,
                nr.model_intelligence_coverage,
                nr.model_intelligence_available_score,
                nr.benchmark_runtime_seconds,
                nr.metadata_json
            FROM normalized_result nr
            JOIN model_variant mv ON nr.variant_id = mv.id
            JOIN base_model bm ON nr.base_model_id = bm.id
            JOIN quantization q ON mv.quantization_id = q.id
            LEFT JOIN benchmark_run br ON nr.run_id = br.id
            LEFT JOIN backend_profile bp ON br.backend_profile_id = bp.id
            LEFT JOIN hardware_profile hp ON br.hardware_profile_id = hp.id
            ORDER BY bm.family, bm.parameter_size_b, bm.name, mv.is_baseline DESC, q.bits DESC NULLS LAST
            """
        ).fetchall()
        columns = [column[0] for column in self.conn.description]
        return [dict(zip(columns, row, strict=True)) for row in rows]

    def raw_variant_metadata(self, variant_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT
                mv.id AS variant_id,
                mv.variant_name,
                mv.model_repo,
                mv.local_path,
                mv.file_name,
                mv.checksum_sha256,
                mv.file_size_bytes,
                mv.metadata_json,
                bm.name AS base_model_name,
                bm.family,
                q.quantization_type,
                list({
                    'metric_name': r.metric_name,
                    'metric_value': r.metric_value,
                    'unit': r.unit,
                    'raw_json': r.raw_json
                }) AS results
            FROM model_variant mv
            JOIN base_model bm ON mv.base_model_id = bm.id
            JOIN quantization q ON mv.quantization_id = q.id
            LEFT JOIN benchmark_result r ON r.variant_id = mv.id
            WHERE mv.id = ?
            GROUP BY
                mv.id, mv.variant_name, mv.model_repo, mv.local_path, mv.file_name,
                mv.checksum_sha256, mv.file_size_bytes, mv.metadata_json,
                bm.name, bm.family, q.quantization_type
            """,
            [variant_id],
        ).fetchone()
        if not row:
            return None
        columns = [column[0] for column in self.conn.description]
        payload = dict(zip(columns, row, strict=True))
        if payload.get("metadata_json"):
            payload["metadata"] = json.loads(payload.pop("metadata_json"))
        return payload


def _infer_quant_bits(quantization_type: str) -> float | None:
    if quantization_type in {"BF16", "FP16"}:
        return 16
    if quantization_type in {"FP32"}:
        return 32
    if quantization_type.startswith("Q"):
        digits = ""
        for char in quantization_type[1:]:
            if char.isdigit():
                digits += char
            else:
                break
        if digits:
            return float(digits)
    return None


def _infer_quant_scheme(quantization_type: str) -> str | None:
    if quantization_type in {"BF16", "FP16", "FP32"}:
        return "floating_point"
    if "_K" in quantization_type:
        return "k_quant"
    if quantization_type.startswith("Q"):
        return "quantized"
    return None
