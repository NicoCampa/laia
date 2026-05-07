from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

from local_ai_analysis.backend import backend_profile_payload
from local_ai_analysis.config import BaseModelConfig, BenchmarkConfig, VariantConfig, load_config
from local_ai_analysis.db import LocalAIAnalysisDB
from local_ai_analysis.eval.bfcl_v4 import BFCLV4Runner
from local_ai_analysis.eval.global_mmlu_lite import GlobalMMLULiteRunner
from local_ai_analysis.eval.harmbench import HarmBenchRunner
from local_ai_analysis.eval.ifbench import IFBenchRunner
from local_ai_analysis.eval.mbpp import MBPPRunner
from local_ai_analysis.eval.mmmu import MMMURunner
from local_ai_analysis.eval.ocrbench_v2 import OCRBenchV2Runner
from local_ai_analysis.eval.rgb import RGBRunner
from local_ai_analysis.eval.simpleqa import SimpleQARunner
from local_ai_analysis.metrics import MetricResult
from local_ai_analysis.normalization import refresh_normalized_results
from local_ai_analysis.utils.jsonl import JsonlWriter
from local_ai_analysis.utils.metadata import collect_hardware_metadata


ProgressCallback = Callable[[str, dict[str, Any]], None]


class BenchmarkRunner:
    def __init__(
        self,
        config_path: str | Path,
        dry_run: bool = False,
        progress_callback: ProgressCallback | None = None,
    ):
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path)
        self.dry_run = dry_run
        self.progress_callback = progress_callback
        self.db = LocalAIAnalysisDB(self.config.run.output_db)
        self.db.init_schema()
        self.events = JsonlWriter(self.config.run.raw_jsonl)

    def run(self) -> dict[str, Any]:
        if self.dry_run:
            return self._dry_run_only()
        if (
            not self.config.global_mmlu_lite.enabled
            and not self.config.ifbench.enabled
            and not self.config.bfcl_v4.enabled
            and not self.config.ocrbench_v2.enabled
            and not self.config.mmmu.enabled
            and not self.config.mbpp.enabled
            and not self.config.rgb.enabled
            and not self.config.simpleqa.enabled
            and not self.config.harmbench.enabled
        ):
            raise ValueError(
                "Enable at least one benchmark: global_mmlu_lite, ifbench, bfcl_v4, "
                "ocrbench_v2, mmmu, mbpp, rgb, simpleqa, or harmbench."
            )

        cfg = self.config
        variants_total = self._variant_count()
        hardware_id = self.db.get_or_create_hardware(collect_hardware_metadata())
        backend_id = self.db.get_or_create_backend(backend_profile_payload(cfg.backend))
        run_id = self.db.create_run(
            config_path=str(self.config_path),
            seed=cfg.run.seed,
            hardware_profile_id=hardware_id,
            backend_profile_id=backend_id,
            command_args={"argv": sys.argv, "dry_run": self.dry_run},
            metadata={
                "project": cfg.project,
                "run_name": cfg.run.name,
                "notes": cfg.run.notes,
                "global_mmlu_lite": cfg.global_mmlu_lite.model_dump(exclude={"api_key"}),
                "ifbench": cfg.ifbench.model_dump(exclude={"api_key"}),
                "bfcl_v4": cfg.bfcl_v4.model_dump(exclude={"api_key"}),
                "ocrbench_v2": cfg.ocrbench_v2.model_dump(exclude={"api_key"}),
                "mmmu": cfg.mmmu.model_dump(exclude={"api_key"}),
                "mbpp": cfg.mbpp.model_dump(exclude={"api_key"}),
                "rgb": cfg.rgb.model_dump(exclude={"api_key"}),
                "simpleqa": cfg.simpleqa.model_dump(
                    exclude={"api_key", "grader_api_key"}
                ),
                "harmbench": cfg.harmbench.model_dump(
                    exclude={"api_key", "judge_api_key"}
                ),
            },
        )
        self.events.write(
            "run_started",
            {
                "run_id": run_id,
                "config_path": str(self.config_path),
                "dry_run": self.dry_run,
                "seed": cfg.run.seed,
                "variants_total": variants_total,
            },
        )
        self._progress(
            "run_started",
            {"run_id": run_id, "variants_total": variants_total, "dry_run": self.dry_run},
        )

        variants_seen = 0
        metrics_written = 0
        try:
            for base_model in cfg.models:
                base_model_id = self._upsert_base_model(base_model)
                for variant in base_model.variants:
                    variants_seen += 1
                    self._progress(
                        "variant_started",
                        {
                            "index": variants_seen,
                            "total": variants_total,
                            "base_model": base_model.name,
                            "variant": variant.name,
                        },
                    )
                    variant_id = self._upsert_variant(base_model_id, base_model, variant)
                    if not self._variant_has_api_model(variant):
                        self._record_skipped_variant(
                            run_id=run_id,
                            variant_id=variant_id,
                            base_model=base_model,
                            variant=variant,
                            reason="missing_api_model",
                        )
                        self._progress(
                            "variant_skipped",
                            {
                                "index": variants_seen,
                                "total": variants_total,
                                "variant": variant.name,
                                "reason": "missing_api_model",
                            },
                        )
                        continue

                    metrics = self._run_variant_metrics(variant)
                    for metric in metrics:
                        task_id = self._task_for_metric(metric)
                        self.db.insert_result(
                            run_id=run_id,
                            variant_id=variant_id,
                            task_id=task_id,
                            metric_name=metric.metric_name,
                            metric_value=metric.metric_value,
                            unit=metric.unit,
                            raw=metric.raw,
                        )
                        self.events.write(
                            "metric_recorded",
                            {
                                "run_id": run_id,
                                "variant_id": variant_id,
                                "metric_name": metric.metric_name,
                                "metric_value": metric.metric_value,
                                "unit": metric.unit,
                            },
                        )
                        metrics_written += 1
                    self._progress(
                        "variant_completed",
                        {
                            "index": variants_seen,
                            "total": variants_total,
                            "variant": variant.name,
                            "metrics_written": len(metrics),
                        },
                    )
            self._progress("normalization_started", {})
            normalized_rows = refresh_normalized_results(self.db)
            self._progress("normalization_completed", {"normalized_rows": normalized_rows})
            self.db.complete_run(
                run_id,
                {
                    "status": "completed",
                    "variants_seen": variants_seen,
                    "metrics_written": metrics_written,
                    "normalized_rows": normalized_rows,
                },
            )
            self.events.write(
                "run_completed",
                {
                    "run_id": run_id,
                    "variants_seen": variants_seen,
                    "metrics_written": metrics_written,
                    "normalized_rows": normalized_rows,
                },
            )
            self._progress(
                "run_completed",
                {
                    "run_id": run_id,
                    "variants_seen": variants_seen,
                    "metrics_written": metrics_written,
                    "normalized_rows": normalized_rows,
                },
            )
            return {
                "run_id": run_id,
                "variants_seen": variants_seen,
                "metrics_written": metrics_written,
                "normalized_rows": normalized_rows,
                "db_path": cfg.run.output_db,
                "raw_jsonl": cfg.run.raw_jsonl,
            }
        except Exception as exc:
            self.db.complete_run(run_id, {"status": "failed", "error": str(exc)})
            self.events.write("run_failed", {"run_id": run_id, "error": str(exc)})
            self._progress("run_failed", {"run_id": run_id, "error": str(exc)})
            raise

    def _dry_run_only(self) -> dict[str, Any]:
        variants_total = self._variant_count()
        self._progress(
            "run_started",
            {"run_id": None, "variants_total": variants_total, "dry_run": True},
        )
        variants_seen = 0
        for base_model in self.config.models:
            for variant in base_model.variants:
                variants_seen += 1
                self._progress(
                    "variant_started",
                    {
                        "index": variants_seen,
                        "total": variants_total,
                        "base_model": base_model.name,
                        "variant": variant.name,
                    },
                )
                self.events.write(
                    "dry_run_variant",
                    {
                        "base_model": base_model.name,
                        "variant": variant.name,
                        "planned_commands": self._planned_commands(variant),
                    },
                )
                self._progress(
                    "variant_skipped",
                    {
                        "index": variants_seen,
                        "total": variants_total,
                        "variant": variant.name,
                        "reason": "dry_run",
                    },
                )
        self.events.write(
            "dry_run_completed",
            {
                "config_path": str(self.config_path),
                "variants_seen": variants_seen,
                "raw_jsonl": self.config.run.raw_jsonl,
            },
        )
        self._progress(
            "run_completed",
            {
                "run_id": None,
                "variants_seen": variants_seen,
                "metrics_written": 0,
                "normalized_rows": 0,
                "dry_run": True,
            },
        )
        return {
            "variants_seen": variants_seen,
            "metrics_written": 0,
            "normalized_rows": 0,
            "db_path": self.config.run.output_db,
            "raw_jsonl": self.config.run.raw_jsonl,
            "dry_run": True,
        }

    def _upsert_base_model(self, base_model: BaseModelConfig) -> str:
        return self.db.get_or_create_base_model(
            {
                "family": base_model.family,
                "name": base_model.name,
                "parameter_size_b": base_model.parameter_size_b,
                "architecture": base_model.architecture,
                "license": base_model.license,
                "source_url": base_model.source_url,
            }
        )

    def _upsert_variant(
        self,
        base_model_id: str,
        base_model: BaseModelConfig,
        variant: VariantConfig,
    ) -> str:
        quantization_id = self.db.get_or_create_quantization(variant.quantization)
        return self.db.upsert_variant(
            {
                "base_model_id": base_model_id,
                "variant_name": variant.name,
                "model_repo": variant.model_repo,
                "local_path": None,
                "file_name": None,
                "quantization_id": quantization_id,
                "precision": variant.precision or variant.quantization,
                "parameter_size_b": base_model.parameter_size_b,
                "checksum_sha256": None,
                "file_size_bytes": None,
                "is_baseline": variant.baseline,
                "metadata": {
                    "variant_config": variant.model_dump(),
                    "model_file": {
                        "exists": False,
                        "file_name": None,
                        "path": None,
                        "sha256": None,
                        "size_bytes": None,
                    },
                },
            }
        )

    def _record_skipped_variant(
        self,
        *,
        run_id: str,
        variant_id: str,
        base_model: BaseModelConfig,
        variant: VariantConfig,
        reason: str,
    ) -> None:
        self.events.write(
            "variant_skipped",
            {
                "run_id": run_id,
                "variant_id": variant_id,
                "base_model": base_model.name,
                "variant": variant.name,
                "reason": reason,
                "planned_commands": self._planned_commands(variant),
            },
        )

    def _planned_commands(self, variant: VariantConfig) -> dict[str, str]:
        planned: dict[str, str] = {}
        if self.config.global_mmlu_lite.enabled:
            planned["global_mmlu_lite"] = GlobalMMLULiteRunner(
                self.config.global_mmlu_lite
            ).planned_command(variant)
        if self.config.ifbench.enabled:
            planned["ifbench"] = IFBenchRunner(self.config.ifbench).planned_command(variant)
        if self.config.bfcl_v4.enabled:
            planned["bfcl_v4"] = BFCLV4Runner(self.config.bfcl_v4).planned_command(variant)
        if self.config.ocrbench_v2.enabled:
            planned["ocrbench_v2"] = OCRBenchV2Runner(
                self.config.ocrbench_v2
            ).planned_command(variant)
        if self.config.mmmu.enabled:
            planned["mmmu"] = MMMURunner(self.config.mmmu).planned_command(variant)
        if self.config.mbpp.enabled:
            planned["mbpp"] = MBPPRunner(self.config.mbpp).planned_command(variant)
        if self.config.rgb.enabled:
            planned["rgb"] = RGBRunner(self.config.rgb).planned_command(variant)
        if self.config.simpleqa.enabled:
            planned["simpleqa"] = SimpleQARunner(self.config.simpleqa).planned_command(variant)
        if self.config.harmbench.enabled:
            planned["harmbench"] = HarmBenchRunner(self.config.harmbench).planned_command(
                variant
            )
        return planned

    def _run_variant_metrics(self, variant: VariantConfig) -> list[MetricResult]:
        metrics: list[MetricResult] = []
        if self.config.global_mmlu_lite.enabled:
            self._progress("task_started", {"task": "global-mmlu-lite", "variant": variant.name})
            metrics.extend(
                GlobalMMLULiteRunner(self.config.global_mmlu_lite).run(
                    variant,
                    progress_callback=self._progress,
                )
            )
            self._progress(
                "task_completed", {"task": "global-mmlu-lite", "variant": variant.name}
            )
        if self.config.ifbench.enabled:
            self._progress("task_started", {"task": "ifbench", "variant": variant.name})
            metrics.extend(
                IFBenchRunner(self.config.ifbench).run(
                    variant,
                    progress_callback=self._progress,
                )
            )
            self._progress("task_completed", {"task": "ifbench", "variant": variant.name})
        if self.config.bfcl_v4.enabled:
            self._progress("task_started", {"task": "bfcl-v4", "variant": variant.name})
            metrics.extend(
                BFCLV4Runner(self.config.bfcl_v4).run(
                    variant,
                    progress_callback=self._progress,
                )
            )
            self._progress("task_completed", {"task": "bfcl-v4", "variant": variant.name})
        if self.config.ocrbench_v2.enabled:
            self._progress("task_started", {"task": "ocrbench-v2", "variant": variant.name})
            metrics.extend(
                OCRBenchV2Runner(self.config.ocrbench_v2).run(
                    variant,
                    progress_callback=self._progress,
                )
            )
            self._progress("task_completed", {"task": "ocrbench-v2", "variant": variant.name})
        if self.config.mmmu.enabled:
            self._progress("task_started", {"task": "mmmu", "variant": variant.name})
            metrics.extend(
                MMMURunner(self.config.mmmu).run(
                    variant,
                    progress_callback=self._progress,
                )
            )
            self._progress("task_completed", {"task": "mmmu", "variant": variant.name})
        if self.config.mbpp.enabled:
            self._progress("task_started", {"task": "mbpp", "variant": variant.name})
            metrics.extend(
                MBPPRunner(self.config.mbpp).run(
                    variant,
                    progress_callback=self._progress,
                )
            )
            self._progress("task_completed", {"task": "mbpp", "variant": variant.name})
        if self.config.rgb.enabled:
            self._progress("task_started", {"task": "rgb", "variant": variant.name})
            metrics.extend(
                RGBRunner(self.config.rgb).run(
                    variant,
                    progress_callback=self._progress,
                )
            )
            self._progress("task_completed", {"task": "rgb", "variant": variant.name})
        if self.config.simpleqa.enabled:
            self._progress("task_started", {"task": "simpleqa", "variant": variant.name})
            metrics.extend(
                SimpleQARunner(self.config.simpleqa).run(
                    variant,
                    progress_callback=self._progress,
                )
            )
            self._progress("task_completed", {"task": "simpleqa", "variant": variant.name})
        if self.config.harmbench.enabled:
            self._progress("task_started", {"task": "harmbench", "variant": variant.name})
            metrics.extend(
                HarmBenchRunner(self.config.harmbench).run(
                    variant,
                    progress_callback=self._progress,
                )
            )
            self._progress("task_completed", {"task": "harmbench", "variant": variant.name})
        return metrics

    def _task_for_metric(self, metric: MetricResult) -> str:
        if metric.metric_name.startswith("harmbench_"):
            return self.db.get_or_create_task(
                {
                    "name": "harmbench",
                    "task_type": "safety_red_team",
                    "task_version": self.config.harmbench.dataset_name,
                    "few_shot": 0,
                    "prompt_template": self.config.harmbench.prompt_template,
                    "decoding": {
                        "provider": self.config.harmbench.provider,
                        "base_url": self.config.harmbench.base_url,
                        "temperature": self.config.harmbench.temperature,
                        "max_tokens": self.config.harmbench.max_tokens,
                        "top_p": self.config.harmbench.top_p,
                        "stop": self.config.harmbench.stop,
                        "seed": self.config.harmbench.seed,
                        "reasoning_effort": self.config.harmbench.reasoning_effort,
                        "response_format": self.config.harmbench.response_format,
                        "request_extra": self.config.harmbench.request_extra,
                        "dataset_url": self.config.harmbench.dataset_url,
                        "dataset_revision": self.config.harmbench.dataset_revision,
                        "functional_categories": (
                            self.config.harmbench.functional_categories
                        ),
                        "sample_limit": self.config.harmbench.sample_limit,
                        "judge": self.config.harmbench.judge,
                        "judge_model": self.config.harmbench.judge_model,
                        "judge_provider": self.config.harmbench.judge_provider,
                        "judge_base_url": self.config.harmbench.judge_base_url,
                        "judge_temperature": self.config.harmbench.judge_temperature,
                        "judge_max_tokens": self.config.harmbench.judge_max_tokens,
                        "judge_reasoning_effort": (
                            self.config.harmbench.judge_reasoning_effort
                        ),
                        "evaluator": self.config.harmbench.evaluator,
                    },
                    "metric_name": metric.metric_name,
                }
            )
        if metric.metric_name.startswith("simpleqa_"):
            return self.db.get_or_create_task(
                {
                    "name": "simpleqa",
                    "task_type": "short_form_factuality",
                    "task_version": self.config.simpleqa.dataset_name,
                    "few_shot": 0,
                    "prompt_template": self.config.simpleqa.prompt_template,
                    "decoding": {
                        "provider": self.config.simpleqa.provider,
                        "base_url": self.config.simpleqa.base_url,
                        "temperature": self.config.simpleqa.temperature,
                        "max_tokens": self.config.simpleqa.max_tokens,
                        "top_p": self.config.simpleqa.top_p,
                        "stop": self.config.simpleqa.stop,
                        "seed": self.config.simpleqa.seed,
                        "reasoning_effort": self.config.simpleqa.reasoning_effort,
                        "response_format": self.config.simpleqa.response_format,
                        "request_extra": self.config.simpleqa.request_extra,
                        "dataset_url": self.config.simpleqa.dataset_url,
                        "dataset_revision": self.config.simpleqa.dataset_revision,
                        "sample_limit": self.config.simpleqa.sample_limit,
                        "grader": self.config.simpleqa.grader,
                        "grader_model": self.config.simpleqa.grader_model,
                        "grader_provider": self.config.simpleqa.grader_provider,
                        "grader_base_url": self.config.simpleqa.grader_base_url,
                        "grader_temperature": self.config.simpleqa.grader_temperature,
                        "grader_max_tokens": self.config.simpleqa.grader_max_tokens,
                        "grader_reasoning_effort": (
                            self.config.simpleqa.grader_reasoning_effort
                        ),
                        "evaluator": self.config.simpleqa.evaluator,
                    },
                    "metric_name": metric.metric_name,
                }
            )
        if metric.metric_name.startswith("rgb_"):
            return self.db.get_or_create_task(
                {
                    "name": "rgb",
                    "task_type": "rag_generation",
                    "task_version": self.config.rgb.dataset_name,
                    "few_shot": 0,
                    "prompt_template": self.config.rgb.instruction_template_en,
                    "decoding": {
                        "provider": self.config.rgb.provider,
                        "base_url": self.config.rgb.base_url,
                        "temperature": self.config.rgb.temperature,
                        "max_tokens": self.config.rgb.max_tokens,
                        "top_p": self.config.rgb.top_p,
                        "stop": self.config.rgb.stop,
                        "seed": self.config.rgb.seed,
                        "reasoning_effort": self.config.rgb.reasoning_effort,
                        "response_format": self.config.rgb.response_format,
                        "request_extra": self.config.rgb.request_extra,
                        "dataset": self.config.rgb.dataset,
                        "dataset_revision": self.config.rgb.dataset_revision,
                        "sample_limit": self.config.rgb.sample_limit,
                        "noise_rate": self.config.rgb.noise_rate,
                        "passage_num": self.config.rgb.passage_num,
                        "correct_rate": self.config.rgb.correct_rate,
                        "evaluator": self.config.rgb.evaluator,
                    },
                    "metric_name": metric.metric_name,
                }
            )
        if metric.metric_name.startswith("mbpp_"):
            return self.db.get_or_create_task(
                {
                    "name": "mbpp",
                    "task_type": "code_generation",
                    "task_version": self.config.mbpp.dataset_name,
                    "few_shot": 0,
                    "prompt_template": self.config.mbpp.prompt_template,
                    "decoding": {
                        "provider": self.config.mbpp.provider,
                        "base_url": self.config.mbpp.base_url,
                        "temperature": self.config.mbpp.temperature,
                        "max_tokens": self.config.mbpp.max_tokens,
                        "top_p": self.config.mbpp.top_p,
                        "stop": self.config.mbpp.stop,
                        "seed": self.config.mbpp.seed,
                        "reasoning_effort": self.config.mbpp.reasoning_effort,
                        "response_format": self.config.mbpp.response_format,
                        "request_extra": self.config.mbpp.request_extra,
                        "dataset_config": self.config.mbpp.dataset_config,
                        "split": self.config.mbpp.split,
                        "dataset_revision": self.config.mbpp.dataset_revision,
                        "sample_limit": self.config.mbpp.sample_limit,
                        "include_tests_in_prompt": self.config.mbpp.include_tests_in_prompt,
                        "include_challenge_tests": self.config.mbpp.include_challenge_tests,
                        "execution_timeout_seconds": (
                            self.config.mbpp.execution_timeout_seconds
                        ),
                        "evaluator": self.config.mbpp.evaluator,
                    },
                    "metric_name": metric.metric_name,
                }
            )
        if metric.metric_name.startswith("mmmu_"):
            return self.db.get_or_create_task(
                {
                    "name": "mmmu",
                    "task_type": "vision_reasoning",
                    "task_version": self.config.mmmu.dataset_name,
                    "few_shot": 0,
                    "prompt_template": self.config.mmmu.multiple_choice_prompt_template,
                    "decoding": {
                        "provider": self.config.mmmu.provider,
                        "base_url": self.config.mmmu.base_url,
                        "temperature": self.config.mmmu.temperature,
                        "max_tokens": self.config.mmmu.max_tokens,
                        "top_p": self.config.mmmu.top_p,
                        "stop": self.config.mmmu.stop,
                        "seed": self.config.mmmu.seed,
                        "reasoning_effort": self.config.mmmu.reasoning_effort,
                        "response_format": self.config.mmmu.response_format,
                        "request_extra": self.config.mmmu.request_extra,
                        "split": self.config.mmmu.split,
                        "dataset_revision": self.config.mmmu.dataset_revision,
                        "subjects": self.config.mmmu.subjects,
                        "sample_limit": self.config.mmmu.sample_limit,
                        "image_format": self.config.mmmu.image_format,
                        "evaluator": self.config.mmmu.evaluator,
                    },
                    "metric_name": metric.metric_name,
                }
            )
        if metric.metric_name.startswith("ocrbench_v2_"):
            return self.db.get_or_create_task(
                {
                    "name": "ocrbench-v2",
                    "task_type": "vision_ocr",
                    "task_version": self.config.ocrbench_v2.dataset_name,
                    "few_shot": 0,
                    "prompt_template": self.config.ocrbench_v2.prompt_template,
                    "decoding": {
                        "provider": self.config.ocrbench_v2.provider,
                        "base_url": self.config.ocrbench_v2.base_url,
                        "temperature": self.config.ocrbench_v2.temperature,
                        "max_tokens": self.config.ocrbench_v2.max_tokens,
                        "top_p": self.config.ocrbench_v2.top_p,
                        "stop": self.config.ocrbench_v2.stop,
                        "seed": self.config.ocrbench_v2.seed,
                        "reasoning_effort": self.config.ocrbench_v2.reasoning_effort,
                        "response_format": self.config.ocrbench_v2.response_format,
                        "request_extra": self.config.ocrbench_v2.request_extra,
                        "split": self.config.ocrbench_v2.split,
                        "dataset_revision": self.config.ocrbench_v2.dataset_revision,
                        "dataset_configs": self.config.ocrbench_v2.dataset_configs,
                        "sample_limit": self.config.ocrbench_v2.sample_limit,
                        "image_format": self.config.ocrbench_v2.image_format,
                        "evaluator": self.config.ocrbench_v2.evaluator,
                    },
                    "metric_name": metric.metric_name,
                }
            )
        if metric.metric_name.startswith("bfcl_v4_"):
            return self.db.get_or_create_task(
                {
                    "name": "bfcl-v4",
                    "task_type": "function_calling",
                    "task_version": self.config.bfcl_v4.version,
                    "few_shot": 0,
                    "prompt_template": None,
                    "decoding": {
                        "provider": self.config.bfcl_v4.provider,
                        "base_url": self.config.bfcl_v4.base_url,
                        "temperature": self.config.bfcl_v4.temperature,
                        "max_tokens": self.config.bfcl_v4.max_tokens,
                        "top_p": self.config.bfcl_v4.top_p,
                        "stop": self.config.bfcl_v4.stop,
                        "seed": self.config.bfcl_v4.seed,
                        "reasoning_effort": self.config.bfcl_v4.reasoning_effort,
                        "response_format": self.config.bfcl_v4.response_format,
                        "request_extra": self.config.bfcl_v4.request_extra,
                        "categories": self.config.bfcl_v4.categories,
                        "sample_limit": self.config.bfcl_v4.sample_limit,
                        "evaluator": self.config.bfcl_v4.evaluator,
                    },
                    "metric_name": metric.metric_name,
                }
            )
        if metric.metric_name.startswith("ifbench_"):
            return self.db.get_or_create_task(
                {
                    "name": "ifbench",
                    "task_type": "instruction_following",
                    "task_version": self.config.ifbench.dataset_name,
                    "few_shot": 0,
                    "prompt_template": None,
                    "decoding": {
                        "provider": self.config.ifbench.provider,
                        "base_url": self.config.ifbench.base_url,
                        "temperature": self.config.ifbench.temperature,
                        "max_tokens": self.config.ifbench.max_tokens,
                        "top_p": self.config.ifbench.top_p,
                        "stop": self.config.ifbench.stop,
                        "seed": self.config.ifbench.seed,
                        "reasoning_effort": self.config.ifbench.reasoning_effort,
                        "response_format": self.config.ifbench.response_format,
                        "request_extra": self.config.ifbench.request_extra,
                        "split": self.config.ifbench.split,
                        "dataset_revision": self.config.ifbench.dataset_revision,
                        "evaluator": self.config.ifbench.evaluator,
                    },
                    "metric_name": metric.metric_name,
                }
            )
        return self.db.get_or_create_task(
            {
                "name": "global-mmlu-lite",
                "task_type": "generation_quality",
                "task_version": self.config.global_mmlu_lite.dataset_name,
                "few_shot": 0,
                "prompt_template": self.config.global_mmlu_lite.prompt_template,
                "decoding": {
                    "provider": self.config.global_mmlu_lite.provider,
                    "base_url": self.config.global_mmlu_lite.base_url,
                    "temperature": self.config.global_mmlu_lite.temperature,
                    "max_tokens": self.config.global_mmlu_lite.max_tokens,
                    "top_p": self.config.global_mmlu_lite.top_p,
                    "stop": self.config.global_mmlu_lite.stop,
                    "seed": self.config.global_mmlu_lite.seed,
                    "reasoning_effort": self.config.global_mmlu_lite.reasoning_effort,
                    "response_format": self.config.global_mmlu_lite.response_format,
                    "request_extra": self.config.global_mmlu_lite.request_extra,
                    "languages": self.config.global_mmlu_lite.languages,
                    "split": self.config.global_mmlu_lite.split,
                    "dataset_revision": self.config.global_mmlu_lite.dataset_revision,
                    "parser": self.config.global_mmlu_lite.parser_version,
                },
                "metric_name": metric.metric_name,
            }
        )

    def _variant_count(self) -> int:
        return sum(len(base_model.variants) for base_model in self.config.models)

    @staticmethod
    def _variant_has_api_model(variant: VariantConfig) -> bool:
        return bool(variant.api_model or variant.model_repo or variant.name)

    def _progress(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.progress_callback is not None:
            self.progress_callback(event_type, payload)


def run_benchmark(config_path: str | Path, dry_run: bool = False) -> dict[str, Any]:
    runner = BenchmarkRunner(config_path=config_path, dry_run=dry_run)
    try:
        return runner.run()
    finally:
        runner.db.close()


def run_benchmark_with_progress(
    config_path: str | Path,
    dry_run: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    runner = BenchmarkRunner(
        config_path=config_path,
        dry_run=dry_run,
        progress_callback=progress_callback,
    )
    try:
        return runner.run()
    finally:
        runner.db.close()
