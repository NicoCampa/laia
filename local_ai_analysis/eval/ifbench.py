from __future__ import annotations

import copy
import json
import re
import warnings
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Callable

from local_ai_analysis.adapters.native import create_native_client
from local_ai_analysis.config import IFBenchSettings, VariantConfig
from local_ai_analysis.eval.efficiency import efficiency_metrics_from_summary
from local_ai_analysis.eval.runtime import maybe_reset_runtime
from local_ai_analysis.metrics import MetricResult


ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass
class ScoreCounts:
    prompt_total: int = 0
    prompt_correct: int = 0
    instruction_total: int = 0
    instruction_correct: int = 0

    @property
    def prompt_accuracy(self) -> float | None:
        return self.prompt_correct / self.prompt_total if self.prompt_total else None

    @property
    def instruction_accuracy(self) -> float | None:
        return (
            self.instruction_correct / self.instruction_total
            if self.instruction_total
            else None
        )


class IFBenchRunner:
    def __init__(self, settings: IFBenchSettings):
        self.settings = settings
        self.client = create_native_client(
            provider=settings.provider,
            base_url=settings.base_url,
            api_key=settings.api_key,
            api_key_env=settings.api_key_env,
            timeout_seconds=settings.timeout_seconds,
        )

    def planned_command(self, variant: VariantConfig) -> str:
        model = self._configured_model_name(variant)
        return (
            f"POST {self.client.planned_endpoint()} "
            f"model={model} dataset={self.settings.dataset_name} split={self.settings.split}"
        )

    def run(
        self,
        variant: VariantConfig,
        progress_callback: ProgressCallback | None = None,
    ) -> list[MetricResult]:
        model = self._model_name(variant)
        out_dir = Path(self.settings.output_dir) / _safe_name(variant.name)
        out_dir.mkdir(parents=True, exist_ok=True)
        samples_path = out_dir / "samples.jsonl"
        summary_path = out_dir / "summary.json"

        if progress_callback:
            progress_callback(
                "dataset_loading_started",
                {
                    "task": "ifbench",
                    "variant": variant.name,
                    "dataset": self.settings.dataset_name,
                    "languages": ["default"],
                    "split": self.settings.split,
                    "dataset_revision": self.settings.dataset_revision,
                },
            )
        rows = self._load_dataset()
        if self.settings.sample_limit is not None:
            rows = rows[: self.settings.sample_limit]
        total_samples = len(rows)
        if progress_callback:
            progress_callback(
                "dataset_loading_completed",
                {
                    "task": "ifbench",
                    "variant": variant.name,
                    "dataset": self.settings.dataset_name,
                    "languages": ["default"],
                    "split": self.settings.split,
                    "total_samples": total_samples,
                },
            )
            progress_callback(
                "task_sample_plan",
                {
                    "task": "ifbench",
                    "variant": variant.name,
                    "total_samples": total_samples,
                    "languages": ["default"],
                },
            )

        strict_outputs: list[Any] = []
        loose_outputs: list[Any] = []
        strict_counts = ScoreCounts()
        loose_counts = ScoreCounts()
        total_runtime = 0.0

        with samples_path.open("w", encoding="utf-8") as sample_file:
            for index, row in enumerate(rows, start=1):
                prompt = str(row.get("prompt") or "")
                response = self.client.generate(
                    model=model,
                    prompt=prompt,
                    temperature=self.settings.temperature,
                    max_tokens=self.settings.max_tokens,
                    top_p=self.settings.top_p,
                    stop=self.settings.stop,
                    seed=self.settings.seed,
                    reasoning_effort=self.settings.reasoning_effort,
                    response_format=self.settings.response_format,
                    request_extra=self.settings.request_extra,
                )
                total_runtime += response.runtime_seconds
                raw_output = response.text
                parsed_output = _strip_reasoning(raw_output) if self.settings.strip_thinking else raw_output
                strict_output, loose_output = _score_response(row, parsed_output)
                strict_outputs.append(strict_output)
                loose_outputs.append(loose_output)
                _add_counts(strict_counts, strict_output)
                _add_counts(loose_counts, loose_output)

                sample_record = {
                    "dataset": self.settings.dataset_name,
                    "dataset_revision": self.settings.dataset_revision,
                    "split": self.settings.split,
                    "sample_id": row.get("key"),
                    "prompt": prompt,
                    "instruction_id_list": row.get("instruction_id_list"),
                    "kwargs": row.get("kwargs"),
                    "raw_output": raw_output,
                    "parsed_output": parsed_output,
                    "strict_follow_all": strict_output.follow_all_instructions,
                    "strict_follow_instruction_list": strict_output.follow_instruction_list,
                    "loose_follow_all": loose_output.follow_all_instructions,
                    "loose_follow_instruction_list": loose_output.follow_instruction_list,
                    "runtime_seconds": response.runtime_seconds,
                    "usage": response.raw.get("usage"),
                    "raw_response": response.raw,
                }
                sample_file.write(json.dumps(sample_record, ensure_ascii=False, default=str) + "\n")

                if progress_callback:
                    progress_callback(
                        "task_progress",
                        {
                            "task": "ifbench",
                            "variant": variant.name,
                            "language": "default",
                            "completed_samples": index,
                            "total_samples": total_samples,
                            "latest_correct": loose_output.follow_all_instructions,
                            "latest_extracted_answer": (
                                "pass" if loose_output.follow_all_instructions else "fail"
                            ),
                            "latest_runtime_seconds": response.runtime_seconds,
                            "latest_subject": ",".join(row.get("instruction_id_list") or []),
                        },
                    )
                maybe_reset_runtime(
                    client=self.client,
                    settings=self.settings,
                    model=model,
                    task="ifbench",
                    variant_name=variant.name,
                    completed_samples=index,
                    total_samples=total_samples,
                    progress_callback=progress_callback,
                    language="default",
                )

        summary = build_summary(
            settings=self.settings,
            model=model,
            strict_counts=strict_counts,
            loose_counts=loose_counts,
            strict_outputs=strict_outputs,
            loose_outputs=loose_outputs,
            total_runtime=total_runtime,
            samples_path=samples_path,
        )
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, sort_keys=True, default=str)

        return metrics_from_summary(summary)

    def _model_name(self, variant: VariantConfig) -> str:
        configured = self._configured_model_name(variant)
        if configured.lower() not in {"auto", "@first", "first-loaded"}:
            self.client.require_model(configured)
            return configured
        models = self.client.list_models()
        if not models:
            raise RuntimeError(
                f"No models returned by {self.client.models_endpoint()}; "
                "set `api_model` explicitly in the config."
            )
        return models[0]

    def _configured_model_name(self, variant: VariantConfig) -> str:
        return variant.api_model or variant.model_repo or variant.name

    def _load_dataset(self) -> list[dict[str, Any]]:
        try:
            from datasets import load_dataset
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "IFBench requires the `datasets` package. "
                "Install with `pip install -e '.[eval]'`."
            ) from exc

        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            dataset = load_dataset(
                self.settings.dataset_name,
                split=self.settings.split,
                revision=self.settings.dataset_revision,
                download_mode="reuse_dataset_if_exists",
            )
        return [dict(row) for row in dataset]


def build_summary(
    *,
    settings: IFBenchSettings,
    model: str,
    strict_counts: ScoreCounts,
    loose_counts: ScoreCounts,
    strict_outputs: list[Any],
    loose_outputs: list[Any],
    total_runtime: float,
    samples_path: Path,
) -> dict[str, Any]:
    return {
        "task": "ifbench",
        "dataset": settings.dataset_name,
        "dataset_revision": settings.dataset_revision,
        "split": settings.split,
        "model": model,
        "provider": settings.provider,
        "base_url": settings.base_url,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "top_p": settings.top_p,
        "stop": settings.stop,
        "seed": settings.seed,
        "reasoning_effort": settings.reasoning_effort,
        "response_format": settings.response_format,
        "request_extra": settings.request_extra,
        "evaluator": settings.evaluator,
        "total": loose_counts.prompt_total,
        "strict_prompt_correct": strict_counts.prompt_correct,
        "strict_instruction_correct": strict_counts.instruction_correct,
        "loose_prompt_correct": loose_counts.prompt_correct,
        "loose_instruction_correct": loose_counts.instruction_correct,
        "instruction_total": loose_counts.instruction_total,
        "ifbench_prompt_level_strict": strict_counts.prompt_accuracy,
        "ifbench_instruction_level_strict": strict_counts.instruction_accuracy,
        "ifbench_prompt_level_loose": loose_counts.prompt_accuracy,
        "ifbench_instruction_level_loose": loose_counts.instruction_accuracy,
        "strict_breakdown": _breakdown(strict_outputs),
        "loose_breakdown": _breakdown(loose_outputs),
        "runtime_seconds": total_runtime,
        "samples_path": str(samples_path),
    }


def metrics_from_summary(summary: dict[str, Any]) -> list[MetricResult]:
    raw = {"summary": summary}
    metrics = [
        MetricResult(
            "ifbench_prompt_level_loose",
            _as_float(summary.get("ifbench_prompt_level_loose")),
            "fraction",
            raw,
        ),
        MetricResult(
            "ifbench_instruction_level_loose",
            _as_float(summary.get("ifbench_instruction_level_loose")),
            "fraction",
            raw,
        ),
        MetricResult(
            "ifbench_prompt_level_strict",
            _as_float(summary.get("ifbench_prompt_level_strict")),
            "fraction",
            raw,
        ),
        MetricResult(
            "ifbench_instruction_level_strict",
            _as_float(summary.get("ifbench_instruction_level_strict")),
            "fraction",
            raw,
        ),
        MetricResult("ifbench_samples", _as_float(summary.get("total")), "count", raw),
        MetricResult(
            "ifbench_runtime_seconds",
            _as_float(summary.get("runtime_seconds")),
            "seconds",
            raw,
        ),
    ]
    metrics.extend(efficiency_metrics_from_summary(summary))
    return metrics


def _score_response(row: dict[str, Any], response: str) -> tuple[Any, Any]:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="pkg_resources is deprecated.*")
            import evaluation_lib
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "IFBench evaluator is missing. Install with `pip install -e '.[eval]'`."
        ) from exc

    prompt_to_response = {str(row.get("prompt") or ""): response}
    strict_input = _input_example(evaluation_lib, row)
    loose_input = _input_example(evaluation_lib, row)
    return (
        evaluation_lib.test_instruction_following_strict(strict_input, prompt_to_response),
        evaluation_lib.test_instruction_following_loose(loose_input, prompt_to_response),
    )


def _input_example(evaluation_lib: Any, row: dict[str, Any]) -> Any:
    kwargs = [
        {key: value for key, value in dict(item).items() if value is not None}
        for item in list(row.get("kwargs") or [])
    ]
    return evaluation_lib.InputExample(
        key=row.get("key"),
        instruction_id_list=list(row.get("instruction_id_list") or []),
        prompt=str(row.get("prompt") or ""),
        kwargs=copy.deepcopy(kwargs),
    )


def _add_counts(counts: ScoreCounts, output: Any) -> None:
    counts.prompt_total += 1
    counts.prompt_correct += int(bool(output.follow_all_instructions))
    counts.instruction_total += len(output.follow_instruction_list)
    counts.instruction_correct += sum(bool(item) for item in output.follow_instruction_list)


def _breakdown(outputs: list[Any]) -> dict[str, dict[str, float | int | None]]:
    totals: dict[str, int] = {}
    correct: dict[str, int] = {}
    for output in outputs:
        for instruction_id, followed in zip(
            output.instruction_id_list,
            output.follow_instruction_list,
            strict=False,
        ):
            family = str(instruction_id).split(":")[0]
            totals[family] = totals.get(family, 0) + 1
            correct[family] = correct.get(family, 0) + int(bool(followed))
    return {
        key: {
            "total": totals[key],
            "correct": correct.get(key, 0),
            "accuracy": correct.get(key, 0) / totals[key] if totals[key] else None,
        }
        for key in sorted(totals)
    }


def _strip_reasoning(text: str) -> str:
    text = re.sub(
        r"<(?:think|thinking|reasoning)>.*?</(?:think|thinking|reasoning)>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(
        r"<(?:think|thinking|reasoning)>.*",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return text.strip()


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _safe_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)
    return safe.strip("_") or "variant"
