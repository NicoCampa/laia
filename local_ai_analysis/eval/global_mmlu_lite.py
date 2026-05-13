from __future__ import annotations

import json
import re
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Callable

from local_ai_analysis.adapters.native import create_native_client
from local_ai_analysis.config import GlobalMMLULiteSettings, VariantConfig
from local_ai_analysis.eval.efficiency import efficiency_metrics_from_summary
from local_ai_analysis.metrics import MetricResult


THINKING_BLOCK_RE = re.compile(
    r"<(?:think|thinking|reasoning)>.*?</(?:think|thinking|reasoning)>",
    flags=re.IGNORECASE | re.DOTALL,
)
OPEN_THINKING_BLOCK_RE = re.compile(
    r"<(?:think|thinking|reasoning)>.*",
    flags=re.IGNORECASE | re.DOTALL,
)
ANSWER_PATTERNS = [
    re.compile(r'(?i)"(?:answer|choice)"\s*:\s*"([ABCD])"'),
    re.compile(r"(?i)(?:final\s+answer|answer)\s*(?:is|:)?\s*[\(\[]?\s*([ABCD])\b"),
    re.compile(r"(?i)^\s*[\(\[]?\s*([ABCD])\s*[\)\].:,\s-]*"),
    re.compile(r"(?i)(?<![A-Z])([ABCD])(?![A-Z])"),
]
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass
class ScoreBucket:
    total: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.total if self.total else None


class GlobalMMLULiteRunner:
    def __init__(self, settings: GlobalMMLULiteSettings):
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
            f"model={model} dataset={self.settings.dataset_name} "
            f"languages={','.join(self.settings.languages)} split={self.settings.split}"
        )

    def run(
        self,
        variant: VariantConfig,
        progress_callback: ProgressCallback | None = None,
    ) -> list[MetricResult]:
        model = self._model_name(variant)
        self._preflight_model(model)
        out_dir = Path(self.settings.output_dir) / _safe_name(variant.name)
        out_dir.mkdir(parents=True, exist_ok=True)
        samples_path = out_dir / "samples.jsonl"
        summary_path = out_dir / "summary.json"
        if progress_callback:
            progress_callback(
                "dataset_loading_started",
                {
                    "task": "global-mmlu-lite",
                    "variant": variant.name,
                    "dataset": self.settings.dataset_name,
                    "languages": self.settings.languages,
                    "split": self.settings.split,
                    "dataset_revision": self.settings.dataset_revision,
                },
            )
        language_datasets = self._load_datasets()
        total_samples = sum(self._planned_sample_count(dataset) for _, dataset in language_datasets)
        if progress_callback:
            progress_callback(
                "dataset_loading_completed",
                {
                    "task": "global-mmlu-lite",
                    "variant": variant.name,
                    "dataset": self.settings.dataset_name,
                    "languages": self.settings.languages,
                    "split": self.settings.split,
                    "total_samples": total_samples,
                },
            )
        completed_samples = 0
        if progress_callback:
            progress_callback(
                "task_sample_plan",
                {
                    "task": "global-mmlu-lite",
                    "variant": variant.name,
                    "total_samples": total_samples,
                    "languages": self.settings.languages,
                },
            )

        language_scores: dict[str, ScoreBucket] = {}
        sensitivity_scores: dict[str, ScoreBucket] = {}
        invalid_count = 0
        total_runtime = 0.0

        with samples_path.open("w", encoding="utf-8") as sample_file:
            for language_index, (language, dataset) in enumerate(language_datasets):
                count = 0
                for row_payload in dataset:
                    row = dict(row_payload)
                    prompt = render_prompt(self.settings.prompt_template, row)
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
                    extracted = extract_answer(
                        raw_output,
                        strip_thinking=self.settings.strip_thinking,
                    )
                    gold = str(row.get("answer") or "").strip().upper()
                    correct = extracted == gold
                    invalid = extracted is None
                    if invalid:
                        invalid_count += 1
                    language_bucket = language_scores.setdefault(language, ScoreBucket())
                    language_bucket.total += 1
                    language_bucket.correct += int(correct)

                    label = str(row.get("cultural_sensitivity_label") or "unknown").lower()
                    sensitivity_bucket = sensitivity_scores.setdefault(label, ScoreBucket())
                    sensitivity_bucket.total += 1
                    sensitivity_bucket.correct += int(correct)

                    sample_record = {
                        "dataset": self.settings.dataset_name,
                        "dataset_revision": self.settings.dataset_revision,
                        "language": language,
                        "split": self.settings.split,
                        "sample_id": row.get("sample_id"),
                        "subject": row.get("subject"),
                        "subject_category": row.get("subject_category"),
                        "cultural_sensitivity_label": row.get("cultural_sensitivity_label"),
                        "prompt": prompt,
                        "raw_output": raw_output,
                        "extracted_answer": extracted,
                        "gold_answer": gold,
                        "correct": correct,
                        "invalid": invalid,
                        "parser": self.settings.parser_version,
                        "runtime_seconds": response.runtime_seconds,
                        "usage": response.raw.get("usage"),
                        "raw_response": response.raw,
                    }
                    sample_file.write(
                        json.dumps(sample_record, ensure_ascii=False, default=str) + "\n"
                    )
                    count += 1
                    completed_samples += 1
                    if progress_callback:
                        progress_callback(
                            "task_progress",
                            {
                                "task": "global-mmlu-lite",
                                "variant": variant.name,
                                "language": language,
                                "completed_samples": completed_samples,
                                "total_samples": total_samples,
                                "latest_correct": correct,
                                "latest_extracted_answer": extracted,
                                "latest_runtime_seconds": response.runtime_seconds,
                                "latest_subject": row.get("subject"),
                            },
                        )
                    if (
                        self.settings.sample_limit_per_language is not None
                        and count >= self.settings.sample_limit_per_language
                    ):
                        break
                if (
                    self.settings.restart_between_languages
                    and language_index < len(language_datasets) - 1
                ):
                    reset_error = None
                    try:
                        instance_id = self.client.reset_model_runtime(
                            model,
                            request_extra=self.settings.request_extra,
                        )
                    except Exception as exc:
                        instance_id = None
                        reset_error = str(exc)
                    if progress_callback:
                        progress_callback(
                            "runtime_cache_reset",
                            {
                                "task": "global-mmlu-lite",
                                "variant": variant.name,
                                "language": language,
                                "provider": self.settings.provider,
                                "model": model,
                                "instance_id": instance_id,
                                "error": reset_error,
                            },
                        )

        summary = build_summary(
            settings=self.settings,
            model=model,
            language_scores=language_scores,
            sensitivity_scores=sensitivity_scores,
            invalid_count=invalid_count,
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

    def _preflight_model(self, model: str) -> None:
        response = self.client.generate(
            model=model,
            prompt="Reply with only this letter: A",
            temperature=self.settings.temperature,
            max_tokens=16,
            top_p=self.settings.top_p,
            stop=self.settings.stop,
            seed=self.settings.seed,
            reasoning_effort=self.settings.reasoning_effort,
            response_format=self.settings.response_format,
            request_extra=self.settings.request_extra,
        )
        if extract_answer(response.text, strip_thinking=self.settings.strip_thinking) is not None:
            return

        message = _first_message(response.raw)
        reasoning_content = (
            message.get("reasoning_content")
            or message.get("reasoning")
            or message.get("thinking")
            or _first_reasoning_output(response.raw)
        )
        usage = response.raw.get("usage") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        reasoning_tokens = completion_details.get("reasoning_tokens")
        finish_reason = (response.raw.get("choices") or [{}])[0].get("finish_reason")
        if response.text == "" and reasoning_content:
            raise RuntimeError(
                "Preflight returned empty content but non-empty reasoning output. "
                "The server is still running the model in thinking/reasoning mode, so the "
                "benchmark would parse every answer as invalid. Disable the model's "
                "thinking/reasoning mode in the native server, then retry. "
                f"reasoning_effort={self.settings.reasoning_effort!r}, "
                f"finish_reason={finish_reason!r}, reasoning_tokens={reasoning_tokens!r}."
            )
        raise RuntimeError(
            "Preflight response could not be parsed as A/B/C/D. "
            f"content={response.text!r}, finish_reason={finish_reason!r}."
        )

    def _load_datasets(self) -> list[tuple[str, Any]]:
        try:
            from datasets import load_dataset
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Global MMLU Lite generation requires `datasets`. "
                "Install with `pip install -e '.[eval]'`."
            ) from exc

        language_datasets: list[tuple[str, Any]] = []
        for language in self.settings.languages:
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                dataset = load_dataset(
                    self.settings.dataset_name,
                    language,
                    split=self.settings.split,
                    revision=self.settings.dataset_revision,
                    download_mode="reuse_dataset_if_exists",
                )
            language_datasets.append((language, dataset))
        return language_datasets

    def _planned_sample_count(self, dataset: Any) -> int:
        dataset_count = len(dataset)
        if self.settings.sample_limit_per_language is None:
            return dataset_count
        return min(dataset_count, self.settings.sample_limit_per_language)


def render_prompt(template: str, row: dict[str, Any]) -> str:
    return template.format(
        question=row.get("question", ""),
        option_a=row.get("option_a", ""),
        option_b=row.get("option_b", ""),
        option_c=row.get("option_c", ""),
        option_d=row.get("option_d", ""),
        subject=row.get("subject", ""),
        subject_category=row.get("subject_category", ""),
    )


def extract_answer(text: str, *, strip_thinking: bool = True) -> str | None:
    candidate = text.strip()
    if strip_thinking:
        candidate = THINKING_BLOCK_RE.sub("", candidate).strip()
        candidate = OPEN_THINKING_BLOCK_RE.sub("", candidate).strip()
    for pattern in ANSWER_PATTERNS:
        match = pattern.search(candidate)
        if match:
            return match.group(1).upper()
    return None


def build_summary(
    *,
    settings: GlobalMMLULiteSettings,
    model: str,
    language_scores: dict[str, ScoreBucket],
    sensitivity_scores: dict[str, ScoreBucket],
    invalid_count: int,
    total_runtime: float,
    samples_path: Path,
) -> dict[str, Any]:
    total = sum(bucket.total for bucket in language_scores.values())
    correct = sum(bucket.correct for bucket in language_scores.values())
    language_accuracy = {
        language: bucket.accuracy for language, bucket in sorted(language_scores.items())
    }
    sensitivity_accuracy = {
        label: bucket.accuracy for label, bucket in sorted(sensitivity_scores.items())
    }
    macro_values = [value for value in language_accuracy.values() if value is not None]
    return {
        "task": "global_mmlu_lite_generate_pass_at_1",
        "dataset": settings.dataset_name,
        "dataset_revision": settings.dataset_revision,
        "split": settings.split,
        "languages": settings.languages,
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
        "restart_between_languages": settings.restart_between_languages,
        "parser": settings.parser_version,
        "prompt_template": settings.prompt_template,
        "total": total,
        "correct": correct,
        "invalid": invalid_count,
        "global_mmlu_lite_pass_at_1": sum(macro_values) / len(macro_values)
        if macro_values
        else None,
        "global_mmlu_lite_micro_pass_at_1": correct / total if total else None,
        "global_mmlu_lite_invalid_rate": invalid_count / total if total else None,
        "language_accuracy": language_accuracy,
        "cultural_sensitivity_accuracy": sensitivity_accuracy,
        "runtime_seconds": total_runtime,
        "samples_path": str(samples_path),
    }


def metrics_from_summary(summary: dict[str, Any]) -> list[MetricResult]:
    raw = {"summary": summary}
    metrics = [
        MetricResult(
            "global_mmlu_lite_pass_at_1",
            _as_float(summary.get("global_mmlu_lite_pass_at_1")),
            "fraction",
            raw,
        ),
        MetricResult(
            "global_mmlu_lite_micro_pass_at_1",
            _as_float(summary.get("global_mmlu_lite_micro_pass_at_1")),
            "fraction",
            raw,
        ),
        MetricResult(
            "global_mmlu_lite_invalid_rate",
            _as_float(summary.get("global_mmlu_lite_invalid_rate")),
            "fraction",
            raw,
        ),
        MetricResult(
            "global_mmlu_lite_samples",
            _as_float(summary.get("total")),
            "samples",
            raw,
        ),
        MetricResult(
            "global_mmlu_lite_runtime_seconds",
            _as_float(summary.get("runtime_seconds")),
            "seconds",
            raw,
        ),
    ]
    for language, value in (summary.get("language_accuracy") or {}).items():
        metrics.append(
            MetricResult(
                f"global_mmlu_lite_pass_at_1_{language}",
                _as_float(value),
                "fraction",
                raw,
            )
        )
    for label, value in (summary.get("cultural_sensitivity_accuracy") or {}).items():
        metrics.append(
            MetricResult(
                f"global_mmlu_lite_pass_at_1_{label}",
                _as_float(value),
                "fraction",
                raw,
            )
        )
    metrics.extend(efficiency_metrics_from_summary(summary))
    return metrics


def _first_message(raw: dict[str, Any]) -> dict[str, Any]:
    choices = raw.get("choices") or []
    if not choices:
        return {}
    message = choices[0].get("message")
    return message if isinstance(message, dict) else {}


def _first_reasoning_output(raw: dict[str, Any]) -> str | None:
    output = raw.get("output") or []
    for item in output:
        if isinstance(item, dict) and item.get("type") == "reasoning":
            return str(item.get("content") or "")
    return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "variant"
