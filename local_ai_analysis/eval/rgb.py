from __future__ import annotations

import json
import math
import random
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from local_ai_analysis.adapters.native import create_native_client
from local_ai_analysis.config import RGBSettings, VariantConfig
from local_ai_analysis.metrics import MetricResult


ProgressCallback = Callable[[str, dict[str, Any]], None]

RGB_DATASETS = [
    "en_refine",
    "zh_refine",
    "en",
    "zh",
    "en_int",
    "zh_int",
    "en_fact",
    "zh_fact",
]

THINKING_BLOCK_RE = re.compile(
    r"<(?:think|thinking|reasoning)>.*?</(?:think|thinking|reasoning)>",
    flags=re.IGNORECASE | re.DOTALL,
)
OPEN_THINKING_BLOCK_RE = re.compile(
    r"<(?:think|thinking|reasoning)>.*",
    flags=re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class ScoredRGBResponse:
    labels: list[int]
    correct: bool
    rejected: bool
    fact_detected: bool
    fact_corrected: bool


class RGBRunner:
    def __init__(self, settings: RGBSettings):
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
            f"POST {self.client.planned_endpoint()} model={model} "
            f"dataset={self.settings.dataset_name}/{self.settings.dataset} "
            f"noise_rate={self.settings.noise_rate} passage_num={self.settings.passage_num}"
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
        task_mode = _task_mode(self.settings.dataset, self.settings.noise_rate)

        if progress_callback:
            progress_callback(
                "dataset_loading_started",
                {
                    "task": "rgb",
                    "variant": variant.name,
                    "dataset": self.settings.dataset_name,
                    "languages": [self.settings.dataset],
                    "split": task_mode,
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
                    "task": "rgb",
                    "variant": variant.name,
                    "dataset": self.settings.dataset_name,
                    "languages": [self.settings.dataset],
                    "split": task_mode,
                    "total_samples": total_samples,
                },
            )
            progress_callback(
                "task_sample_plan",
                {
                    "task": "rgb",
                    "variant": variant.name,
                    "total_samples": total_samples,
                    "languages": [self.settings.dataset],
                },
            )

        total_runtime = 0.0
        score_rows: list[dict[str, Any]] = []

        with samples_path.open("w", encoding="utf-8") as sample_file:
            for index, row in enumerate(rows, start=1):
                query, answer, docs = process_data(row, self.settings)
                prompt = render_prompt(self.settings, query=query, docs=docs)
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
                parsed_output = (
                    _strip_reasoning(raw_output) if self.settings.strip_thinking else raw_output
                )
                scored = score_response(
                    parsed_output,
                    answer,
                    dataset=self.settings.dataset,
                    noise_rate=self.settings.noise_rate,
                )
                score_rows.append(
                    {
                        "sample_id": row.get("id"),
                        "correct": scored.correct,
                        "rejected": scored.rejected,
                        "fact_detected": scored.fact_detected,
                        "fact_corrected": scored.fact_corrected,
                    }
                )

                sample_record = {
                    "dataset": self.settings.dataset_name,
                    "dataset_revision": self.settings.dataset_revision,
                    "rgb_dataset": self.settings.dataset,
                    "task_mode": task_mode,
                    "sample_id": row.get("id"),
                    "query": query,
                    "answer": answer,
                    "docs": docs,
                    "noise_rate": self.settings.noise_rate,
                    "passage_num": self.settings.passage_num,
                    "correct_rate": self.settings.correct_rate,
                    "prompt": prompt,
                    "raw_output": raw_output,
                    "parsed_output": parsed_output,
                    "labels": scored.labels,
                    "correct": scored.correct,
                    "rejected": scored.rejected,
                    "fact_detected": scored.fact_detected,
                    "fact_corrected": scored.fact_corrected,
                    "evaluator": self.settings.evaluator,
                    "runtime_seconds": response.runtime_seconds,
                    "usage": response.raw.get("usage"),
                    "raw_response": response.raw,
                }
                sample_file.write(
                    json.dumps(sample_record, ensure_ascii=False, default=str) + "\n"
                )

                if progress_callback:
                    progress_callback(
                        "task_progress",
                        {
                            "task": "rgb",
                            "variant": variant.name,
                            "language": self.settings.dataset,
                            "completed_samples": index,
                            "total_samples": total_samples,
                            "latest_score": 1.0 if scored.correct else 0.0,
                            "latest_correct": scored.correct,
                            "latest_invalid": False,
                            "latest_extracted_answer": "pass" if scored.correct else "fail",
                            "latest_runtime_seconds": response.runtime_seconds,
                            "latest_subject": task_mode,
                        },
                    )

        summary = build_summary(
            settings=self.settings,
            model=model,
            task_mode=task_mode,
            score_rows=score_rows,
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
        path = _dataset_cache_path(self.settings)
        if not path.exists():
            _download_dataset(self.settings, path)
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows


def process_data(instance: dict[str, Any], settings: RGBSettings) -> tuple[str, Any, list[str]]:
    rng = random.Random(settings.seed if settings.seed is not None else 2333)
    query = str(instance["query"])
    answer = instance["answer"]
    passage_num = settings.passage_num
    noise_rate = settings.noise_rate
    neg_num = math.ceil(passage_num * noise_rate)
    pos_num = passage_num - neg_num

    if "_int" in settings.dataset:
        positive_groups = [
            [str(doc) for doc in group]
            for group in (instance.get("positive") or [])
            if isinstance(group, list)
        ]
        for group in positive_groups:
            rng.shuffle(group)
        docs = [group[0] for group in positive_groups if group]
        if len(docs) < pos_num and positive_groups:
            max_docs = max(len(group) for group in positive_groups)
            for offset in range(1, max_docs):
                for group in positive_groups:
                    if len(group) > offset:
                        docs.append(group[offset])
                        if len(docs) == pos_num:
                            break
                if len(docs) == pos_num:
                    break
        neg_num = passage_num - len(docs)
        if neg_num > 0:
            docs.extend(str(doc) for doc in (instance.get("negative") or [])[:neg_num])
    elif "_fact" in settings.dataset:
        correct_num = math.ceil(passage_num * settings.correct_rate)
        pos_num = max(0, passage_num - neg_num - correct_num)
        positives = [str(doc) for doc in (instance.get("positive") or [])]
        wrong_positives = [str(doc) for doc in (instance.get("positive_wrong") or [])]
        index_pool = list(range(len(positives)))
        selected = rng.sample(index_pool, min(len(index_pool), pos_num))
        docs = [wrong_positives[i] for i in selected if i < len(wrong_positives)]
        remain = [idx for idx in index_pool if idx not in selected]
        if correct_num > 0 and remain:
            docs.extend(positives[i] for i in rng.sample(remain, min(len(remain), correct_num)))
        if neg_num > 0:
            docs.extend(str(doc) for doc in (instance.get("negative") or [])[:neg_num])
    else:
        if noise_rate == 1:
            neg_num = passage_num
            pos_num = 0
        else:
            positive_count = len(instance.get("positive") or [])
            negative_count = len(instance.get("negative") or [])
            if neg_num > negative_count:
                neg_num = negative_count
                pos_num = passage_num - neg_num
            elif pos_num > positive_count:
                pos_num = positive_count
                neg_num = passage_num - pos_num
        docs = [str(doc) for doc in (instance.get("positive") or [])[:pos_num]]
        docs.extend(str(doc) for doc in (instance.get("negative") or [])[:neg_num])

    rng.shuffle(docs)
    return query, answer, docs


def render_prompt(settings: RGBSettings, *, query: str, docs: list[str]) -> str:
    language = _language(settings.dataset)
    if language == "zh":
        system = settings.system_prompt_zh
        instruction = settings.instruction_template_zh
    else:
        system = settings.system_prompt_en
        instruction = settings.instruction_template_en
    docs_text = "\n".join(docs)
    return f"{system}\n\n{instruction.format(QUERY=query, DOCS=docs_text)}"


def score_response(
    prediction: str,
    ground_truth: Any,
    *,
    dataset: str,
    noise_rate: float,
) -> ScoredRGBResponse:
    normalized_prediction = prediction.replace(" ", "") if "zh" in dataset else prediction
    rejected = _is_rejection(normalized_prediction)
    if rejected:
        labels = [-1]
    else:
        labels = check_answer(normalized_prediction, ground_truth)

    fact_detected = _has_factual_error_marker(normalized_prediction)
    correct = labels[0] == -1 if noise_rate == 1 else (0 not in labels and 1 in labels)
    fact_corrected = bool(fact_detected and 0 not in labels and 1 in labels)
    return ScoredRGBResponse(
        labels=labels,
        correct=correct,
        rejected=rejected,
        fact_detected=fact_detected,
        fact_corrected=fact_corrected,
    )


def check_answer(prediction: str, ground_truth: Any) -> list[int]:
    candidate = prediction.lower()
    answers = ground_truth if isinstance(ground_truth, list) else [ground_truth]
    labels: list[int] = []
    for answer in answers:
        if isinstance(answer, list):
            labels.append(int(any(str(item).lower() in candidate for item in answer)))
        else:
            labels.append(int(str(answer).lower() in candidate))
    return labels


def build_summary(
    *,
    settings: RGBSettings,
    model: str,
    task_mode: str,
    score_rows: list[dict[str, Any]],
    total_runtime: float,
    samples_path: Path,
) -> dict[str, Any]:
    total = len(score_rows)
    correct = sum(int(row["correct"]) for row in score_rows)
    rejected = sum(int(row["rejected"]) for row in score_rows)
    fact_detected = sum(int(row["fact_detected"]) for row in score_rows)
    fact_corrected = sum(int(row["fact_corrected"]) for row in score_rows)
    return {
        "task": "rgb",
        "dataset": settings.dataset_name,
        "dataset_revision": settings.dataset_revision,
        "rgb_dataset": settings.dataset,
        "task_mode": task_mode,
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
        "noise_rate": settings.noise_rate,
        "passage_num": settings.passage_num,
        "correct_rate": settings.correct_rate,
        "evaluator": settings.evaluator,
        "total": total,
        "correct": correct,
        "rejected": rejected,
        "fact_detected": fact_detected,
        "fact_corrected": fact_corrected,
        "rgb_all_rate": correct / total if total else None,
        "rgb_accuracy": correct / total if total and task_mode != "negative_rejection" else None,
        "rgb_rejection_rate": rejected / total if total and settings.noise_rate == 1 else None,
        "rgb_fact_check_rate": (
            fact_detected / total if total and "_fact" in settings.dataset else None
        ),
        "rgb_error_correction_rate": (
            fact_corrected / fact_detected
            if fact_detected and "_fact" in settings.dataset
            else (0.0 if "_fact" in settings.dataset else None)
        ),
        "runtime_seconds": total_runtime,
        "samples_path": str(samples_path),
    }


def metrics_from_summary(summary: dict[str, Any]) -> list[MetricResult]:
    raw = {"summary": summary}
    metrics = [
        MetricResult("rgb_all_rate", _as_float(summary.get("rgb_all_rate")), "fraction", raw),
        MetricResult("rgb_accuracy", _as_float(summary.get("rgb_accuracy")), "fraction", raw),
        MetricResult(
            "rgb_rejection_rate",
            _as_float(summary.get("rgb_rejection_rate")),
            "fraction",
            raw,
        ),
        MetricResult(
            "rgb_fact_check_rate",
            _as_float(summary.get("rgb_fact_check_rate")),
            "fraction",
            raw,
        ),
        MetricResult(
            "rgb_error_correction_rate",
            _as_float(summary.get("rgb_error_correction_rate")),
            "fraction",
            raw,
        ),
        MetricResult("rgb_samples", _as_float(summary.get("total")), "samples", raw),
        MetricResult(
            "rgb_runtime_seconds",
            _as_float(summary.get("runtime_seconds")),
            "seconds",
            raw,
        ),
    ]
    return metrics


def _dataset_cache_path(settings: RGBSettings) -> Path:
    revision = _safe_name(settings.dataset_revision)
    return Path(settings.data_cache_dir) / revision / f"{settings.dataset}.json"


def _download_dataset(settings: RGBSettings, path: Path) -> None:
    if settings.dataset not in RGB_DATASETS:
        raise RuntimeError(
            f"Unsupported RGB dataset: {settings.dataset}. Supported: {', '.join(RGB_DATASETS)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    url = (
        "https://raw.githubusercontent.com/chen700564/RGB/"
        f"{settings.dataset_revision}/data/{settings.dataset}.json"
    )
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            path.write_bytes(response.read())
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Could not download RGB dataset from {url}: {exc}") from exc
    if path.stat().st_size == 0:
        raise RuntimeError(f"Downloaded RGB dataset from {url} is empty")


def _task_mode(dataset: str, noise_rate: float) -> str:
    if "_int" in dataset:
        return "information_integration"
    if "_fact" in dataset:
        return "counterfactual_robustness"
    if noise_rate == 1:
        return "negative_rejection"
    return "noise_robustness"


def _language(dataset: str) -> str:
    return "zh" if dataset.startswith("zh") else "en"


def _is_rejection(prediction: str) -> bool:
    lowered = prediction.lower()
    return "信息不足" in prediction or "insufficient information" in lowered


def _has_factual_error_marker(prediction: str) -> bool:
    lowered = prediction.lower()
    return "事实性错误" in prediction or "factual errors" in lowered


def _strip_reasoning(text: str) -> str:
    stripped = THINKING_BLOCK_RE.sub("", text)
    stripped = OPEN_THINKING_BLOCK_RE.sub("", stripped)
    return stripped.strip()


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return safe or "value"


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
