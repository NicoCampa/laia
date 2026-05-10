from __future__ import annotations

import base64
import json
import math
import random
import re
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Callable

from local_ai_analysis.adapters.native import ImagePayload, create_native_client
from local_ai_analysis.config import OCRBenchV2Settings, VariantConfig
from local_ai_analysis.metrics import MetricResult


ProgressCallback = Callable[[str, dict[str, Any]], None]

THINKING_BLOCK_RE = re.compile(
    r"<(?:think|thinking|reasoning)>.*?</(?:think|thinking|reasoning)>",
    flags=re.IGNORECASE | re.DOTALL,
)
OPEN_THINKING_BLOCK_RE = re.compile(
    r"<(?:think|thinking|reasoning)>.*",
    flags=re.IGNORECASE | re.DOTALL,
)

EN_GROUPS: dict[str, set[str]] = {
    "text_recognition": {
        "text recognition en",
        "fine-grained text recognition en",
        "full-page OCR en",
    },
    "text_detection": {"text grounding en", "VQA with position en"},
    "text_spotting": {"text spotting en"},
    "relationship_extraction": {
        "key information extraction en",
        "key information mapping en",
    },
    "element_parsing": {
        "document parsing en",
        "chart parsing en",
        "table parsing en",
        "formula recognition en",
    },
    "mathematical_calculation": {"math QA en", "text counting en"},
    "visual_text_understanding": {
        "document classification en",
        "cognition VQA en",
        "diagram QA en",
    },
    "knowledge_reasoning": {
        "reasoning VQA en",
        "science QA en",
        "APP agent en",
        "ASCII art classification en",
    },
}

CN_GROUPS: dict[str, set[str]] = {
    "text_recognition": {"full-page OCR cn"},
    "relationship_extraction": {
        "key information extraction cn",
        "handwritten answer extraction cn",
    },
    "element_parsing": {
        "document parsing cn",
        "table parsing cn",
        "formula recognition cn",
    },
    "visual_text_understanding": {"cognition VQA cn"},
    "knowledge_reasoning": {"reasoning VQA cn", "text translation cn"},
}

OCR_TEXT_TASKS = {
    "fine-grained text recognition en",
    "full-page OCR en",
    "full-page OCR cn",
    "text translation cn",
    "handwritten answer extraction cn",
}
FORMULA_TASKS = {"formula recognition en", "formula recognition cn"}


@dataclass
class ScoreBucket:
    total: int = 0
    score: float = 0.0

    @property
    def average(self) -> float | None:
        return self.score / self.total if self.total else None


@dataclass(frozen=True)
class SampleRef:
    config_name: str
    config_order: int
    dataset: Any
    index: int


class OCRBenchV2Runner:
    def __init__(self, settings: OCRBenchV2Settings):
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
            f"dataset={self.settings.dataset_name} "
            f"configs={','.join(self.settings.dataset_configs)} split={self.settings.split}"
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
                    "task": "ocrbench-v2",
                    "variant": variant.name,
                    "dataset": self.settings.dataset_name,
                    "languages": self.settings.dataset_configs,
                    "split": self.settings.split,
                    "dataset_revision": self.settings.dataset_revision,
                },
            )

        datasets = self._load_datasets()
        selected_samples = select_samples(datasets, self.settings)
        total_samples = len(selected_samples)
        if progress_callback:
            progress_callback(
                "dataset_loading_completed",
                {
                    "task": "ocrbench-v2",
                    "variant": variant.name,
                    "dataset": self.settings.dataset_name,
                    "languages": self.settings.dataset_configs,
                    "split": self.settings.split,
                    "total_samples": total_samples,
                },
            )
            progress_callback(
                "task_sample_plan",
                {
                    "task": "ocrbench-v2",
                    "variant": variant.name,
                    "total_samples": total_samples,
                    "languages": self.settings.dataset_configs,
                },
            )

        completed_samples = 0
        total_runtime = 0.0
        sample_scores: list[dict[str, Any]] = []

        with samples_path.open("w", encoding="utf-8") as sample_file:
            for sample_ref in selected_samples:
                row = dict(sample_ref.dataset[int(sample_ref.index)])
                prompt = render_prompt(self.settings.prompt_template, row)
                image = _image_payload(row, self.settings.image_format)
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
                    images=[image],
                )
                total_runtime += response.runtime_seconds
                raw_output = response.text
                parsed_output = (
                    _strip_reasoning(raw_output)
                    if self.settings.strip_thinking
                    else raw_output
                )
                score = score_response(row, parsed_output)
                task_type = _task_type(row)
                completed_samples += 1
                sample_scores.append(
                    {
                        "config": sample_ref.config_name,
                        "type": task_type,
                        "score": score,
                    }
                )

                sample_record = {
                    "dataset": self.settings.dataset_name,
                    "dataset_revision": self.settings.dataset_revision,
                    "dataset_config": sample_ref.config_name,
                    "split": self.settings.split,
                    "dataset_name": row.get("dataset_name"),
                    "type": task_type,
                    "sample_id": row.get("id"),
                    "sample_index": sample_ref.index,
                    "image_path": row.get("image_path"),
                    "question": row.get("question"),
                    "prompt": prompt,
                    "answers": _answers(row),
                    "bbox": row.get("bbox"),
                    "image_shape": row.get("image_shape"),
                    "image_count": 1,
                    "raw_output": raw_output,
                    "parsed_output": parsed_output,
                    "score": score,
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
                            "task": "ocrbench-v2",
                            "variant": variant.name,
                            "language": task_type,
                            "completed_samples": completed_samples,
                            "total_samples": total_samples,
                            "latest_score": score,
                            "latest_correct": score >= 0.5,
                            "latest_extracted_answer": f"{score:.2f}",
                            "latest_runtime_seconds": response.runtime_seconds,
                            "latest_subject": task_type,
                        },
                    )

        summary = build_summary(
            settings=self.settings,
            model=model,
            sample_scores=sample_scores,
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

    def _load_datasets(self) -> list[tuple[str, Any]]:
        try:
            from datasets import load_dataset
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "OCRBench v2 requires the `datasets` package. "
                "Install with `pip install -e '.[eval]'`."
            ) from exc

        datasets: list[tuple[str, Any]] = []
        for config_name in self.settings.dataset_configs:
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                dataset = load_dataset(
                    self.settings.dataset_name,
                    config_name,
                    split=self.settings.split,
                    revision=self.settings.dataset_revision,
                    download_mode="reuse_dataset_if_exists",
                )
            datasets.append((config_name, dataset))
        return datasets


def render_prompt(template: str, row: dict[str, Any]) -> str:
    return template.format(
        question=row.get("question", ""),
        dataset_name=row.get("dataset_name", ""),
        type=row.get("type", ""),
        raw_text=row.get("raw_text", ""),
        content=row.get("content", ""),
    )


def build_summary(
    *,
    settings: OCRBenchV2Settings,
    model: str,
    sample_scores: list[dict[str, Any]],
    total_runtime: float,
    samples_path: Path,
) -> dict[str, Any]:
    total = len(sample_scores)
    score_sum = sum(float(item["score"]) for item in sample_scores)
    type_scores = _bucket_by(sample_scores, "type")
    config_scores = _bucket_by(sample_scores, "config")
    en_group_scores, cn_group_scores = _group_scores(sample_scores)
    en_score = _mean(en_group_scores.values())
    cn_score = _mean(cn_group_scores.values())
    macro_candidates = [value for value in [en_score, cn_score] if value is not None]

    return {
        "task": "ocrbench_v2",
        "dataset": settings.dataset_name,
        "dataset_revision": settings.dataset_revision,
        "split": settings.split,
        "dataset_configs": settings.dataset_configs,
        "sample_limit": settings.sample_limit,
        "sample_strategy": settings.sample_strategy,
        "sample_seed": settings.sample_seed,
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
        "image_format": settings.image_format,
        "total": total,
        "ocrbench_v2_score": sum(macro_candidates) / len(macro_candidates)
        if macro_candidates
        else None,
        "ocrbench_v2_micro_score": score_sum / total if total else None,
        "ocrbench_v2_en_score": en_score,
        "ocrbench_v2_cn_score": cn_score,
        "english_group_scores": en_group_scores,
        "chinese_group_scores": cn_group_scores,
        "type_scores": type_scores,
        "config_scores": config_scores,
        "runtime_seconds": total_runtime,
        "samples_path": str(samples_path),
    }


def metrics_from_summary(summary: dict[str, Any]) -> list[MetricResult]:
    raw = {"summary": summary}
    metrics = [
        MetricResult(
            "ocrbench_v2_score",
            _as_float(summary.get("ocrbench_v2_score")),
            "fraction",
            raw,
        ),
        MetricResult(
            "ocrbench_v2_micro_score",
            _as_float(summary.get("ocrbench_v2_micro_score")),
            "fraction",
            raw,
        ),
        MetricResult(
            "ocrbench_v2_en_score",
            _as_float(summary.get("ocrbench_v2_en_score")),
            "fraction",
            raw,
        ),
        MetricResult(
            "ocrbench_v2_cn_score",
            _as_float(summary.get("ocrbench_v2_cn_score")),
            "fraction",
            raw,
        ),
        MetricResult("ocrbench_v2_samples", _as_float(summary.get("total")), "samples", raw),
        MetricResult(
            "ocrbench_v2_runtime_seconds",
            _as_float(summary.get("runtime_seconds")),
            "seconds",
            raw,
        ),
    ]
    for prefix, group_scores in (
        ("en", summary.get("english_group_scores") or {}),
        ("cn", summary.get("chinese_group_scores") or {}),
    ):
        for group, value in group_scores.items():
            metrics.append(
                MetricResult(
                    f"ocrbench_v2_{prefix}_{group}",
                    _as_float(value),
                    "fraction",
                    raw,
                )
            )
    return metrics


def select_samples(
    datasets: list[tuple[str, Any]],
    settings: OCRBenchV2Settings,
) -> list[SampleRef]:
    total = sum(len(dataset) for _, dataset in datasets)
    if settings.sample_limit is None or settings.sample_limit >= total:
        return [
            SampleRef(config_name, config_order, dataset, index)
            for config_order, (config_name, dataset) in enumerate(datasets)
            for index in range(len(dataset))
        ]

    limit = max(0, int(settings.sample_limit))
    strategy = (settings.sample_strategy or "stratified").strip().lower()
    if strategy in {"first", "head"}:
        return _first_samples(datasets, limit)
    if strategy not in {"stratified", "balanced", "macro"}:
        raise RuntimeError(
            "OCRBench v2 sample_strategy must be one of: stratified, balanced, macro, first."
        )
    return _stratified_samples(datasets, limit, seed=settings.sample_seed)


def _first_samples(datasets: list[tuple[str, Any]], limit: int) -> list[SampleRef]:
    selected: list[SampleRef] = []
    remaining = limit
    for config_order, (config_name, dataset) in enumerate(datasets):
        if remaining <= 0:
            break
        count = min(len(dataset), remaining)
        selected.extend(
            SampleRef(config_name, config_order, dataset, index) for index in range(count)
        )
        remaining -= count
    return selected


def _stratified_samples(
    datasets: list[tuple[str, Any]],
    limit: int,
    *,
    seed: int,
) -> list[SampleRef]:
    rng = random.Random(seed)
    strata: dict[str, list[SampleRef]] = {}
    for config_order, (config_name, dataset) in enumerate(datasets):
        task_types = _dataset_task_types(dataset)
        for index, task_type in enumerate(task_types):
            key = _score_group_key(config_name, task_type)
            strata.setdefault(key, []).append(SampleRef(config_name, config_order, dataset, index))

    allocations = _balanced_allocations(strata, limit)
    selected: list[SampleRef] = []
    for key, count in allocations.items():
        candidates = list(strata[key])
        rng.shuffle(candidates)
        selected.extend(candidates[:count])
    return sorted(selected, key=lambda item: (item.config_order, item.index))


def _dataset_task_types(dataset: Any) -> list[str]:
    column_names = set(getattr(dataset, "column_names", []) or [])
    if "type" in column_names:
        return [str(value or "unknown") for value in dataset["type"]]
    if "dataset_name" in column_names:
        return [str(value or "unknown") for value in dataset["dataset_name"]]
    return [_task_type(dict(dataset[index])) for index in range(len(dataset))]


def _score_group_key(config_name: str, task_type: str) -> str:
    for group, task_types in EN_GROUPS.items():
        if task_type in task_types:
            return f"en:{group}"
    for group, task_types in CN_GROUPS.items():
        if task_type in task_types:
            return f"cn:{group}"
    return f"{config_name}:{task_type}"


def _balanced_allocations(strata: dict[str, list[SampleRef]], limit: int) -> dict[str, int]:
    if not strata or limit <= 0:
        return {}

    allocations = {key: 0 for key in strata}
    remaining = min(limit, sum(len(items) for items in strata.values()))
    active = set(strata)
    while active and remaining > 0:
        quota = max(1, remaining // len(active))
        progressed = False
        for key in sorted(active):
            available = len(strata[key]) - allocations[key]
            if available <= 0:
                continue
            add = min(quota, available, remaining)
            allocations[key] += add
            remaining -= add
            progressed = progressed or add > 0
            if remaining <= 0:
                break
        active = {key for key in active if allocations[key] < len(strata[key])}
        if not progressed:
            break
    return {key: count for key, count in allocations.items() if count}


def score_response(row: dict[str, Any], prediction: str) -> float:
    task_type = _task_type(row)
    answers = _answers(row)
    if not prediction:
        return 0.0

    eval_method = str(row.get("eval") or "").strip().lower()
    if eval_method == "multiple choice":
        return _multiple_choice_score(prediction, answers)
    if eval_method == "case sensitive":
        return _vqa_score(prediction, answers, case_sensitive=True)

    if task_type == "text grounding en":
        return _bbox_iou_score(prediction, answers)
    if task_type == "VQA with position en":
        content_score = _vqa_score(prediction, answers)
        bbox_score = _bbox_iou_score(prediction, row.get("bbox"))
        return 0.5 * content_score + 0.5 * bbox_score if row.get("bbox") else content_score
    if task_type == "text counting en":
        return _counting_score(prediction, answers, eval_method)
    if task_type in FORMULA_TASKS:
        return _formula_score(prediction, answers)
    if task_type.endswith(" cn"):
        if task_type in OCR_TEXT_TASKS:
            return _ocr_text_score(prediction, answers)
        return _vqa_score(prediction, answers, chinese=True)
    if task_type in OCR_TEXT_TASKS:
        return _ocr_text_score(prediction, answers)
    return _vqa_score(prediction, answers)


def _image_payload(row: dict[str, Any], image_format: str) -> ImagePayload:
    image = row.get("image")
    if image is None:
        raise RuntimeError(
            "OCRBench v2 row does not include a decoded image. "
            "Install Pillow with `pip install -e '.[eval]'` and retry."
        )
    if isinstance(image, bytes):
        return ImagePayload(data=base64.b64encode(image).decode("ascii"), mime_type="image/png")
    if not hasattr(image, "save"):
        raise RuntimeError(f"Unsupported OCRBench v2 image object: {type(image).__name__}")

    normalized_format = image_format.upper()
    mime_type = f"image/{'jpeg' if normalized_format == 'JPEG' else normalized_format.lower()}"
    image_to_save = image
    if normalized_format == "JPEG" and getattr(image, "mode", None) not in {"RGB", "L"}:
        image_to_save = image.convert("RGB")
    buffer = BytesIO()
    image_to_save.save(buffer, format=normalized_format)
    return ImagePayload(data=base64.b64encode(buffer.getvalue()).decode("ascii"), mime_type=mime_type)


def _strip_reasoning(text: str) -> str:
    candidate = THINKING_BLOCK_RE.sub("", text).strip()
    return OPEN_THINKING_BLOCK_RE.sub("", candidate).strip()


def _task_type(row: dict[str, Any]) -> str:
    return str(row.get("type") or row.get("dataset_name") or "unknown")


def _answers(row: dict[str, Any]) -> list[Any]:
    answers = row.get("answers")
    if answers is None:
        return []
    if isinstance(answers, list):
        return answers
    return [answers]


def _vqa_score(
    prediction: Any,
    answers: list[Any],
    *,
    chinese: bool = False,
    case_sensitive: bool = False,
) -> float:
    score = 0.0
    candidate = _normalize_text(prediction, chinese=chinese, case_sensitive=case_sensitive)
    for answer_payload in answers:
        answer = _normalize_text(answer_payload, chinese=chinese, case_sensitive=case_sensitive)
        if not answer:
            continue
        token_count = len(answer.split("," if chinese else None))
        if token_count < (4 if chinese else 5):
            if answer in candidate:
                score = max(score, 1.0)
        else:
            similarity = _normalized_similarity(candidate, answer)
            if similarity >= 0.5:
                score = max(score, similarity)
    return score


def _ocr_text_score(prediction: str, answers: list[Any]) -> float:
    candidate = _normalize_text(prediction)
    return max((_normalized_similarity(candidate, _normalize_text(answer)) for answer in answers), default=0.0)


def _formula_score(prediction: str, answers: list[Any]) -> float:
    candidate = _remove_latex_text(_normalize_formula(prediction))
    for answer in answers:
        normalized_answer = _remove_latex_text(_normalize_formula(answer))
        if normalized_answer and normalized_answer in candidate:
            return 1.0
    return 0.0


def _counting_score(prediction: str, answers: list[Any], eval_method: str) -> float:
    if eval_method == "regression":
        predicted = _first_number(prediction)
        if predicted is None:
            return 0.0
        best = 0.0
        for answer in answers:
            expected = _first_number(str(answer))
            if expected is None or expected <= 0 or predicted <= 0 or predicted >= 2 * expected:
                continue
            score = 1 - abs(predicted - expected) / expected
            if score > 0.5:
                best = max(best, score)
        return best
    return _vqa_score(prediction, answers)


def _multiple_choice_score(prediction: str, answers: list[Any]) -> float:
    compact = "".join(char for char in prediction if char.isalpha())
    if not answers:
        return 0.0
    return 1.0 if compact == str(answers[0]) else 0.0


def _bbox_iou_score(prediction: str, answers: Any) -> float:
    predicted = _extract_bbox(prediction)
    expected = _extract_bbox(answers)
    if predicted is None or expected is None:
        return 0.0
    return _calculate_iou(predicted, expected)


def _extract_bbox(value: Any) -> list[float] | None:
    if isinstance(value, list | tuple) and len(value) >= 4:
        try:
            return [float(item) for item in value[:4]]
        except (TypeError, ValueError):
            return None
    text = str(value)
    bracket_match = re.findall(
        r"[\(\[]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
        r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\)\]]",
        text,
    )
    if bracket_match:
        return [float(item) for item in bracket_match[-1]]
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(numbers) >= 4:
        return [float(item) for item in numbers[-4:]]
    return None


def _calculate_iou(box1: list[float], box2: list[float]) -> float:
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    inter_area = max(0.0, x2_inter - x1_inter) * max(0.0, y2_inter - y1_inter)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0


def _normalize_text(value: Any, *, chinese: bool = False, case_sensitive: bool = False) -> str:
    text = str(value).strip().replace("\n", " ")
    if not case_sensitive:
        text = text.lower()
    if chinese:
        text = text.replace(" ", "")
    return re.sub(r"\s+", " ", text).strip()


def _normalize_formula(value: Any) -> str:
    return str(value).strip().replace("\n", " ").replace(" ", "")


def _remove_latex_text(value: str) -> str:
    return re.sub(r"\\text\{([^{}]*)\}", r"\1", value)


def _first_number(value: str) -> int | None:
    match = re.search(r"\d+", value)
    return int(match.group()) if match else None


def _normalized_similarity(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    length = max(len(left), len(right))
    if length == 0:
        return 0.0
    return 1.0 - (_levenshtein_distance(left, right) / length)


def _levenshtein_distance(left: str, right: str) -> int:
    if len(left) > len(right):
        left, right = right, left
    distances = list(range(len(left) + 1))
    for index_right, char_right in enumerate(right):
        next_distances = [index_right + 1]
        for index_left, char_left in enumerate(left):
            if char_left == char_right:
                next_distances.append(distances[index_left])
            else:
                next_distances.append(
                    1
                    + min(
                        distances[index_left],
                        distances[index_left + 1],
                        next_distances[-1],
                    )
                )
        distances = next_distances
    return distances[-1]


def _bucket_by(sample_scores: list[dict[str, Any]], key: str) -> dict[str, float]:
    buckets: dict[str, ScoreBucket] = {}
    for item in sample_scores:
        bucket = buckets.setdefault(str(item.get(key) or "unknown"), ScoreBucket())
        bucket.total += 1
        bucket.score += float(item.get("score") or 0.0)
    return {
        name: value.average
        for name, value in sorted(buckets.items())
        if value.average is not None
    }


def _group_scores(
    sample_scores: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, float]]:
    en_buckets = {group: ScoreBucket() for group in EN_GROUPS}
    cn_buckets = {group: ScoreBucket() for group in CN_GROUPS}
    for item in sample_scores:
        task_type = str(item.get("type") or "")
        score = float(item.get("score") or 0.0)
        for group, task_types in EN_GROUPS.items():
            if task_type in task_types:
                en_buckets[group].total += 1
                en_buckets[group].score += score
        for group, task_types in CN_GROUPS.items():
            if task_type in task_types:
                cn_buckets[group].total += 1
                cn_buckets[group].score += score
    return (
        {
            group: bucket.average
            for group, bucket in en_buckets.items()
            if bucket.average is not None
        },
        {
            group: bucket.average
            for group, bucket in cn_buckets.items()
            if bucket.average is not None
        },
    )


def _mean(values: Any) -> float | None:
    numeric_values = [
        float(value)
        for value in values
        if value is not None and not math.isnan(float(value))
    ]
    return sum(numeric_values) / len(numeric_values) if numeric_values else None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "variant"
