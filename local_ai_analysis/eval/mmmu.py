from __future__ import annotations

import ast
import base64
import json
import random
import re
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Callable

from local_ai_analysis.adapters.native import ImagePayload, create_native_client
from local_ai_analysis.config import MMMUSettings, VariantConfig
from local_ai_analysis.eval.efficiency import efficiency_metrics_from_summary
from local_ai_analysis.eval.resume import (
    cache_key,
    load_resume_records,
    maybe_announce_resume,
    record_matches,
    sample_file_mode,
)
from local_ai_analysis.eval.runtime import maybe_reset_runtime
from local_ai_analysis.metrics import MetricResult


ProgressCallback = Callable[[str, dict[str, Any]], None]

MMMU_SUBJECTS = [
    "Accounting",
    "Agriculture",
    "Architecture_and_Engineering",
    "Art",
    "Art_Theory",
    "Basic_Medical_Science",
    "Biology",
    "Chemistry",
    "Clinical_Medicine",
    "Computer_Science",
    "Design",
    "Diagnostics_and_Laboratory_Medicine",
    "Economics",
    "Electronics",
    "Energy_and_Power",
    "Finance",
    "Geography",
    "History",
    "Literature",
    "Manage",
    "Marketing",
    "Materials",
    "Math",
    "Mechanical_Engineering",
    "Music",
    "Pharmacy",
    "Physics",
    "Psychology",
    "Public_Health",
    "Sociology",
]

DOMAIN_TO_SUBJECTS = {
    "Art and Design": ["Art", "Art_Theory", "Design", "Music"],
    "Business": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
    "Science": ["Biology", "Chemistry", "Geography", "Math", "Physics"],
    "Health and Medicine": [
        "Basic_Medical_Science",
        "Clinical_Medicine",
        "Diagnostics_and_Laboratory_Medicine",
        "Pharmacy",
        "Public_Health",
    ],
    "Humanities and Social Science": ["History", "Literature", "Sociology", "Psychology"],
    "Tech and Engineering": [
        "Agriculture",
        "Architecture_and_Engineering",
        "Computer_Science",
        "Electronics",
        "Energy_and_Power",
        "Materials",
        "Mechanical_Engineering",
    ],
}

SUBJECT_TO_DOMAIN = {
    subject: domain for domain, subjects in DOMAIN_TO_SUBJECTS.items() for subject in subjects
}

THINKING_BLOCK_RE = re.compile(
    r"<(?:think|thinking|reasoning)>.*?</(?:think|thinking|reasoning)>",
    flags=re.IGNORECASE | re.DOTALL,
)
OPEN_THINKING_BLOCK_RE = re.compile(
    r"<(?:think|thinking|reasoning)>.*",
    flags=re.IGNORECASE | re.DOTALL,
)


@dataclass
class ScoreBucket:
    total: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.total if self.total else None


@dataclass(frozen=True)
class ScoredPrediction:
    parsed: Any
    correct: bool
    invalid: bool


class MMMURunner:
    def __init__(self, settings: MMMUSettings):
        self.settings = settings
        self.client = create_native_client(
            provider=settings.provider,
            base_url=settings.base_url,
            api_key=settings.api_key,
            api_key_env=settings.api_key_env,
            timeout_seconds=settings.timeout_seconds,
        )
        self._rng = random.Random(settings.seed if settings.seed is not None else 42)

    def planned_command(self, variant: VariantConfig) -> str:
        model = self._configured_model_name(variant)
        subjects = self._subjects()
        return (
            f"POST {self.client.planned_endpoint()} model={model} "
            f"dataset={self.settings.dataset_name} subjects={','.join(subjects)} "
            f"split={self.settings.split}"
        )

    def run(
        self,
        variant: VariantConfig,
        progress_callback: ProgressCallback | None = None,
    ) -> list[MetricResult]:
        model = self._model_name(variant)
        subjects = self._subjects()
        out_dir = Path(self.settings.output_dir) / _safe_name(variant.name)
        out_dir.mkdir(parents=True, exist_ok=True)
        samples_path = out_dir / "samples.jsonl"
        summary_path = out_dir / "summary.json"

        if progress_callback:
            progress_callback(
                "dataset_loading_started",
                {
                    "task": "mmmu",
                    "variant": variant.name,
                    "dataset": self.settings.dataset_name,
                    "languages": subjects,
                    "split": self.settings.split,
                    "dataset_revision": self.settings.dataset_revision,
                },
            )

        datasets = self._load_datasets(subjects)
        total_samples = self._planned_sample_count(datasets)
        if progress_callback:
            progress_callback(
                "dataset_loading_completed",
                {
                    "task": "mmmu",
                    "variant": variant.name,
                    "dataset": self.settings.dataset_name,
                    "languages": subjects,
                    "split": self.settings.split,
                    "total_samples": total_samples,
                },
            )
            progress_callback(
                "task_sample_plan",
                {
                    "task": "mmmu",
                    "variant": variant.name,
                    "total_samples": total_samples,
                    "languages": subjects,
                },
            )

        completed_samples = 0
        invalid_count = 0
        total_runtime = 0.0
        score_rows: list[dict[str, Any]] = []
        resume_records = (
            load_resume_records(samples_path, key=_mmmu_record_key)
            if self.settings.resume_samples
            else {}
        )
        resume_announced = False

        with samples_path.open(sample_file_mode(resume_records), encoding="utf-8") as sample_file:
            for subject, dataset in datasets:
                for row_payload in dataset:
                    if (
                        self.settings.sample_limit is not None
                        and completed_samples >= self.settings.sample_limit
                    ):
                        break
                    row = dict(row_payload)
                    options = _parse_options(row.get("options"))
                    prompt = render_prompt(self.settings, row, options)
                    images = _image_payloads(row, self.settings.image_format)
                    row_key = _mmmu_sample_key(
                        self.settings,
                        subject=subject,
                        row=row,
                        prompt=prompt,
                        image_count=len(images),
                    )
                    cached_record = resume_records.get(row_key)
                    if cached_record and record_matches(
                        cached_record,
                        {
                            "dataset": self.settings.dataset_name,
                            "dataset_revision": self.settings.dataset_revision,
                            "split": self.settings.split,
                            "subject": subject,
                            "sample_id": row.get("id"),
                            "question": row.get("question"),
                            "question_type": row.get("question_type"),
                            "prompt": prompt,
                            "image_count": len(images),
                            "evaluator": self.settings.evaluator,
                        },
                    ):
                        if (
                            cached_record.get("invalid")
                            and str(row.get("question_type") or "") == "multiple-choice"
                        ):
                            score_response(
                                row,
                                str(cached_record.get("parsed_output") or ""),
                                options,
                                self._rng,
                            )
                        invalid = bool(cached_record.get("invalid"))
                        correct = bool(cached_record.get("correct"))
                        invalid_count += int(invalid)
                        total_runtime += _as_float(cached_record.get("runtime_seconds")) or 0.0
                        completed_samples += 1
                        score_rows.append(
                            {
                                "subject": subject,
                                "domain": SUBJECT_TO_DOMAIN.get(subject, "Other"),
                                "question_type": row.get("question_type"),
                                "correct": correct,
                                "invalid": invalid,
                            }
                        )
                        continue
                    resume_announced = maybe_announce_resume(
                        progress_callback=progress_callback,
                        announced=resume_announced,
                        task="mmmu",
                        variant=variant.name,
                        completed_samples=completed_samples,
                        total_samples=total_samples,
                        correct_samples=sum(int(row["correct"]) for row in score_rows),
                        invalid_samples=invalid_count,
                        runtime_seconds=total_runtime,
                    )
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
                        images=images,
                    )
                    total_runtime += response.runtime_seconds
                    raw_output = response.text
                    parsed_output = (
                        _strip_reasoning(raw_output)
                        if self.settings.strip_thinking
                        else raw_output
                    )
                    scored = score_response(row, parsed_output, options, self._rng)
                    invalid_count += int(scored.invalid)
                    completed_samples += 1
                    score_rows.append(
                        {
                            "subject": subject,
                            "domain": SUBJECT_TO_DOMAIN.get(subject, "Other"),
                            "question_type": row.get("question_type"),
                            "correct": scored.correct,
                            "invalid": scored.invalid,
                        }
                    )

                    sample_record = {
                        "dataset": self.settings.dataset_name,
                        "dataset_revision": self.settings.dataset_revision,
                        "split": self.settings.split,
                        "subject": subject,
                        "domain": SUBJECT_TO_DOMAIN.get(subject),
                        "sample_id": row.get("id"),
                        "question": row.get("question"),
                        "question_type": row.get("question_type"),
                        "options": options,
                        "answer": row.get("answer"),
                        "subfield": row.get("subfield"),
                        "topic_difficulty": row.get("topic_difficulty"),
                        "img_type": row.get("img_type"),
                        "image_count": len(images),
                        "image_format": self.settings.image_format,
                        "prompt": prompt,
                        "raw_output": raw_output,
                        "parsed_output": parsed_output,
                        "parsed_prediction": scored.parsed,
                        "correct": scored.correct,
                        "invalid": scored.invalid,
                        "evaluator": self.settings.evaluator,
                        "runtime_seconds": response.runtime_seconds,
                        "usage": response.raw.get("usage"),
                        "raw_response": response.raw,
                    }
                    sample_file.write(
                        json.dumps(sample_record, ensure_ascii=False, default=str) + "\n"
                    )
                    sample_file.flush()

                    if progress_callback:
                        progress_callback(
                            "task_progress",
                            {
                                "task": "mmmu",
                                "variant": variant.name,
                                "language": subject,
                                "completed_samples": completed_samples,
                                "total_samples": total_samples,
                                "latest_score": 1.0 if scored.correct else 0.0,
                                "latest_correct": scored.correct,
                                "latest_extracted_answer": _display_prediction(scored.parsed),
                                "latest_runtime_seconds": response.runtime_seconds,
                                "latest_subject": row.get("subfield") or subject,
                            },
                        )
                    maybe_reset_runtime(
                        client=self.client,
                        settings=self.settings,
                        model=model,
                        task="mmmu",
                        variant_name=variant.name,
                        completed_samples=completed_samples,
                        total_samples=total_samples,
                        progress_callback=progress_callback,
                        language=subject,
                    )
                if (
                    self.settings.sample_limit is not None
                    and completed_samples >= self.settings.sample_limit
                ):
                    break

        summary = build_summary(
            settings=self.settings,
            model=model,
            subjects=subjects,
            score_rows=score_rows,
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

    def _subjects(self) -> list[str]:
        return self.settings.subjects or MMMU_SUBJECTS

    def _load_datasets(self, subjects: list[str]) -> list[tuple[str, Any]]:
        try:
            from datasets import load_dataset
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "MMMU requires the `datasets` package. Install with `pip install -e '.[eval]'`."
            ) from exc

        datasets: list[tuple[str, Any]] = []
        for subject in subjects:
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                dataset = load_dataset(
                    self.settings.dataset_name,
                    subject,
                    split=self.settings.split,
                    revision=self.settings.dataset_revision,
                    download_mode="reuse_dataset_if_exists",
                )
            datasets.append((subject, dataset))
        return datasets

    def _planned_sample_count(self, datasets: list[tuple[str, Any]]) -> int:
        total = sum(len(dataset) for _, dataset in datasets)
        if self.settings.sample_limit is None:
            return total
        return min(total, self.settings.sample_limit)


def _mmmu_sample_key(
    settings: MMMUSettings,
    *,
    subject: str,
    row: dict[str, Any],
    prompt: str,
    image_count: int,
) -> str:
    return cache_key(
        settings.dataset_name,
        settings.dataset_revision,
        settings.split,
        subject,
        row.get("id"),
        row.get("question"),
        row.get("question_type"),
        prompt,
        image_count,
        settings.evaluator,
    )


def _mmmu_record_key(record: dict[str, Any]) -> str | None:
    return cache_key(
        record.get("dataset"),
        record.get("dataset_revision"),
        record.get("split"),
        record.get("subject"),
        record.get("sample_id"),
        record.get("question"),
        record.get("question_type"),
        record.get("prompt"),
        record.get("image_count"),
        record.get("evaluator"),
    )
def render_prompt(settings: MMMUSettings, row: dict[str, Any], options: list[str]) -> str:
    question = str(row.get("question") or "")
    if _image_payloads_available(row):
        question = (
            "The images are attached in order. References like <image 1> correspond "
            f"to the attached images.\n\n{question}"
        )
    if str(row.get("question_type") or "") == "multiple-choice":
        rendered_options = _render_options(options)
        return settings.multiple_choice_prompt_template.format(
            question=question,
            options=rendered_options,
            subject=_subject_from_id(row.get("id")),
        )
    return settings.short_answer_prompt_template.format(
        question=question,
        subject=_subject_from_id(row.get("id")),
    )


def build_summary(
    *,
    settings: MMMUSettings,
    model: str,
    subjects: list[str],
    score_rows: list[dict[str, Any]],
    invalid_count: int,
    total_runtime: float,
    samples_path: Path,
) -> dict[str, Any]:
    total = len(score_rows)
    correct = sum(int(row["correct"]) for row in score_rows)
    question_type_scores = _bucket_by(score_rows, "question_type")
    subject_scores = _bucket_by(score_rows, "subject")
    domain_scores = _bucket_by(score_rows, "domain")
    return {
        "task": "mmmu",
        "dataset": settings.dataset_name,
        "dataset_revision": settings.dataset_revision,
        "split": settings.split,
        "subjects": subjects,
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
        "correct": correct,
        "invalid": invalid_count,
        "mmmu_accuracy": correct / total if total else None,
        "mmmu_invalid_rate": invalid_count / total if total else None,
        "mmmu_multiple_choice_accuracy": question_type_scores.get("multiple-choice"),
        "mmmu_open_accuracy": _open_question_accuracy(question_type_scores),
        "subject_accuracy": subject_scores,
        "domain_accuracy": domain_scores,
        "question_type_accuracy": question_type_scores,
        "runtime_seconds": total_runtime,
        "samples_path": str(samples_path),
    }


def metrics_from_summary(summary: dict[str, Any]) -> list[MetricResult]:
    raw = {"summary": summary}
    metrics = [
        MetricResult("mmmu_accuracy", _as_float(summary.get("mmmu_accuracy")), "fraction", raw),
        MetricResult(
            "mmmu_invalid_rate",
            _as_float(summary.get("mmmu_invalid_rate")),
            "fraction",
            raw,
        ),
        MetricResult(
            "mmmu_multiple_choice_accuracy",
            _as_float(summary.get("mmmu_multiple_choice_accuracy")),
            "fraction",
            raw,
        ),
        MetricResult(
            "mmmu_open_accuracy",
            _as_float(summary.get("mmmu_open_accuracy")),
            "fraction",
            raw,
        ),
        MetricResult("mmmu_samples", _as_float(summary.get("total")), "samples", raw),
        MetricResult(
            "mmmu_runtime_seconds",
            _as_float(summary.get("runtime_seconds")),
            "seconds",
            raw,
        ),
    ]
    for domain, value in (summary.get("domain_accuracy") or {}).items():
        metrics.append(
            MetricResult(
                f"mmmu_domain_{_metric_slug(domain)}",
                _as_float(value),
                "fraction",
                raw,
            )
        )
    for subject, value in (summary.get("subject_accuracy") or {}).items():
        metrics.append(
            MetricResult(
                f"mmmu_subject_{_metric_slug(subject)}",
                _as_float(value),
                "fraction",
                raw,
            )
        )
    metrics.extend(efficiency_metrics_from_summary(summary))
    return metrics


def score_response(
    row: dict[str, Any],
    response: str,
    options: list[str],
    rng: random.Random | None = None,
) -> ScoredPrediction:
    if str(row.get("question_type") or "") == "multiple-choice":
        index2ans, all_choices = _multi_choice_info(options)
        parsed, invalid = parse_multi_choice_response(
            response,
            all_choices,
            index2ans,
            rng or random.Random(42),
        )
        return ScoredPrediction(
            parsed=parsed,
            correct=parsed == str(row.get("answer") or ""),
            invalid=invalid,
        )
    parsed = parse_open_response(response)
    return ScoredPrediction(
        parsed=parsed,
        correct=eval_open(row.get("answer"), parsed),
        invalid=not parsed,
    )


def parse_multi_choice_response(
    response: str,
    all_choices: list[str],
    index2ans: dict[str, str],
    rng: random.Random,
) -> tuple[str | None, bool]:
    candidate_response = response
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        candidate_response = candidate_response.strip(char)
    candidate_response = re.sub(r"\s+", " ", candidate_response)
    candidate_response = f" {candidate_response} "

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:
        if f"({choice})" in candidate_response:
            candidates.append(choice)
            ans_with_brack = True

    if not candidates:
        for choice in all_choices:
            if f" {choice} " in candidate_response:
                candidates.append(choice)

    if not candidates and len(candidate_response.split()) > 5:
        for index, answer in index2ans.items():
            if answer.lower() in candidate_response.lower():
                candidates.append(index)
                index_ans = False

    if not candidates:
        return (rng.choice(all_choices) if all_choices else None), True
    if len(candidates) > 1:
        start_indexes = []
        if index_ans:
            for candidate in candidates:
                needle = f"({candidate})" if ans_with_brack else f" {candidate} "
                start_indexes.append(candidate_response.rfind(needle))
        else:
            for candidate in candidates:
                start_indexes.append(candidate_response.lower().rfind(index2ans[candidate].lower()))
        return candidates[max(range(len(start_indexes)), key=start_indexes.__getitem__)], False
    return candidates[0], False


def parse_open_response(response: str) -> list[str | float]:
    key_responses = _key_subresponses(response)
    pred_list = key_responses.copy()
    for item in key_responses:
        pred_list.extend(_extract_numbers(item))

    normalized: list[str | float] = []
    for item in pred_list:
        normalized.extend(_normalize_str(item))
    return list(set(normalized))


def eval_open(gold: Any, predictions: list[str | float]) -> bool:
    if isinstance(gold, list):
        normalized_answers = []
        for answer in gold:
            normalized_answers.extend(_normalize_str(answer))
    else:
        normalized_answers = _normalize_str(gold)
    for prediction in predictions:
        if isinstance(prediction, str):
            for answer in normalized_answers:
                if isinstance(answer, str) and answer in prediction:
                    return True
        elif prediction in normalized_answers:
            return True
    return False


def _key_subresponses(response: str) -> list[str]:
    candidate = response.strip().strip(".").lower()
    sub_responses = re.split(r"\.\s(?=[A-Z])|\n", candidate)
    indicators = [
        "could be ",
        "so ",
        "is ",
        "thus ",
        "therefore ",
        "final ",
        "answer ",
        "result ",
    ]
    key_responses = []
    for index, item in enumerate(sub_responses):
        item_indicators = indicators + (["="] if index == len(sub_responses) - 1 else [])
        shortest = None
        for indicator in item_indicators:
            if indicator in item:
                trailing = item.split(indicator)[-1].strip()
                if shortest is None or len(trailing) < len(shortest):
                    shortest = trailing
        if shortest and shortest not in [":", ",", ".", "!", "?", ";", "'"]:
            key_responses.append(shortest)
    return key_responses or [candidate]


def _normalize_str(value: Any) -> list[str | float]:
    text = str(value).strip()
    if _is_number(text):
        return [round(float(text.replace(",", "")), 2)]
    text = text.lower()
    if len(text) == 1:
        return [f" {text}", f"{text} "]
    return [text]


def _is_number(value: str) -> bool:
    try:
        float(value.replace(",", ""))
        return True
    except ValueError:
        return False


def _extract_numbers(value: str) -> list[str]:
    pattern_commas = r"-?\b\d{1,3}(?:,\d{3})+\b"
    pattern_scientific = r"-?\d+(?:\.\d+)?[eE][+-]?\d+"
    pattern_simple = r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])"
    return (
        re.findall(pattern_commas, value)
        + re.findall(pattern_scientific, value)
        + re.findall(pattern_simple, value)
    )


def _image_payloads(row: dict[str, Any], image_format: str) -> list[ImagePayload]:
    payloads: list[ImagePayload] = []
    for index in range(1, 8):
        image = row.get(f"image_{index}")
        if image is None:
            continue
        payloads.append(_image_payload(image, image_format))
    return payloads


def _image_payload(image: Any, image_format: str) -> ImagePayload:
    if isinstance(image, bytes):
        return ImagePayload(data=base64.b64encode(image).decode("ascii"), mime_type="image/png")
    if not hasattr(image, "save"):
        raise RuntimeError(f"Unsupported MMMU image object: {type(image).__name__}")
    normalized_format = image_format.upper()
    mime_type = f"image/{'jpeg' if normalized_format == 'JPEG' else normalized_format.lower()}"
    image_to_save = image
    if normalized_format == "JPEG" and getattr(image, "mode", None) not in {"RGB", "L"}:
        image_to_save = image.convert("RGB")
    buffer = BytesIO()
    image_to_save.save(buffer, format=normalized_format)
    return ImagePayload(data=base64.b64encode(buffer.getvalue()).decode("ascii"), mime_type=mime_type)


def _image_payloads_available(row: dict[str, Any]) -> bool:
    return any(row.get(f"image_{index}") is not None for index in range(1, 8))


def _parse_options(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def _render_options(options: list[str]) -> str:
    index2ans, _ = _multi_choice_info(options)
    return "\n".join(f"({index}) {answer}" for index, answer in index2ans.items())


def _multi_choice_info(options: list[str]) -> tuple[dict[str, str], list[str]]:
    index2ans = {chr(ord("A") + index): option for index, option in enumerate(options)}
    return index2ans, list(index2ans)


def _strip_reasoning(text: str) -> str:
    candidate = THINKING_BLOCK_RE.sub("", text).strip()
    return OPEN_THINKING_BLOCK_RE.sub("", candidate).strip()


def _bucket_by(score_rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    buckets: dict[str, ScoreBucket] = {}
    for row in score_rows:
        name = str(row.get(key) or "unknown")
        bucket = buckets.setdefault(name, ScoreBucket())
        bucket.total += 1
        bucket.correct += int(bool(row.get("correct")))
    return {
        name: bucket.accuracy
        for name, bucket in sorted(buckets.items())
        if bucket.accuracy is not None
    }


def _open_question_accuracy(question_type_scores: dict[str, float]) -> float | None:
    values = [
        value
        for key, value in question_type_scores.items()
        if key != "multiple-choice" and value is not None
    ]
    return sum(values) / len(values) if values else None


def _display_prediction(value: Any) -> str:
    if value is None:
        return "?"
    if isinstance(value, list):
        if not value:
            return "?"
        text = ", ".join(str(item) for item in value[:2])
    else:
        text = str(value)
    return text[:24] if text else "?"


def _subject_from_id(value: Any) -> str:
    text = str(value or "")
    parts = text.split("_")
    return "_".join(parts[1:-1]) if len(parts) > 2 else ""


def _metric_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "unknown"


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "variant"
