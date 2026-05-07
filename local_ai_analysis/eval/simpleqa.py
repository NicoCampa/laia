from __future__ import annotations

import ast
import csv
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from local_ai_analysis.adapters.native import create_native_client
from local_ai_analysis.config import SimpleQASettings, VariantConfig
from local_ai_analysis.metrics import MetricResult


ProgressCallback = Callable[[str, dict[str, Any]], None]

SIMPLEQA_TEST_SET_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
)

GRADER_TEMPLATE = """
Your job is to look at a question, a gold target, and a predicted answer, and then
assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].

Grade as CORRECT when the predicted answer fully contains the important information
in the gold target and does not contradict it. Semantic meaning matters; capitalization,
punctuation, grammar, order, typos, and harmless extra detail should not matter.

Grade as INCORRECT when the predicted answer contradicts the gold target in any way,
even if the contradiction is hedged.

Grade as NOT_ATTEMPTED when the predicted answer does not include the important
information in the gold target and also does not contradict it.

For numeric targets, the predicted answer needs to be correct to the last significant
figure in the gold answer. The gold target may contain more detail than the question
requires; only require the information asked for by the question.

Here is the example to grade.

Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}

Grade the predicted answer as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Return only "A", "B", or "C".
""".strip()

THINKING_BLOCK_RE = re.compile(
    r"<(?:think|thinking|reasoning)>.*?</(?:think|thinking|reasoning)>",
    flags=re.IGNORECASE | re.DOTALL,
)
OPEN_THINKING_BLOCK_RE = re.compile(
    r"<(?:think|thinking|reasoning)>.*",
    flags=re.IGNORECASE | re.DOTALL,
)
GRADE_LETTER_RE = re.compile(r"\b([ABC])\b", flags=re.IGNORECASE)
REFUSAL_RE = re.compile(
    r"\b("
    r"i do not know|i don't know|i dont know|unknown|not sure|cannot answer|can't answer|"
    r"can not answer|unable to answer|do not have enough|don't have enough|insufficient"
    r")\b",
    flags=re.IGNORECASE,
)

GRADE_TO_LABEL = {
    "A": "correct",
    "B": "incorrect",
    "C": "not_attempted",
}


@dataclass(frozen=True)
class SimpleQAGrade:
    letter: str
    label: str
    raw_output: str
    runtime_seconds: float
    raw_response: dict[str, Any]

    @property
    def is_correct(self) -> bool:
        return self.letter == "A"

    @property
    def is_incorrect(self) -> bool:
        return self.letter == "B"

    @property
    def is_not_attempted(self) -> bool:
        return self.letter == "C"


class SimpleQARunner:
    def __init__(self, settings: SimpleQASettings):
        self.settings = settings
        self.client = create_native_client(
            provider=settings.provider,
            base_url=settings.base_url,
            api_key=settings.api_key,
            api_key_env=settings.api_key_env,
            timeout_seconds=settings.timeout_seconds,
        )
        grader_provider = settings.grader_provider or settings.provider
        grader_base_url = settings.grader_base_url or settings.base_url
        self.grader_client = create_native_client(
            provider=grader_provider,
            base_url=grader_base_url,
            api_key=settings.grader_api_key or settings.api_key,
            api_key_env=settings.grader_api_key_env or settings.api_key_env,
            timeout_seconds=settings.grader_timeout_seconds,
        )

    def planned_command(self, variant: VariantConfig) -> str:
        model = self._configured_model_name(variant)
        grader = self._configured_grader_model(model)
        return (
            f"POST {self.client.planned_endpoint()} model={model} "
            f"dataset={self.settings.dataset_name} grader={self.settings.grader} "
            f"grader_model={grader}"
        )

    def run(
        self,
        variant: VariantConfig,
        progress_callback: ProgressCallback | None = None,
    ) -> list[MetricResult]:
        model = self._model_name(variant)
        grader_model = self._grader_model_name(model)
        out_dir = Path(self.settings.output_dir) / _safe_name(variant.name)
        out_dir.mkdir(parents=True, exist_ok=True)
        samples_path = out_dir / "samples.jsonl"
        summary_path = out_dir / "summary.json"

        if progress_callback:
            progress_callback(
                "dataset_loading_started",
                {
                    "task": "simpleqa",
                    "variant": variant.name,
                    "dataset": self.settings.dataset_name,
                    "languages": ["en"],
                    "split": "test",
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
                    "task": "simpleqa",
                    "variant": variant.name,
                    "dataset": self.settings.dataset_name,
                    "languages": ["en"],
                    "split": "test",
                    "total_samples": total_samples,
                },
            )
            progress_callback(
                "task_sample_plan",
                {
                    "task": "simpleqa",
                    "variant": variant.name,
                    "total_samples": total_samples,
                    "languages": ["en"],
                },
            )

        total_runtime = 0.0
        grader_runtime = 0.0
        grade_counts = {"correct": 0, "incorrect": 0, "not_attempted": 0}

        with samples_path.open("w", encoding="utf-8") as sample_file:
            for index, row in enumerate(rows, start=1):
                question = str(row.get("problem") or "")
                answer = str(row.get("answer") or "")
                prompt = render_prompt(self.settings.prompt_template, question)
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
                grade = self._grade(
                    question=question,
                    target=answer,
                    predicted_answer=parsed_output,
                    grader_model=grader_model,
                )
                grader_runtime += grade.runtime_seconds
                grade_counts[grade.label] += 1

                sample_record = {
                    "dataset": self.settings.dataset_name,
                    "dataset_revision": self.settings.dataset_revision,
                    "sample_id": index - 1,
                    "metadata": _metadata(row.get("metadata")),
                    "question": question,
                    "answer": answer,
                    "prompt": prompt,
                    "raw_output": raw_output,
                    "parsed_output": parsed_output,
                    "grade_letter": grade.letter,
                    "grade": grade.label,
                    "grader": self.settings.grader,
                    "grader_model": grader_model,
                    "grader_raw_output": grade.raw_output,
                    "runtime_seconds": response.runtime_seconds,
                    "grader_runtime_seconds": grade.runtime_seconds,
                    "usage": response.raw.get("usage"),
                    "grader_usage": grade.raw_response.get("usage"),
                    "raw_response": response.raw,
                    "grader_raw_response": grade.raw_response,
                }
                sample_file.write(
                    json.dumps(sample_record, ensure_ascii=False, default=str) + "\n"
                )

                if progress_callback:
                    progress_callback(
                        "task_progress",
                        {
                            "task": "simpleqa",
                            "variant": variant.name,
                            "language": "en",
                            "completed_samples": index,
                            "total_samples": total_samples,
                            "latest_score": 1.0 if grade.is_correct else 0.0,
                            "latest_correct": grade.is_correct,
                            "latest_invalid": False,
                            "latest_extracted_answer": grade.label,
                            "latest_runtime_seconds": response.runtime_seconds
                            + grade.runtime_seconds,
                            "latest_subject": str(_metadata(row.get("metadata")).get("topic", "")),
                        },
                    )

        summary = build_summary(
            settings=self.settings,
            model=model,
            grader_model=grader_model,
            grade_counts=grade_counts,
            total_runtime=total_runtime,
            grader_runtime=grader_runtime,
            samples_path=samples_path,
        )
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, sort_keys=True, default=str)

        return metrics_from_summary(summary)

    def _grade(
        self,
        *,
        question: str,
        target: str,
        predicted_answer: str,
        grader_model: str,
    ) -> SimpleQAGrade:
        if self.settings.grader == "heuristic":
            return heuristic_grade(target=target, predicted_answer=predicted_answer)
        prompt = GRADER_TEMPLATE.format(
            question=question,
            target=target,
            predicted_answer=predicted_answer,
        )
        response = self.grader_client.generate(
            model=grader_model,
            prompt=prompt,
            temperature=self.settings.grader_temperature,
            max_tokens=self.settings.grader_max_tokens,
            top_p=self.settings.grader_top_p,
            stop=self.settings.grader_stop,
            seed=self.settings.grader_seed,
            reasoning_effort=self.settings.grader_reasoning_effort,
            response_format=self.settings.grader_response_format,
            request_extra=self.settings.grader_request_extra,
        )
        grade_text = _strip_reasoning(response.text)
        letter = parse_grade_letter(grade_text)
        return SimpleQAGrade(
            letter=letter,
            label=GRADE_TO_LABEL[letter],
            raw_output=response.text,
            runtime_seconds=response.runtime_seconds,
            raw_response=response.raw,
        )

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

    def _grader_model_name(self, model: str) -> str:
        configured = self._configured_grader_model(model)
        if self.settings.grader == "heuristic":
            return "heuristic"
        if configured.lower() not in {"auto", "@first", "first-loaded"}:
            self.grader_client.require_model(configured)
            return configured
        models = self.grader_client.list_models()
        if not models:
            raise RuntimeError(
                f"No grader models returned by {self.grader_client.models_endpoint()}; "
                "set `simpleqa.grader_model` explicitly in the config."
            )
        return models[0]

    def _configured_model_name(self, variant: VariantConfig) -> str:
        return variant.api_model or variant.model_repo or variant.name

    def _configured_grader_model(self, model: str) -> str:
        configured = (self.settings.grader_model or "same").strip()
        return model if configured.lower() in {"same", "@same"} else configured

    def _load_dataset(self) -> list[dict[str, Any]]:
        cache_dir = Path(self.settings.data_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "simple_qa_test_set.csv"
        if not cache_path.exists() or self.settings.refresh_cache:
            try:
                with urllib.request.urlopen(
                    self.settings.dataset_url,
                    timeout=self.settings.download_timeout_seconds,
                ) as response:
                    cache_path.write_bytes(response.read())
            except urllib.error.URLError as exc:
                raise RuntimeError(
                    f"Could not download SimpleQA dataset from {self.settings.dataset_url}: {exc}"
                ) from exc
        with cache_path.open("r", encoding="utf-8", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]


def render_prompt(template: str, question: str) -> str:
    return template.format(question=question)


def heuristic_grade(*, target: str, predicted_answer: str) -> SimpleQAGrade:
    target_text = _normalize_answer(target)
    predicted_text = _normalize_answer(predicted_answer)
    if target_text and target_text in predicted_text:
        letter = "A"
    elif not predicted_text or REFUSAL_RE.search(predicted_answer):
        letter = "C"
    else:
        letter = "B"
    return SimpleQAGrade(
        letter=letter,
        label=GRADE_TO_LABEL[letter],
        raw_output=f"heuristic:{letter}",
        runtime_seconds=0.0,
        raw_response={"grader": "heuristic", "letter": letter},
    )


def parse_grade_letter(text: str) -> str:
    normalized = text.strip().upper().replace("NOT ATTEMPTED", "NOT_ATTEMPTED")
    match = GRADE_LETTER_RE.search(normalized)
    if match:
        return match.group(1).upper()
    if "CORRECT" in normalized and "INCORRECT" not in normalized:
        return "A"
    if "INCORRECT" in normalized:
        return "B"
    if "NOT_ATTEMPTED" in normalized or "NOT ATTEMPTED" in normalized:
        return "C"
    return "C"


def build_summary(
    *,
    settings: SimpleQASettings,
    model: str,
    grader_model: str,
    grade_counts: dict[str, int],
    total_runtime: float,
    grader_runtime: float,
    samples_path: Path,
) -> dict[str, Any]:
    total = sum(grade_counts.values())
    correct = grade_counts["correct"]
    incorrect = grade_counts["incorrect"]
    not_attempted = grade_counts["not_attempted"]
    attempted = correct + incorrect
    correct_rate = correct / total if total else None
    incorrect_rate = incorrect / total if total else None
    not_attempted_rate = not_attempted / total if total else None
    attempted_rate = attempted / total if total else None
    accuracy_given_attempted = correct / attempted if attempted else 0.0
    f1 = (
        2 * accuracy_given_attempted * correct_rate / (accuracy_given_attempted + correct_rate)
        if correct_rate is not None and (accuracy_given_attempted + correct_rate) > 0
        else 0.0
    )
    return {
        "task": "simpleqa",
        "dataset": settings.dataset_name,
        "dataset_url": settings.dataset_url,
        "dataset_revision": settings.dataset_revision,
        "model": model,
        "grader": settings.grader,
        "grader_model": grader_model,
        "evaluator": settings.evaluator,
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "not_attempted": not_attempted,
        "attempted": attempted,
        "simpleqa_correct_rate": correct_rate,
        "simpleqa_incorrect_rate": incorrect_rate,
        "simpleqa_hallucination_rate": incorrect_rate,
        "simpleqa_not_attempted_rate": not_attempted_rate,
        "simpleqa_attempted_rate": attempted_rate,
        "simpleqa_accuracy_given_attempted": accuracy_given_attempted,
        "simpleqa_f1": f1,
        "simpleqa_samples": total,
        "simpleqa_runtime_seconds": total_runtime + grader_runtime,
        "simpleqa_answer_runtime_seconds": total_runtime,
        "simpleqa_grader_runtime_seconds": grader_runtime,
        "samples_path": str(samples_path),
    }


def metrics_from_summary(summary: dict[str, Any]) -> list[MetricResult]:
    raw = dict(summary)
    metrics = [
        ("simpleqa_f1", summary.get("simpleqa_f1"), "ratio"),
        ("simpleqa_correct_rate", summary.get("simpleqa_correct_rate"), "ratio"),
        ("simpleqa_incorrect_rate", summary.get("simpleqa_incorrect_rate"), "ratio"),
        ("simpleqa_hallucination_rate", summary.get("simpleqa_hallucination_rate"), "ratio"),
        (
            "simpleqa_not_attempted_rate",
            summary.get("simpleqa_not_attempted_rate"),
            "ratio",
        ),
        ("simpleqa_attempted_rate", summary.get("simpleqa_attempted_rate"), "ratio"),
        (
            "simpleqa_accuracy_given_attempted",
            summary.get("simpleqa_accuracy_given_attempted"),
            "ratio",
        ),
        ("simpleqa_samples", summary.get("simpleqa_samples"), "samples"),
        (
            "simpleqa_runtime_seconds",
            summary.get("simpleqa_runtime_seconds"),
            "seconds",
        ),
        (
            "simpleqa_answer_runtime_seconds",
            summary.get("simpleqa_answer_runtime_seconds"),
            "seconds",
        ),
        (
            "simpleqa_grader_runtime_seconds",
            summary.get("simpleqa_grader_runtime_seconds"),
            "seconds",
        ),
    ]
    return [
        MetricResult(name, _as_float(value), unit, raw)
        for name, value, unit in metrics
        if value is not None
    ]


def _metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return {"raw": str(value)}
    return parsed if isinstance(parsed, dict) else {"raw": str(value)}


def _normalize_answer(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _strip_reasoning(text: str) -> str:
    stripped = THINKING_BLOCK_RE.sub("", text)
    stripped = OPEN_THINKING_BLOCK_RE.sub("", stripped)
    return stripped.strip()


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip())
    return safe.strip("-") or "variant"
