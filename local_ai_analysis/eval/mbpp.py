from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Callable

from local_ai_analysis.adapters.native import create_native_client
from local_ai_analysis.config import MBPPSettings, VariantConfig
from local_ai_analysis.eval.efficiency import efficiency_metrics_from_summary
from local_ai_analysis.eval.runtime import maybe_reset_runtime
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
CODE_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*\n(?P<code>.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)
BEGIN_DONE_RE = re.compile(
    r"\[BEGIN\](?P<code>.*?)(?:\[DONE\]|$)",
    flags=re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class ExecutionResult:
    passed: bool
    error_type: str | None
    stdout: str
    stderr: str
    return_code: int | None
    runtime_seconds: float


class MBPPRunner:
    def __init__(self, settings: MBPPSettings):
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
            f"dataset={self.settings.dataset_name}/{self.settings.dataset_config} "
            f"split={self.settings.split}"
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
                    "task": "mbpp",
                    "variant": variant.name,
                    "dataset": self.settings.dataset_name,
                    "languages": [self.settings.dataset_config],
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
                    "task": "mbpp",
                    "variant": variant.name,
                    "dataset": self.settings.dataset_name,
                    "languages": [self.settings.dataset_config],
                    "split": self.settings.split,
                    "total_samples": total_samples,
                },
            )
            progress_callback(
                "task_sample_plan",
                {
                    "task": "mbpp",
                    "variant": variant.name,
                    "total_samples": total_samples,
                    "languages": [self.settings.dataset_config],
                },
            )

        total_runtime = 0.0
        score_rows: list[dict[str, Any]] = []

        with samples_path.open("w", encoding="utf-8") as sample_file:
            for index, row in enumerate(rows, start=1):
                prompt_text = _prompt_text(row)
                tests = _test_list(row, include_challenge=self.settings.include_challenge_tests)
                prompt = render_prompt(self.settings, prompt_text, tests)
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
                raw_output = response.text
                parsed_output = (
                    _strip_reasoning(raw_output) if self.settings.strip_thinking else raw_output
                )
                extracted_code = extract_code(parsed_output)
                execution = evaluate_code(
                    extracted_code,
                    tests=tests,
                    setup_code=_setup_code(row),
                    timeout_seconds=self.settings.execution_timeout_seconds,
                )
                total_runtime += response.runtime_seconds + execution.runtime_seconds
                invalid = execution.error_type in {"no_code", "syntax_error"}
                score_rows.append(
                    {
                        "task_id": row.get("task_id"),
                        "passed": execution.passed,
                        "invalid": invalid,
                        "error_type": execution.error_type,
                    }
                )

                sample_record = {
                    "dataset": self.settings.dataset_name,
                    "dataset_config": self.settings.dataset_config,
                    "dataset_revision": self.settings.dataset_revision,
                    "split": self.settings.split,
                    "task_id": row.get("task_id"),
                    "prompt_text": prompt_text,
                    "tests": tests,
                    "test_setup_code": row.get("test_setup_code"),
                    "test_imports": row.get("test_imports"),
                    "include_tests_in_prompt": self.settings.include_tests_in_prompt,
                    "include_challenge_tests": self.settings.include_challenge_tests,
                    "prompt": prompt,
                    "raw_output": raw_output,
                    "parsed_output": parsed_output,
                    "extracted_code": extracted_code,
                    "passed": execution.passed,
                    "invalid": invalid,
                    "error_type": execution.error_type,
                    "execution_stdout": execution.stdout,
                    "execution_stderr": execution.stderr,
                    "execution_return_code": execution.return_code,
                    "generation_runtime_seconds": response.runtime_seconds,
                    "execution_runtime_seconds": execution.runtime_seconds,
                    "runtime_seconds": response.runtime_seconds + execution.runtime_seconds,
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
                            "task": "mbpp",
                            "variant": variant.name,
                            "language": "python",
                            "completed_samples": index,
                            "total_samples": total_samples,
                            "latest_score": 1.0 if execution.passed else 0.0,
                            "latest_correct": execution.passed,
                            "latest_invalid": invalid,
                            "latest_extracted_answer": (
                                "pass" if execution.passed else execution.error_type or "fail"
                            ),
                            "latest_runtime_seconds": (
                                response.runtime_seconds + execution.runtime_seconds
                            ),
                            "latest_subject": str(row.get("task_id") or index),
                        },
                    )
                maybe_reset_runtime(
                    client=self.client,
                    settings=self.settings,
                    model=model,
                    task="mbpp",
                    variant_name=variant.name,
                    completed_samples=index,
                    total_samples=total_samples,
                    progress_callback=progress_callback,
                    language="python",
                )

        summary = build_summary(
            settings=self.settings,
            model=model,
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
        try:
            from datasets import load_dataset
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "MBPP requires the `datasets` package. Install with `pip install -e '.[eval]'`."
            ) from exc

        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            dataset = load_dataset(
                self.settings.dataset_name,
                self.settings.dataset_config,
                split=self.settings.split,
                revision=self.settings.dataset_revision,
                download_mode="reuse_dataset_if_exists",
            )
        return [dict(row) for row in dataset]


def render_prompt(settings: MBPPSettings, prompt_text: str, tests: list[str]) -> str:
    rendered_tests = "\n".join(tests) if settings.include_tests_in_prompt else "No tests shown."
    return settings.prompt_template.format(prompt=prompt_text, tests=rendered_tests)


def evaluate_code(
    code: str,
    *,
    tests: list[str],
    setup_code: str,
    timeout_seconds: float,
) -> ExecutionResult:
    normalized_code = code.strip()
    if not normalized_code:
        return ExecutionResult(
            passed=False,
            error_type="no_code",
            stdout="",
            stderr="",
            return_code=None,
            runtime_seconds=0.0,
        )

    script = build_execution_script(normalized_code, tests=tests, setup_code=setup_code)
    try:
        ast.parse(script)
    except SyntaxError as exc:
        return ExecutionResult(
            passed=False,
            error_type="syntax_error",
            stdout="",
            stderr=_truncate(f"{exc.__class__.__name__}: {exc}"),
            return_code=None,
            runtime_seconds=0.0,
        )

    with tempfile.TemporaryDirectory(prefix="laia-mbpp-") as tmpdir:
        script_path = Path(tmpdir) / "candidate.py"
        script_path.write_text(script, encoding="utf-8")
        start = time.perf_counter()
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(script_path)],
                cwd=tmpdir,
                input="",
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
                env=_execution_env(),
            )
        except subprocess.TimeoutExpired as exc:
            return ExecutionResult(
                passed=False,
                error_type="timeout",
                stdout=_truncate(exc.stdout or ""),
                stderr=_truncate(exc.stderr or ""),
                return_code=None,
                runtime_seconds=time.perf_counter() - start,
            )
        runtime = time.perf_counter() - start

    stderr = _truncate(completed.stderr or "")
    stdout = _truncate(completed.stdout or "")
    if completed.returncode == 0:
        return ExecutionResult(
            passed=True,
            error_type=None,
            stdout=stdout,
            stderr=stderr,
            return_code=completed.returncode,
            runtime_seconds=runtime,
        )
    return ExecutionResult(
        passed=False,
        error_type=_classify_failure(stderr),
        stdout=stdout,
        stderr=stderr,
        return_code=completed.returncode,
        runtime_seconds=runtime,
    )


def build_execution_script(code: str, *, tests: list[str], setup_code: str) -> str:
    sections = [
        "# Candidate solution",
        code,
        "# Test setup",
        setup_code.strip(),
        "# MBPP tests",
        "\n".join(test.strip() for test in tests if test.strip()),
        "",
    ]
    return "\n\n".join(section for section in sections if section is not None)


def extract_code(response: str) -> str:
    text = response.strip()
    if not text:
        return ""
    begin_match = BEGIN_DONE_RE.search(text)
    if begin_match:
        code = begin_match.group("code").strip()
        fence_match = CODE_FENCE_RE.search(code)
        return (fence_match.group("code") if fence_match else code).strip()
    fence_match = CODE_FENCE_RE.search(text)
    if fence_match:
        return fence_match.group("code").strip()
    if "[DONE]" in text:
        text = text.split("[DONE]", 1)[0].strip()
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if _looks_like_code_start(line):
            return "\n".join(lines[index:]).strip()
    return text


def build_summary(
    *,
    settings: MBPPSettings,
    model: str,
    score_rows: list[dict[str, Any]],
    total_runtime: float,
    samples_path: Path,
) -> dict[str, Any]:
    total = len(score_rows)
    passed = sum(int(row["passed"]) for row in score_rows)
    invalid = sum(int(row["invalid"]) for row in score_rows)
    syntax_errors = sum(int(row.get("error_type") == "syntax_error") for row in score_rows)
    no_code = sum(int(row.get("error_type") == "no_code") for row in score_rows)
    timeouts = sum(int(row.get("error_type") == "timeout") for row in score_rows)
    runtime_errors = sum(int(row.get("error_type") == "runtime_error") for row in score_rows)
    assertion_failures = sum(
        int(row.get("error_type") == "assertion_failure") for row in score_rows
    )
    compile_success = total - syntax_errors - no_code
    return {
        "task": "mbpp",
        "dataset": settings.dataset_name,
        "dataset_config": settings.dataset_config,
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
        "strip_thinking": settings.strip_thinking,
        "include_tests_in_prompt": settings.include_tests_in_prompt,
        "include_challenge_tests": settings.include_challenge_tests,
        "execution_timeout_seconds": settings.execution_timeout_seconds,
        "evaluator": settings.evaluator,
        "total": total,
        "passed": passed,
        "invalid": invalid,
        "syntax_errors": syntax_errors,
        "no_code": no_code,
        "timeouts": timeouts,
        "runtime_errors": runtime_errors,
        "assertion_failures": assertion_failures,
        "mbpp_pass_at_1": passed / total if total else None,
        "mbpp_invalid_rate": invalid / total if total else None,
        "mbpp_compile_rate": compile_success / total if total else None,
        "mbpp_runtime_error_rate": (runtime_errors + timeouts) / total if total else None,
        "mbpp_assertion_failure_rate": assertion_failures / total if total else None,
        "runtime_seconds": total_runtime,
        "samples_path": str(samples_path),
    }


def metrics_from_summary(summary: dict[str, Any]) -> list[MetricResult]:
    raw = {"summary": summary}
    metrics = [
        MetricResult("mbpp_pass_at_1", _as_float(summary.get("mbpp_pass_at_1")), "fraction", raw),
        MetricResult(
            "mbpp_invalid_rate",
            _as_float(summary.get("mbpp_invalid_rate")),
            "fraction",
            raw,
        ),
        MetricResult(
            "mbpp_compile_rate",
            _as_float(summary.get("mbpp_compile_rate")),
            "fraction",
            raw,
        ),
        MetricResult(
            "mbpp_runtime_error_rate",
            _as_float(summary.get("mbpp_runtime_error_rate")),
            "fraction",
            raw,
        ),
        MetricResult(
            "mbpp_assertion_failure_rate",
            _as_float(summary.get("mbpp_assertion_failure_rate")),
            "fraction",
            raw,
        ),
        MetricResult("mbpp_samples", _as_float(summary.get("total")), "samples", raw),
        MetricResult(
            "mbpp_runtime_seconds",
            _as_float(summary.get("runtime_seconds")),
            "seconds",
            raw,
        ),
    ]
    metrics.extend(efficiency_metrics_from_summary(summary))
    return metrics


def _prompt_text(row: dict[str, Any]) -> str:
    return str(row.get("prompt") or row.get("text") or "")


def _test_list(row: dict[str, Any], *, include_challenge: bool) -> list[str]:
    tests = [str(test) for test in (row.get("test_list") or []) if str(test).strip()]
    if include_challenge:
        tests.extend(
            str(test) for test in (row.get("challenge_test_list") or []) if str(test).strip()
        )
    return tests


def _setup_code(row: dict[str, Any]) -> str:
    imports = row.get("test_imports") or []
    setup_parts = [str(item) for item in imports if str(item).strip()]
    setup = str(row.get("test_setup_code") or "").strip()
    if setup:
        setup_parts.append(setup)
    return "\n".join(setup_parts)


def _strip_reasoning(text: str) -> str:
    stripped = THINKING_BLOCK_RE.sub("", text)
    stripped = OPEN_THINKING_BLOCK_RE.sub("", stripped)
    return stripped.strip()


def _looks_like_code_start(line: str) -> bool:
    stripped = line.strip()
    return bool(
        re.match(
            r"^(from\s+\w|import\s+\w|def\s+\w|class\s+\w|@|#|[A-Za-z_][A-Za-z0-9_]*\s*=)",
            stripped,
        )
    )


def _classify_failure(stderr: str) -> str:
    if "AssertionError" in stderr:
        return "assertion_failure"
    if "SyntaxError" in stderr or "IndentationError" in stderr:
        return "syntax_error"
    return "runtime_error"


def _execution_env() -> dict[str, str]:
    env = {
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
    }
    if "PATH" in os.environ:
        env["PATH"] = os.environ["PATH"]
    return env


def _truncate(value: str, limit: int = 4000) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 20] + "\n...[truncated]..."


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return safe or "variant"


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
