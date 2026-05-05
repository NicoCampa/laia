from __future__ import annotations

import copy
import json
import re
import sys
import types
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Callable

from local_ai_analysis.adapters.native import GenerationResponse, create_native_client
from local_ai_analysis.config import BFCLV4Settings, VariantConfig
from local_ai_analysis.metrics import MetricResult


ProgressCallback = Callable[[str, dict[str, Any]], None]
BFCL_REGISTRY_NAME = "local-ai-analysis"


@dataclass
class CategoryScore:
    total: int = 0
    correct: int = 0
    invalid: int = 0

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.total if self.total else None

    @property
    def invalid_rate(self) -> float | None:
        return self.invalid / self.total if self.total else None


class BFCLV4Runner:
    def __init__(self, settings: BFCLV4Settings):
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
            f"benchmark={self.settings.version} categories={','.join(self.settings.categories)}"
        )

    def run(
        self,
        variant: VariantConfig,
        progress_callback: ProgressCallback | None = None,
    ) -> list[MetricResult]:
        modules = _load_bfcl_modules()
        model = self._model_name(variant)
        handler = _LocalPromptBFCLHandler(
            settings=self.settings,
            client=self.client,
            model=model,
            modules=modules,
        )

        out_dir = Path(self.settings.output_dir) / _safe_name(variant.name)
        out_dir.mkdir(parents=True, exist_ok=True)
        samples_path = out_dir / "samples.jsonl"
        summary_path = out_dir / "summary.json"

        if progress_callback:
            progress_callback(
                "dataset_loading_started",
                {
                    "task": "bfcl-v4",
                    "variant": variant.name,
                    "dataset": self.settings.version,
                    "languages": ["en"],
                    "split": "local_package",
                    "dataset_revision": None,
                },
            )
        category_entries = _load_category_entries(self.settings, modules)
        selected_entries = _limit_entries(category_entries, self.settings.sample_limit)
        total_samples = sum(len(entries) for entries in selected_entries.values())
        if progress_callback:
            progress_callback(
                "dataset_loading_completed",
                {
                    "task": "bfcl-v4",
                    "variant": variant.name,
                    "dataset": self.settings.version,
                    "languages": ["en"],
                    "split": "local_package",
                    "total_samples": total_samples,
                },
            )
            progress_callback(
                "task_sample_plan",
                {
                    "task": "bfcl-v4",
                    "variant": variant.name,
                    "total_samples": total_samples,
                    "languages": list(selected_entries),
                },
            )

        category_scores: dict[str, CategoryScore] = {
            category: CategoryScore() for category in selected_entries
        }
        total_runtime = 0.0
        completed = 0

        with samples_path.open("w", encoding="utf-8") as sample_file:
            for category, entries in selected_entries.items():
                ground_truth_entries = _ground_truth_by_id(category, modules)
                for entry in entries:
                    completed += 1
                    entry_copy = copy.deepcopy(entry)
                    inference_result, metadata, inference_error = _run_inference(
                        handler,
                        entry_copy,
                        include_input_log=self.settings.include_input_log,
                        exclude_state_log=self.settings.exclude_state_log,
                    )
                    runtime_seconds = _metadata_runtime(metadata)
                    total_runtime += runtime_seconds
                    score_payload = (
                        _score_entry(
                            category=category,
                            entry=entry,
                            model_result=inference_result,
                            ground_truth=ground_truth_entries.get(entry["id"]),
                            modules=modules,
                        )
                        if inference_error is None
                        else {
                            "valid": False,
                            "error": [inference_error],
                            "error_type": "inference_error",
                        }
                    )

                    valid = bool(score_payload.get("valid"))
                    category_score = category_scores[category]
                    category_score.total += 1
                    category_score.correct += int(valid)
                    category_score.invalid += int(_is_invalid_score(score_payload))

                    raw_output = _compact_result(inference_result)
                    sample_record = {
                        "benchmark": self.settings.version,
                        "category": category,
                        "sample_id": entry["id"],
                        "question": entry.get("question"),
                        "function": entry.get("function"),
                        "model_result": inference_result,
                        "raw_output": raw_output,
                        "valid": valid,
                        "error": score_payload.get("error"),
                        "error_type": score_payload.get("error_type"),
                        "runtime_seconds": runtime_seconds,
                        "usage": _metadata_usage(metadata),
                        "metadata": metadata,
                    }
                    sample_file.write(
                        json.dumps(sample_record, ensure_ascii=False, default=str) + "\n"
                    )

                    if progress_callback:
                        progress_callback(
                            "task_progress",
                            {
                                "task": "bfcl-v4",
                                "variant": variant.name,
                                "language": category,
                                "completed_samples": completed,
                                "total_samples": total_samples,
                                "latest_correct": valid,
                                "latest_extracted_answer": "pass" if valid else "fail",
                                "latest_runtime_seconds": runtime_seconds,
                                "latest_subject": category,
                            },
                        )

        summary = build_summary(
            settings=self.settings,
            model=model,
            category_scores=category_scores,
            total_runtime=total_runtime,
            samples_path=samples_path,
            modules=modules,
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


def build_summary(
    *,
    settings: BFCLV4Settings,
    model: str,
    category_scores: dict[str, CategoryScore],
    total_runtime: float,
    samples_path: Path,
    modules: dict[str, Any],
) -> dict[str, Any]:
    total = sum(score.total for score in category_scores.values())
    correct = sum(score.correct for score in category_scores.values())
    invalid = sum(score.invalid for score in category_scores.values())
    category_accuracy = {
        category: score.accuracy for category, score in sorted(category_scores.items())
    }
    category_counts = {
        category: {
            "total": score.total,
            "correct": score.correct,
            "invalid": score.invalid,
            "accuracy": score.accuracy,
            "invalid_rate": score.invalid_rate,
        }
        for category, score in sorted(category_scores.items())
    }
    group_accuracy = _group_accuracy(category_scores, modules)
    return {
        "task": "bfcl_v4_prompt_mode",
        "benchmark": settings.version,
        "model": model,
        "provider": settings.provider,
        "base_url": settings.base_url,
        "categories": settings.categories,
        "resolved_categories": list(category_scores),
        "sample_limit": settings.sample_limit,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "top_p": settings.top_p,
        "stop": settings.stop,
        "seed": settings.seed,
        "reasoning_effort": settings.reasoning_effort,
        "response_format": settings.response_format,
        "request_extra": settings.request_extra,
        "evaluator": settings.evaluator,
        "total": total,
        "correct": correct,
        "invalid": invalid,
        "bfcl_v4_selected_accuracy": correct / total if total else None,
        "bfcl_v4_invalid_rate": invalid / total if total else None,
        "bfcl_v4_non_live_accuracy": group_accuracy.get("non_live"),
        "bfcl_v4_live_accuracy": group_accuracy.get("live"),
        "bfcl_v4_multi_turn_accuracy": group_accuracy.get("multi_turn"),
        "bfcl_v4_agentic_accuracy": group_accuracy.get("agentic"),
        "category_accuracy": category_accuracy,
        "category_counts": category_counts,
        "runtime_seconds": total_runtime,
        "samples_path": str(samples_path),
    }


def metrics_from_summary(summary: dict[str, Any]) -> list[MetricResult]:
    raw = {"summary": summary}
    return [
        MetricResult(
            "bfcl_v4_selected_accuracy",
            _as_float(summary.get("bfcl_v4_selected_accuracy")),
            "fraction",
            raw,
        ),
        MetricResult(
            "bfcl_v4_invalid_rate",
            _as_float(summary.get("bfcl_v4_invalid_rate")),
            "fraction",
            raw,
        ),
        MetricResult(
            "bfcl_v4_non_live_accuracy",
            _as_float(summary.get("bfcl_v4_non_live_accuracy")),
            "fraction",
            raw,
        ),
        MetricResult(
            "bfcl_v4_live_accuracy",
            _as_float(summary.get("bfcl_v4_live_accuracy")),
            "fraction",
            raw,
        ),
        MetricResult(
            "bfcl_v4_multi_turn_accuracy",
            _as_float(summary.get("bfcl_v4_multi_turn_accuracy")),
            "fraction",
            raw,
        ),
        MetricResult(
            "bfcl_v4_agentic_accuracy",
            _as_float(summary.get("bfcl_v4_agentic_accuracy")),
            "fraction",
            raw,
        ),
        MetricResult("bfcl_v4_samples", _as_float(summary.get("total")), "count", raw),
        MetricResult(
            "bfcl_v4_runtime_seconds",
            _as_float(summary.get("runtime_seconds")),
            "seconds",
            raw,
        ),
    ]


class _LocalPromptBFCLHandler:
    def __init__(
        self,
        *,
        settings: BFCLV4Settings,
        client: Any,
        model: str,
        modules: dict[str, Any],
    ) -> None:
        base_handler = modules["BaseHandler"]
        self.__class__ = type(
            "_RuntimeLocalPromptBFCLHandler",
            (self.__class__, base_handler),
            {},
        )
        base_handler.__init__(
            self,
            model_name=BFCL_REGISTRY_NAME,
            temperature=settings.temperature,
            registry_name=BFCL_REGISTRY_NAME,
            is_fc_model=False,
        )
        self.settings = settings
        self.native_client = client
        self.native_model = model
        self.modules = modules

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions = copy.deepcopy(test_entry.get("function", []))
        test_entry["question"][0] = self.modules["system_prompt_pre_processing_chat_model"](
            test_entry["question"][0],
            functions,
            test_entry["id"],
        )
        return {"message": [], "function": functions}

    def _format_prompt(self, messages: list[dict], function: list[dict]) -> str:
        del function
        chunks: list[str] = []
        for message in messages:
            role = str(message.get("role") or "user").upper()
            content = message.get("content")
            if isinstance(content, list):
                content = json.dumps(content, ensure_ascii=False)
            chunks.append(f"{role}:\n{content or ''}")
        chunks.append("ASSISTANT:")
        return "\n\n".join(chunks)

    def _query_prompting(self, inference_data: dict) -> tuple[dict[str, Any], float]:
        prompt = self._format_prompt(inference_data["message"], inference_data["function"])
        inference_data["inference_input_log"] = {"formatted_prompt": prompt}
        response: GenerationResponse = self.native_client.generate(
            model=self.native_model,
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
        return {"response": response, "prompt": prompt}, response.runtime_seconds

    def _parse_query_response_prompting(self, api_response: dict[str, Any]) -> dict[str, Any]:
        response: GenerationResponse = api_response["response"]
        usage = response.raw.get("usage") or {}
        text = response.text
        if self.settings.strip_thinking:
            text = _strip_reasoning(text)
        reasoning_content = _reasoning_content(response.text)
        return {
            "model_responses": text,
            "reasoning_content": reasoning_content,
            "input_token": usage.get("prompt_tokens") or 0,
            "output_token": usage.get("completion_tokens") or usage.get("total_tokens") or 0,
        }

    def add_first_turn_message_prompting(
        self,
        inference_data: dict,
        first_turn_message: list[dict],
    ) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    def _add_next_turn_user_message_prompting(
        self,
        inference_data: dict,
        user_message: list[dict],
    ) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    def _add_assistant_message_prompting(
        self,
        inference_data: dict,
        model_response_data: dict,
    ) -> dict:
        inference_data["message"].append(
            {
                "role": "assistant",
                "content": model_response_data["model_responses"],
            }
        )
        return inference_data

    def _add_execution_results_prompting(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        for execution_result, decoded_model_response in zip(
            execution_results,
            model_response_data["model_responses_decoded"],
            strict=False,
        ):
            inference_data["message"].append(
                {
                    "role": "tool",
                    "name": decoded_model_response,
                    "content": execution_result,
                }
            )
        return inference_data

    def decode_ast(self, result: str, language: Any, has_tool_call_tag: bool) -> list[dict]:
        return self.modules["default_decode_ast_prompting"](
            result,
            language=language,
            has_tool_call_tag=has_tool_call_tag,
        )

    def decode_execute(self, result: str, has_tool_call_tag: bool) -> list[str]:
        return self.modules["default_decode_execute_prompting"](
            result,
            has_tool_call_tag=has_tool_call_tag,
        )


def _load_bfcl_modules() -> dict[str, Any]:
    try:
        _install_bfcl_model_config_stub()
        from bfcl_eval.constants.category_mapping import (
            AGENTIC_CATEGORY,
            LIVE_CATEGORY,
            MULTI_TURN_CATEGORY,
            NON_LIVE_CATEGORY,
        )
        from bfcl_eval.constants.enums import Language, ReturnFormat
        from bfcl_eval.eval_checker.agentic_eval.agentic_checker import agentic_checker
        from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
            is_empty_execute_response,
        )
        from bfcl_eval.model_handler.base_handler import BaseHandler
        from bfcl_eval.model_handler.utils import (
            default_decode_ast_prompting,
            default_decode_execute_prompting,
            parse_prompt_variation_params,
            system_prompt_pre_processing_chat_model,
        )
        from bfcl_eval.utils import (
            is_agentic,
            is_empty_output,
            is_format_sensitivity,
            is_function_calling_format_output,
            is_java,
            is_js,
            is_multi_turn,
            is_relevance_or_irrelevance,
            load_dataset_entry,
            load_ground_truth_entry,
            parse_test_category_argument,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "BFCL v4 requires Berkeley's `bfcl-eval` package and parser helpers. "
            "For this Python 3.13 environment use: "
            "`pip install --no-deps bfcl-eval==2026.3.23 overrides "
            "tree_sitter==0.21.3 tree-sitter-java==0.21.0 "
            "tree-sitter-javascript==0.21.4`."
        ) from exc

    return {
        "AGENTIC_CATEGORY": AGENTIC_CATEGORY,
        "LIVE_CATEGORY": LIVE_CATEGORY,
        "MULTI_TURN_CATEGORY": MULTI_TURN_CATEGORY,
        "NON_LIVE_CATEGORY": NON_LIVE_CATEGORY,
        "Language": Language,
        "ReturnFormat": ReturnFormat,
        "BaseHandler": BaseHandler,
        "agentic_checker": agentic_checker,
        "ast_checker": ast_checker,
        "multi_turn_checker": multi_turn_checker,
        "is_empty_execute_response": is_empty_execute_response,
        "default_decode_ast_prompting": default_decode_ast_prompting,
        "default_decode_execute_prompting": default_decode_execute_prompting,
        "system_prompt_pre_processing_chat_model": system_prompt_pre_processing_chat_model,
        "is_agentic": is_agentic,
        "is_empty_output": is_empty_output,
        "is_format_sensitivity": is_format_sensitivity,
        "is_function_calling_format_output": is_function_calling_format_output,
        "is_java": is_java,
        "is_js": is_js,
        "is_multi_turn": is_multi_turn,
        "is_relevance_or_irrelevance": is_relevance_or_irrelevance,
        "load_dataset_entry": load_dataset_entry,
        "load_ground_truth_entry": load_ground_truth_entry,
        "parse_prompt_variation_params": parse_prompt_variation_params,
        "parse_test_category_argument": parse_test_category_argument,
    }


def _install_bfcl_model_config_stub() -> None:
    if "bfcl_eval.constants.model_config" in sys.modules:
        return
    module = types.ModuleType("bfcl_eval.constants.model_config")

    class _LocalModelConfig:
        underscore_to_dot = False

    module.MODEL_CONFIG_MAPPING = {BFCL_REGISTRY_NAME: _LocalModelConfig()}
    sys.modules["bfcl_eval.constants.model_config"] = module


def _load_category_entries(
    settings: BFCLV4Settings,
    modules: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    categories = modules["parse_test_category_argument"](settings.categories)
    entries_by_category: dict[str, list[dict[str, Any]]] = {}
    for category in categories:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            entries = modules["load_dataset_entry"](
                category,
                include_prereq=False,
                include_language_specific_hint=True,
            )
        entries_by_category[category] = [dict(entry) for entry in entries]
    return entries_by_category


def _limit_entries(
    category_entries: dict[str, list[dict[str, Any]]],
    sample_limit: int | None,
) -> dict[str, list[dict[str, Any]]]:
    if sample_limit is None:
        return category_entries
    remaining = sample_limit
    limited: dict[str, list[dict[str, Any]]] = {}
    for category, entries in category_entries.items():
        if remaining <= 0:
            limited[category] = []
            continue
        limited[category] = entries[:remaining]
        remaining -= len(limited[category])
    return {category: entries for category, entries in limited.items() if entries}


def _ground_truth_by_id(category: str, modules: dict[str, Any]) -> dict[str, Any]:
    if modules["is_relevance_or_irrelevance"](category):
        return {}
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        entries = modules["load_ground_truth_entry"](category)
    return {str(entry.get("id")): entry.get("ground_truth") for entry in entries}


def _run_inference(
    handler: Any,
    entry: dict[str, Any],
    *,
    include_input_log: bool,
    exclude_state_log: bool,
) -> tuple[Any, dict[str, Any], str | None]:
    try:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            result, metadata = handler.inference(
                entry,
                include_input_log=include_input_log,
                exclude_state_log=exclude_state_log,
            )
        return result, metadata, None
    except Exception as exc:
        return None, {}, str(exc)


def _score_entry(
    *,
    category: str,
    entry: dict[str, Any],
    model_result: Any,
    ground_truth: Any,
    modules: dict[str, Any],
) -> dict[str, Any]:
    if modules["is_relevance_or_irrelevance"](category):
        return _score_relevance(category, model_result, modules)
    if modules["is_agentic"](category):
        return _score_agentic(category, entry, model_result, ground_truth, modules)
    if modules["is_multi_turn"](category):
        return _score_multi_turn(category, entry, model_result, ground_truth, modules)
    return _score_ast(category, entry, model_result, ground_truth, modules)


def _score_relevance(
    category: str,
    model_result: Any,
    modules: dict[str, Any],
) -> dict[str, Any]:
    ReturnFormat = modules["ReturnFormat"]
    try:
        decoded = modules["default_decode_ast_prompting"](
            str(model_result or ""),
            language=ReturnFormat.PYTHON,
            has_tool_call_tag=False,
        )
        contains_function_call = not modules["is_empty_output"](decoded)
        decode_error = None
    except Exception as exc:
        decoded = None
        contains_function_call = False
        decode_error = str(exc)

    valid = not contains_function_call if "irrelevance" in category else contains_function_call
    if valid:
        return {"valid": True}
    return {
        "valid": False,
        "error": [
            "Function call present when none should be emitted."
            if "irrelevance" in category
            else f"No valid function call emitted. {decode_error or ''}".strip()
        ],
        "error_type": "relevance_error",
        "model_result_decoded": decoded,
    }


def _score_ast(
    category: str,
    entry: dict[str, Any],
    model_result: Any,
    ground_truth: Any,
    modules: dict[str, Any],
) -> dict[str, Any]:
    Language = modules["Language"]
    ReturnFormat = modules["ReturnFormat"]
    return_format = ReturnFormat.PYTHON
    language = Language.PYTHON
    has_tool_call_tag = False
    effective_category = category
    if modules["is_format_sensitivity"](category):
        config = str(entry["id"]).split(":")[1]
        return_format_name, has_tool_call_tag, _, _, _ = modules[
            "parse_prompt_variation_params"
        ](config)
        return_format = ReturnFormat(return_format_name)
        effective_category = str(entry["id"]).split(":")[-1]
    elif modules["is_java"](category):
        language = Language.JAVA
        return_format = ReturnFormat.JAVA
    elif modules["is_js"](category):
        language = Language.JAVASCRIPT
        return_format = ReturnFormat.JAVASCRIPT

    try:
        decoded = modules["default_decode_ast_prompting"](
            str(model_result or ""),
            language=return_format,
            has_tool_call_tag=has_tool_call_tag,
        )
    except Exception as exc:
        return {
            "valid": False,
            "error": [f"Invalid syntax. Failed to decode AST. {exc}"],
            "error_type": "ast_decoder:decoder_failed",
        }
    if not modules["is_function_calling_format_output"](decoded):
        return {
            "valid": False,
            "error": ["Model did not output BFCL function-calling format."],
            "error_type": "ast_decoder:decoder_wrong_output_format",
            "model_result_decoded": decoded,
        }

    result = modules["ast_checker"](
        entry["function"],
        decoded,
        ground_truth,
        language,
        effective_category,
        BFCL_REGISTRY_NAME,
    )
    return result if not result.get("valid") else {"valid": True}


def _score_multi_turn(
    category: str,
    entry: dict[str, Any],
    model_result: Any,
    ground_truth: Any,
    modules: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(model_result, list):
        return {"valid": False, "error": ["Model result is not a list."], "error_type": "multi_turn"}
    decoded_turns: list[list[list[str]]] = []
    for turn in model_result:
        decoded_steps: list[list[str]] = []
        for item in turn if isinstance(turn, list) else []:
            try:
                decoded = modules["default_decode_execute_prompting"](
                    str(item),
                    has_tool_call_tag=False,
                )
                if not modules["is_empty_execute_response"](decoded):
                    decoded_steps.append(decoded)
            except Exception:
                continue
        decoded_turns.append(decoded_steps)
    result = modules["multi_turn_checker"](
        decoded_turns,
        ground_truth,
        entry,
        category,
        BFCL_REGISTRY_NAME,
    )
    return result if not result.get("valid") else {"valid": True}


def _score_agentic(
    category: str,
    entry: dict[str, Any],
    model_result: Any,
    ground_truth: Any,
    modules: dict[str, Any],
) -> dict[str, Any]:
    del category, entry
    if not isinstance(model_result, list) or len(model_result) != 1:
        return {
            "valid": False,
            "error": ["Agentic result should be a list containing one conversation."],
            "error_type": "agentic:inference_error",
        }
    last_non_function_message = None
    for item in model_result[0]:
        try:
            decoded = modules["default_decode_execute_prompting"](
                str(item),
                has_tool_call_tag=False,
            )
            if modules["is_empty_execute_response"](decoded):
                last_non_function_message = item
        except Exception:
            last_non_function_message = item
    if last_non_function_message is None:
        return {
            "valid": False,
            "error": ["Could not find final non-function-call message."],
            "error_type": "agentic:no_last_message",
        }
    result = modules["agentic_checker"](last_non_function_message, ground_truth)
    return result if not result.get("valid") else {"valid": True}


def _group_accuracy(
    category_scores: dict[str, CategoryScore],
    modules: dict[str, Any],
) -> dict[str, float | None]:
    groups = {
        "non_live": set(modules["NON_LIVE_CATEGORY"]),
        "live": set(modules["LIVE_CATEGORY"]),
        "multi_turn": set(modules["MULTI_TURN_CATEGORY"]),
        "agentic": set(modules["AGENTIC_CATEGORY"]),
    }
    result: dict[str, float | None] = {}
    for group_name, categories in groups.items():
        selected = [score for category, score in category_scores.items() if category in categories]
        total = sum(score.total for score in selected)
        correct = sum(score.correct for score in selected)
        result[group_name] = correct / total if total else None
    return result


def _metadata_runtime(metadata: dict[str, Any]) -> float:
    return _sum_numeric(metadata.get("latency"))


def _metadata_usage(metadata: dict[str, Any]) -> dict[str, Any]:
    prompt_tokens = _sum_numeric(metadata.get("input_token_count"))
    completion_tokens = _sum_numeric(metadata.get("output_token_count"))
    usage: dict[str, Any] = {}
    if prompt_tokens:
        usage["prompt_tokens"] = prompt_tokens
    if completion_tokens:
        usage["completion_tokens"] = completion_tokens
    if prompt_tokens or completion_tokens:
        usage["total_tokens"] = prompt_tokens + completion_tokens
    return usage


def _sum_numeric(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        return sum(_sum_numeric(item) for item in value)
    return 0.0


def _compact_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    if result is None:
        return ""
    return json.dumps(result, ensure_ascii=False, default=str)


def _is_invalid_score(score_payload: dict[str, Any]) -> bool:
    error_type = str(score_payload.get("error_type") or "")
    return "decoder" in error_type or "inference" in error_type


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


def _reasoning_content(text: str) -> str:
    match = re.search(
        r"<(?:think|thinking|reasoning)>(.*?)</(?:think|thinking|reasoning)>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _safe_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)
    return safe.strip("_") or "variant"
