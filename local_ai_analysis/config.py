from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class RunSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = "local-ai-analysis-run"
    output_db: str = "results/local_ai_analysis.duckdb"
    raw_jsonl: str = "results/raw_results.jsonl"
    seed: int = 42
    notes: str | None = None


class GlobalMMLULiteSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    dataset_name: str = "CohereLabs/Global-MMLU-Lite"
    dataset_revision: str | None = None
    split: str = "test"
    languages: list[str] = Field(default_factory=lambda: ["en"])
    sample_limit_per_language: int | None = None
    output_dir: str = "results/global_mmlu_lite"
    provider: str = "ollama"
    base_url: str = "http://127.0.0.1:11434"
    api_key: str | None = None
    api_key_env: str | None = None
    timeout_seconds: int = 120
    temperature: float = 0.0
    max_tokens: int = 16
    top_p: float | None = None
    stop: list[str] | None = None
    seed: int | None = None
    reasoning_effort: str | None = None
    response_format: dict[str, Any] | None = None
    request_extra: dict[str, Any] = Field(default_factory=dict)
    strip_thinking: bool = True
    parser_version: str = "global_mmlu_lite_regex_v1"
    prompt_template: str = (
        "Answer the following multiple-choice question.\n\n"
        "Question:\n{question}\n\n"
        "A. {option_a}\n"
        "B. {option_b}\n"
        "C. {option_c}\n"
        "D. {option_d}\n\n"
        "Do not explain. Reply with only one letter: A, B, C, or D.\n"
        "Answer:"
    )


class IFBenchSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    dataset_name: str = "allenai/IFBench_test"
    dataset_revision: str | None = None
    split: str = "train"
    sample_limit: int | None = None
    output_dir: str = "results/ifbench"
    provider: str = "ollama"
    base_url: str = "http://127.0.0.1:11434"
    api_key: str | None = None
    api_key_env: str | None = None
    timeout_seconds: int = 120
    temperature: float = 0.01
    max_tokens: int = 4096
    top_p: float | None = 0.95
    stop: list[str] | None = None
    seed: int | None = None
    reasoning_effort: str | None = None
    response_format: dict[str, Any] | None = None
    request_extra: dict[str, Any] = Field(default_factory=dict)
    strip_thinking: bool = True
    evaluator: str = "allenai_ifbench_loose_v1"


class BFCLV4Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    version: str = "BFCL_v4"
    categories: list[str] = Field(default_factory=lambda: ["single_turn"])
    sample_limit: int | None = None
    output_dir: str = "results/bfcl_v4"
    provider: str = "ollama"
    base_url: str = "http://127.0.0.1:11434"
    api_key: str | None = None
    api_key_env: str | None = None
    timeout_seconds: int = 240
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float | None = None
    stop: list[str] | None = None
    seed: int | None = None
    reasoning_effort: str | None = None
    response_format: dict[str, Any] | None = None
    request_extra: dict[str, Any] = Field(default_factory=dict)
    strip_thinking: bool = True
    include_input_log: bool = False
    exclude_state_log: bool = True
    evaluator: str = "bfcl_eval_prompt_mode_v4"


class BackendSettings(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = "Ollama"
    backend_type: str = "ollama"
    version: str | None = None
    commit: str | None = None
    command: str | None = None


class VariantConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    quantization: str = "SERVER"
    precision: str | None = None
    model_repo: str | None = None
    baseline: bool = False
    api_model: str | None = None


class BaseModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    family: str
    name: str
    parameter_size_b: float
    architecture: str | None = None
    license: str | None = None
    source_url: str | None = None
    variants: list[VariantConfig]


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    project: str = "Local AI Analysis"
    run: RunSettings = Field(default_factory=RunSettings)
    backend: BackendSettings = Field(default_factory=BackendSettings)
    global_mmlu_lite: GlobalMMLULiteSettings = Field(default_factory=GlobalMMLULiteSettings)
    ifbench: IFBenchSettings = Field(default_factory=IFBenchSettings)
    bfcl_v4: BFCLV4Settings = Field(default_factory=BFCLV4Settings)
    models: list[BaseModelConfig]


def load_config(path: str | Path) -> BenchmarkConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        payload: dict[str, Any] = yaml.safe_load(f) or {}
    return BenchmarkConfig.model_validate(payload)
