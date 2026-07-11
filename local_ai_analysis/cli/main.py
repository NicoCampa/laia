from __future__ import annotations

import json
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from local_ai_analysis import __version__
from local_ai_analysis.config import load_config
from local_ai_analysis.db import LocalAIAnalysisDB
from local_ai_analysis.eval.mmmu import MMMU_SUBJECTS
from local_ai_analysis.eval.rgb import RGB_DATASETS
from local_ai_analysis.export import export_leaderboard, leaderboard_payload
from local_ai_analysis.normalization import refresh_normalized_results
from local_ai_analysis.runner import (
    BenchmarkRunError,
    run_benchmark,
    run_benchmark_with_progress,
)

app = typer.Typer(
    name="laia",
    help="Local AI Analysis: local API benchmarks for local model servers.",
    no_args_is_help=True,
)
console = Console()

GLOBAL_MMLU_LITE_REVISION = "cbf2f73663ff201d4d56e891c8c2c18467aeea06"
IFBENCH_REVISION = "2e8a48de45ff3bf41242f927254ca81b59ca3ae2"
OCRBENCH_V2_REVISION = "458b55b5f62bfd6eba7b5080da34fbc9a68c2626"
MMMU_REVISION = "4619a102cf5ad2da1abf7e220fde1258d2434cb7"
MBPP_REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
RGB_REVISION = "65ec39e40e7dc9abb50e9bf1b4f32be3f6f16615"
SIMPLE_EVALS_REVISION = "652c89d0ca9df547706735883097e9537d40dc47"
HARMBENCH_REVISION = "8e1604d1171fe8a48d8febecd22f600e462bdcdd"
GLOBAL_MMLU_LITE_LANGUAGES = [
    "ar",
    "bn",
    "cy",
    "de",
    "en",
    "es",
    "fr",
    "hi",
    "id",
    "it",
    "ja",
    "ko",
    "my",
    "pt",
    "sq",
    "sw",
    "yo",
    "zh",
]
GLOBAL_MMLU_LITE_PROMPT = """Answer the following multiple-choice question.

Question:
{question}

A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}

Do not explain. Do not use thinking tags. Reply with only one letter: A, B, C, or D.
Answer:
"""
BFCL_V4_DEFAULT_CATEGORIES = "single_turn"
BFCL_V4_DEFAULT_SAMPLE_LIMIT = 1000
OCRBENCH_V2_DEFAULT_CONFIGS = "EN,CN"
OCRBENCH_V2_SMOKE_CONFIGS = "text recognition en"
OCRBENCH_V2_DEFAULT_SAMPLE_LIMIT = 1000
OCRBENCH_V2_PROMPT = "{question}\nAnswer directly. Do not explain."
MMMU_DEFAULT_SUBJECTS = "all"
MMMU_SMOKE_SUBJECTS = "Accounting"
MMMU_DEFAULT_SPLIT = "validation"
MMMU_MULTI_CHOICE_PROMPT = (
    "{question}\n\n{options}\n\n"
    "Answer with the option's letter from the given choices directly."
)
MMMU_SHORT_ANSWER_PROMPT = "{question}\n\nAnswer the question using a single word or phrase."
MBPP_DEFAULT_CONFIG = "full"
MBPP_DEFAULT_SPLIT = "test"
MBPP_PROMPT = (
    "You are an expert Python programmer, and here is your task: {prompt}\n"
    "Your code should pass these tests:\n\n{tests}\n"
    "[BEGIN]\n"
)
RGB_DEFAULT_DATASET = "suite"
RGB_DEFAULT_SAMPLE_LIMIT = 100
RGB_DEFAULT_NOISE_RATE = 0.8
RGB_DEFAULT_PASSAGE_NUM = 5
RGB_DEFAULT_CORRECT_RATE = 0.0
SIMPLEQA_DATASET_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
)
SIMPLEQA_DEFAULT_SAMPLE_LIMIT = 500
SIMPLEQA_PROMPT = "{question}"
HARMBENCH_DATASET_URL = (
    f"https://raw.githubusercontent.com/centerforaisafety/HarmBench/{HARMBENCH_REVISION}/"
    "data/behavior_datasets/harmbench_behaviors_text_all.csv"
)
HARMBENCH_DEFAULT_CATEGORIES = "standard,contextual"
DEFAULT_CONTEXT_LENGTH = 8192
CORE_BENCHMARKS = ["global-mmlu-lite", "ifbench", "bfcl-v4", "mbpp", "rgb"]
MULTIMODAL_BENCHMARKS = ["ocrbench-v2", "mmmu"]
JUDGED_BENCHMARKS = ["simpleqa", "harmbench"]
ALL_NON_JUDGED_BENCHMARKS = [*CORE_BENCHMARKS, *MULTIMODAL_BENCHMARKS]
FULL_SUITE_BENCHMARKS = [*ALL_NON_JUDGED_BENCHMARKS, *JUDGED_BENCHMARKS]


@app.callback(invoke_without_command=True)
def callback(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show Local AI Analysis version and exit.",
            is_eager=True,
        ),
    ] = False,
) -> None:
    if version:
        console.print(f"local-ai-analysis {__version__}")
        raise typer.Exit()


@app.command()
def init_db(
    db: Annotated[
        Path,
        typer.Option("--db", help="DuckDB database path."),
    ] = Path("results/local_ai_analysis.duckdb"),
) -> None:
    """Create the benchmark database schema."""
    store = LocalAIAnalysisDB(db)
    try:
        store.init_schema()
    finally:
        store.close()
    console.print(f"[green]Initialized[/green] {db}")


@app.command()
def run(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="YAML benchmark config."),
    ],
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Record planned commands without running external tools."),
    ] = False,
    no_progress: Annotated[
        bool,
        typer.Option("--no-progress", help="Disable the live progress display."),
    ] = False,
    auto_export: Annotated[
        bool,
        typer.Option(
            "--auto-export/--no-auto-export",
            help="Export website results JSON after a successful benchmark run.",
        ),
    ] = True,
    auto_push: Annotated[
        bool,
        typer.Option(
            "--auto-push/--no-auto-push",
            help="Commit and push exported website result files after each benchmark is saved.",
        ),
    ] = True,
) -> None:
    """Run benchmark jobs from a YAML config file."""
    _execute_run(
        config,
        dry_run=dry_run,
        no_progress=no_progress,
        auto_export=auto_export,
        auto_push=auto_push,
    )


@app.command("ollama")
def run_ollama_shortcut(
    model: Annotated[
        str,
        typer.Argument(help="Ollama model tag, for example qwen3.5:0.8b-mlx-bf16."),
    ],
    smoke: Annotated[
        bool,
        typer.Option("--smoke", help="Run a 5-sample smoke set for the selected benchmark."),
    ] = False,
    languages: Annotated[
        str,
        typer.Option(
            "--languages",
            help="Comma-separated language codes, or 'all'. Ignored when --smoke is used.",
        ),
    ] = "all",
    benchmark: Annotated[
        str,
        typer.Option(
            "--benchmark",
            help=(
                "Benchmark or suite to run. Suites: text = text-only non-judge; "
                "vision = multimodal non-judge; judge = LLM-as-judge; "
                "suite = text + vision; recommended/full = everything. "
                "Individual names include global-mmlu-lite, ifbench, bfcl, ocrbench, "
                "mmmu, mbpp, rgb, simpleqa, and harmbench."
            ),
        ),
    ] = "global-mmlu-lite",
    bfcl_categories: Annotated[
        str,
        typer.Option(
            "--bfcl-categories",
            help=(
                "BFCL v4 category alias/list, for example single_turn, non_live, "
                "live, multi_turn, agentic, all_scoring."
            ),
        ),
    ] = BFCL_V4_DEFAULT_CATEGORIES,
    ocrbench_configs: Annotated[
        str,
        typer.Option(
            "--ocrbench-configs",
            help="OCRBench v2 dataset configs, for example EN,CN or text recognition en.",
        ),
    ] = OCRBENCH_V2_DEFAULT_CONFIGS,
    mmmu_subjects: Annotated[
        str,
        typer.Option(
            "--mmmu-subjects",
            help="MMMU subjects, for example all, Accounting,Math, or Computer_Science.",
        ),
    ] = MMMU_DEFAULT_SUBJECTS,
    mmmu_split: Annotated[
        str,
        typer.Option("--mmmu-split", help="MMMU split: dev, validation, or test."),
    ] = MMMU_DEFAULT_SPLIT,
    mbpp_config: Annotated[
        str,
        typer.Option("--mbpp-config", help="MBPP dataset config: full or sanitized."),
    ] = MBPP_DEFAULT_CONFIG,
    mbpp_split: Annotated[
        str,
        typer.Option("--mbpp-split", help="MBPP split: train, test, validation, or prompt."),
    ] = MBPP_DEFAULT_SPLIT,
    mbpp_challenge_tests: Annotated[
        bool,
        typer.Option(
            "--mbpp-challenge-tests/--no-mbpp-challenge-tests",
            help="Also run full MBPP challenge_test_list assertions when present.",
        ),
    ] = False,
    rgb_dataset: Annotated[
        str,
        typer.Option(
            "--rgb-dataset",
            help="RGB dataset: suite, en_refine, zh_refine, en_int, zh_int, en_fact, or zh_fact.",
        ),
    ] = RGB_DEFAULT_DATASET,
    rgb_noise_rate: Annotated[
        float,
        typer.Option("--rgb-noise-rate", min=0.0, max=1.0, help="RGB noisy passage rate."),
    ] = RGB_DEFAULT_NOISE_RATE,
    rgb_passage_num: Annotated[
        int,
        typer.Option("--rgb-passage-num", min=0, help="RGB number of supplied passages."),
    ] = RGB_DEFAULT_PASSAGE_NUM,
    rgb_correct_rate: Annotated[
        float,
        typer.Option(
            "--rgb-correct-rate",
            min=0.0,
            max=1.0,
            help="RGB correct-passage rate for counterfactual datasets.",
        ),
    ] = RGB_DEFAULT_CORRECT_RATE,
    simpleqa_grader: Annotated[
        str,
        typer.Option("--simpleqa-grader", help="SimpleQA grader: llm or heuristic."),
    ] = "llm",
    simpleqa_grader_model: Annotated[
        str,
        typer.Option(
            "--simpleqa-grader-model",
            help="SimpleQA judge model id. Use 'same' to grade with the tested model.",
        ),
    ] = "same",
    harmbench_categories: Annotated[
        str,
        typer.Option(
            "--harmbench-categories",
            help=(
                "HarmBench functional categories: standard, contextual, copyright, "
                "text, or all. Default excludes copyright."
            ),
        ),
    ] = HARMBENCH_DEFAULT_CATEGORIES,
    harmbench_judge: Annotated[
        str,
        typer.Option("--harmbench-judge", help="HarmBench judge: llm or heuristic."),
    ] = "llm",
    harmbench_judge_model: Annotated[
        str,
        typer.Option(
            "--harmbench-judge-model",
            help=(
                "HarmBench judge model id. Use 'same' to reuse the tested model; "
                "this mirrors the SimpleQA judge-model default."
            ),
        ),
    ] = "same",
    base_url: Annotated[
        str,
        typer.Option("--base-url", help="Ollama native API base URL."),
    ] = "http://127.0.0.1:11434",
    reasoning_effort: Annotated[
        str,
        typer.Option(
            "--reasoning-effort",
            help=(
                "Reasoning control sent to Ollama's native `think` parameter. "
                "Defaults to none. Use auto to disable it for Qwen and use high otherwise."
            ),
        ),
    ] = "none",
    modality: Annotated[
        str,
        typer.Option(
            "--modality",
            help=(
                "Model modality metadata: auto, text, vision, or multimodal. "
                "Auto marks runs with vision benchmarks as multimodal."
            ),
        ),
    ] = "auto",
    max_tokens: Annotated[
        int | None,
        typer.Option("--max-tokens", help="Override max generated tokens for the benchmark."),
    ] = None,
    resume_samples: Annotated[
        bool,
        typer.Option(
            "--resume-samples/--no-resume-samples",
            help="Resume benchmarks from existing matching samples.jsonl files.",
        ),
    ] = False,
    context_length: Annotated[
        int | None,
        typer.Option(
            "--context-length",
            min=1,
            help=(
                "Request context window for backends that support it. "
                "Defaults to 8192 for Ollama and LM Studio."
            ),
        ),
    ] = DEFAULT_CONTEXT_LENGTH,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Create the generated config and print planned work only."),
    ] = False,
    no_progress: Annotated[
        bool,
        typer.Option("--no-progress", help="Disable the live progress display."),
    ] = False,
    auto_export: Annotated[
        bool,
        typer.Option(
            "--auto-export/--no-auto-export",
            help="Export website results JSON after a successful benchmark run.",
        ),
    ] = True,
    auto_push: Annotated[
        bool,
        typer.Option(
            "--auto-push/--no-auto-push",
            help="Commit and push exported website result files after each benchmark is saved.",
        ),
    ] = True,
) -> None:
    """Run a benchmark against an Ollama model tag."""
    config = _write_api_benchmark_config(
        provider="Ollama",
        model=model,
        base_url=base_url,
        api_key_env=None,
        smoke=smoke,
        languages=languages,
        benchmark=benchmark,
        bfcl_categories=bfcl_categories,
        ocrbench_configs=ocrbench_configs,
        mmmu_subjects=mmmu_subjects,
        mmmu_split=mmmu_split,
        mbpp_config=mbpp_config,
        mbpp_split=mbpp_split,
        mbpp_challenge_tests=mbpp_challenge_tests,
        rgb_dataset=rgb_dataset,
        rgb_noise_rate=rgb_noise_rate,
        rgb_passage_num=rgb_passage_num,
        rgb_correct_rate=rgb_correct_rate,
        simpleqa_grader=simpleqa_grader,
        simpleqa_grader_model=simpleqa_grader_model,
        harmbench_categories=harmbench_categories,
        harmbench_judge=harmbench_judge,
        harmbench_judge_model=harmbench_judge_model,
        reasoning_effort=reasoning_effort,
        modality=modality,
        max_tokens=max_tokens,
        context_length=context_length,
        restart_between_languages=False,
        restart_every_calls=None,
        restart_cooldown_seconds=0.0,
        restart_cooldown_every_calls=None,
    )
    console.print(f"🧾 Generated config: {config}")
    _execute_run(
        config,
        dry_run=dry_run,
        no_progress=no_progress,
        auto_export=auto_export,
        auto_push=auto_push,
    )


@app.command("lmstudio")
def run_lmstudio_shortcut(
    model: Annotated[
        str,
        typer.Argument(
            help="LM Studio model id. Use the default 'auto' to benchmark the first served model.",
        ),
    ] = "auto",
    smoke: Annotated[
        bool,
        typer.Option("--smoke", help="Run a 5-sample smoke set for the selected benchmark."),
    ] = False,
    languages: Annotated[
        str,
        typer.Option(
            "--languages",
            help="Comma-separated language codes, or 'all'. Ignored when --smoke is used.",
        ),
    ] = "all",
    benchmark: Annotated[
        str,
        typer.Option(
            "--benchmark",
            help=(
                "Benchmark or suite to run. Suites: text = text-only non-judge; "
                "vision = multimodal non-judge; judge = LLM-as-judge; "
                "suite = text + vision; recommended/full = everything. "
                "Individual names include global-mmlu-lite, ifbench, bfcl, ocrbench, "
                "mmmu, mbpp, rgb, simpleqa, and harmbench."
            ),
        ),
    ] = "global-mmlu-lite",
    bfcl_categories: Annotated[
        str,
        typer.Option(
            "--bfcl-categories",
            help=(
                "BFCL v4 category alias/list, for example single_turn, non_live, "
                "live, multi_turn, agentic, all_scoring."
            ),
        ),
    ] = BFCL_V4_DEFAULT_CATEGORIES,
    ocrbench_configs: Annotated[
        str,
        typer.Option(
            "--ocrbench-configs",
            help="OCRBench v2 dataset configs, for example EN,CN or text recognition en.",
        ),
    ] = OCRBENCH_V2_DEFAULT_CONFIGS,
    mmmu_subjects: Annotated[
        str,
        typer.Option(
            "--mmmu-subjects",
            help="MMMU subjects, for example all, Accounting,Math, or Computer_Science.",
        ),
    ] = MMMU_DEFAULT_SUBJECTS,
    mmmu_split: Annotated[
        str,
        typer.Option("--mmmu-split", help="MMMU split: dev, validation, or test."),
    ] = MMMU_DEFAULT_SPLIT,
    mbpp_config: Annotated[
        str,
        typer.Option("--mbpp-config", help="MBPP dataset config: full or sanitized."),
    ] = MBPP_DEFAULT_CONFIG,
    mbpp_split: Annotated[
        str,
        typer.Option("--mbpp-split", help="MBPP split: train, test, validation, or prompt."),
    ] = MBPP_DEFAULT_SPLIT,
    mbpp_challenge_tests: Annotated[
        bool,
        typer.Option(
            "--mbpp-challenge-tests/--no-mbpp-challenge-tests",
            help="Also run full MBPP challenge_test_list assertions when present.",
        ),
    ] = False,
    rgb_dataset: Annotated[
        str,
        typer.Option(
            "--rgb-dataset",
            help="RGB dataset: suite, en_refine, zh_refine, en_int, zh_int, en_fact, or zh_fact.",
        ),
    ] = RGB_DEFAULT_DATASET,
    rgb_noise_rate: Annotated[
        float,
        typer.Option("--rgb-noise-rate", min=0.0, max=1.0, help="RGB noisy passage rate."),
    ] = RGB_DEFAULT_NOISE_RATE,
    rgb_passage_num: Annotated[
        int,
        typer.Option("--rgb-passage-num", min=0, help="RGB number of supplied passages."),
    ] = RGB_DEFAULT_PASSAGE_NUM,
    rgb_correct_rate: Annotated[
        float,
        typer.Option(
            "--rgb-correct-rate",
            min=0.0,
            max=1.0,
            help="RGB correct-passage rate for counterfactual datasets.",
        ),
    ] = RGB_DEFAULT_CORRECT_RATE,
    simpleqa_grader: Annotated[
        str,
        typer.Option("--simpleqa-grader", help="SimpleQA grader: llm or heuristic."),
    ] = "llm",
    simpleqa_grader_model: Annotated[
        str,
        typer.Option(
            "--simpleqa-grader-model",
            help="SimpleQA judge model id. Use 'same' to grade with the tested model.",
        ),
    ] = "same",
    harmbench_categories: Annotated[
        str,
        typer.Option(
            "--harmbench-categories",
            help=(
                "HarmBench functional categories: standard, contextual, copyright, "
                "text, or all. Default excludes copyright."
            ),
        ),
    ] = HARMBENCH_DEFAULT_CATEGORIES,
    harmbench_judge: Annotated[
        str,
        typer.Option("--harmbench-judge", help="HarmBench judge: llm or heuristic."),
    ] = "llm",
    harmbench_judge_model: Annotated[
        str,
        typer.Option(
            "--harmbench-judge-model",
            help=(
                "HarmBench judge model id. Use 'same' to reuse the tested model; "
                "this mirrors the SimpleQA judge-model default."
            ),
        ),
    ] = "same",
    base_url: Annotated[
        str,
        typer.Option("--base-url", help="LM Studio native API base URL."),
    ] = "http://127.0.0.1:1234",
    reasoning_effort: Annotated[
        str,
        typer.Option(
            "--reasoning-effort",
            help=(
                "Reasoning control sent to LM Studio's native `reasoning` parameter. "
                "Defaults to none. Use auto to disable it for Qwen model ids and use high otherwise."
            ),
        ),
    ] = "none",
    modality: Annotated[
        str,
        typer.Option(
            "--modality",
            help=(
                "Model modality metadata: auto, text, vision, or multimodal. "
                "Auto marks runs with vision benchmarks as multimodal."
            ),
        ),
    ] = "auto",
    max_tokens: Annotated[
        int | None,
        typer.Option("--max-tokens", help="Override max generated tokens for the benchmark."),
    ] = None,
    resume_samples: Annotated[
        bool,
        typer.Option(
            "--resume-samples/--no-resume-samples",
            help="Resume benchmarks from existing matching samples.jsonl files.",
        ),
    ] = False,
    context_length: Annotated[
        int | None,
        typer.Option(
            "--context-length",
            min=1,
            help=(
                "Request context window for backends that support it. "
                "Defaults to 8192 for Ollama and LM Studio."
            ),
        ),
    ] = DEFAULT_CONTEXT_LENGTH,
    restart_between_languages: Annotated[
        bool,
        typer.Option(
            "--restart-between-languages",
            help=(
                "For Global MMLU Lite, unload/reload the model between language "
                "groups to release runtime cache. Currently only LM Studio performs "
                "a real unload; other providers ignore it."
            ),
        ),
    ] = False,
    restart_every_calls: Annotated[
        int | None,
        typer.Option(
            "--restart-every-calls",
            min=1,
            help=(
                "For LM Studio runs, unload/reload the model every N completed "
                "sample calls to release runtime cache."
            ),
        ),
    ] = None,
    restart_cooldown_seconds: Annotated[
        float,
        typer.Option(
            "--restart-cooldown-seconds",
            min=0.0,
            help=(
                "For LM Studio runs, wait this many seconds after unloading "
                "the model and before loading it again."
            ),
        ),
    ] = 0.0,
    restart_cooldown_every_calls: Annotated[
        int | None,
        typer.Option(
            "--restart-cooldown-every-calls",
            min=1,
            help=(
                "For LM Studio runs, only apply --restart-cooldown-seconds after "
                "every N completed sample calls. Without this, every restart uses "
                "the cooldown."
            ),
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Create the generated config and print planned work only."),
    ] = False,
    no_progress: Annotated[
        bool,
        typer.Option("--no-progress", help="Disable the live progress display."),
    ] = False,
    auto_export: Annotated[
        bool,
        typer.Option(
            "--auto-export/--no-auto-export",
            help="Export website results JSON after a successful benchmark run.",
        ),
    ] = True,
    auto_push: Annotated[
        bool,
        typer.Option(
            "--auto-push/--no-auto-push",
            help="Commit and push exported website result files after each benchmark is saved.",
        ),
    ] = True,
) -> None:
    """Run a benchmark against the model served by LM Studio."""
    config = _write_api_benchmark_config(
        provider="LM Studio",
        model=model,
        base_url=base_url,
        api_key_env="LM_API_TOKEN",
        smoke=smoke,
        languages=languages,
        benchmark=benchmark,
        bfcl_categories=bfcl_categories,
        ocrbench_configs=ocrbench_configs,
        mmmu_subjects=mmmu_subjects,
        mmmu_split=mmmu_split,
        mbpp_config=mbpp_config,
        mbpp_split=mbpp_split,
        mbpp_challenge_tests=mbpp_challenge_tests,
        rgb_dataset=rgb_dataset,
        rgb_noise_rate=rgb_noise_rate,
        rgb_passage_num=rgb_passage_num,
        rgb_correct_rate=rgb_correct_rate,
        simpleqa_grader=simpleqa_grader,
        simpleqa_grader_model=simpleqa_grader_model,
        harmbench_categories=harmbench_categories,
        harmbench_judge=harmbench_judge,
        harmbench_judge_model=harmbench_judge_model,
        reasoning_effort=reasoning_effort,
        modality=modality,
        max_tokens=max_tokens,
        context_length=context_length,
        restart_between_languages=restart_between_languages,
        restart_every_calls=restart_every_calls,
        restart_cooldown_seconds=restart_cooldown_seconds,
        restart_cooldown_every_calls=restart_cooldown_every_calls,
        resume_samples=resume_samples,
    )
    console.print(f"🧾 Generated config: {config}")
    _execute_run(
        config,
        dry_run=dry_run,
        no_progress=no_progress,
        auto_export=auto_export,
        auto_push=auto_push,
    )


@app.command("omlx")
def run_omlx_shortcut(
    model: Annotated[
        str,
        typer.Argument(
            help="oMLX model id. Use the default 'auto' to benchmark the first discovered model.",
        ),
    ] = "auto",
    smoke: Annotated[
        bool,
        typer.Option("--smoke", help="Run a 5-sample smoke set for the selected benchmark."),
    ] = False,
    languages: Annotated[
        str,
        typer.Option(
            "--languages",
            help="Comma-separated language codes, or 'all'. Ignored when --smoke is used.",
        ),
    ] = "all",
    benchmark: Annotated[
        str,
        typer.Option(
            "--benchmark",
            help=(
                "Benchmark or suite to run. Suites: text = text-only non-judge; "
                "vision = multimodal non-judge; judge = LLM-as-judge; "
                "suite = text + vision; recommended/full = everything. "
                "Individual names include global-mmlu-lite, ifbench, bfcl, ocrbench, "
                "mmmu, mbpp, rgb, simpleqa, and harmbench."
            ),
        ),
    ] = "global-mmlu-lite",
    bfcl_categories: Annotated[
        str,
        typer.Option(
            "--bfcl-categories",
            help=(
                "BFCL v4 category alias/list, for example single_turn, non_live, "
                "live, multi_turn, agentic, all_scoring."
            ),
        ),
    ] = BFCL_V4_DEFAULT_CATEGORIES,
    ocrbench_configs: Annotated[
        str,
        typer.Option(
            "--ocrbench-configs",
            help="OCRBench v2 dataset configs, for example EN,CN or text recognition en.",
        ),
    ] = OCRBENCH_V2_DEFAULT_CONFIGS,
    mmmu_subjects: Annotated[
        str,
        typer.Option(
            "--mmmu-subjects",
            help="MMMU subjects, for example all, Accounting,Math, or Computer_Science.",
        ),
    ] = MMMU_DEFAULT_SUBJECTS,
    mmmu_split: Annotated[
        str,
        typer.Option("--mmmu-split", help="MMMU split: dev, validation, or test."),
    ] = MMMU_DEFAULT_SPLIT,
    mbpp_config: Annotated[
        str,
        typer.Option("--mbpp-config", help="MBPP dataset config: full or sanitized."),
    ] = MBPP_DEFAULT_CONFIG,
    mbpp_split: Annotated[
        str,
        typer.Option("--mbpp-split", help="MBPP split: train, test, validation, or prompt."),
    ] = MBPP_DEFAULT_SPLIT,
    mbpp_challenge_tests: Annotated[
        bool,
        typer.Option(
            "--mbpp-challenge-tests/--no-mbpp-challenge-tests",
            help="Also run full MBPP challenge_test_list assertions when present.",
        ),
    ] = False,
    rgb_dataset: Annotated[
        str,
        typer.Option(
            "--rgb-dataset",
            help="RGB dataset: suite, en_refine, zh_refine, en_int, zh_int, en_fact, or zh_fact.",
        ),
    ] = RGB_DEFAULT_DATASET,
    rgb_noise_rate: Annotated[
        float,
        typer.Option("--rgb-noise-rate", min=0.0, max=1.0, help="RGB noisy passage rate."),
    ] = RGB_DEFAULT_NOISE_RATE,
    rgb_passage_num: Annotated[
        int,
        typer.Option("--rgb-passage-num", min=0, help="RGB number of supplied passages."),
    ] = RGB_DEFAULT_PASSAGE_NUM,
    rgb_correct_rate: Annotated[
        float,
        typer.Option(
            "--rgb-correct-rate",
            min=0.0,
            max=1.0,
            help="RGB correct-passage rate for counterfactual datasets.",
        ),
    ] = RGB_DEFAULT_CORRECT_RATE,
    simpleqa_grader: Annotated[
        str,
        typer.Option("--simpleqa-grader", help="SimpleQA grader: llm or heuristic."),
    ] = "llm",
    simpleqa_grader_model: Annotated[
        str,
        typer.Option(
            "--simpleqa-grader-model",
            help="SimpleQA judge model id. Use 'same' to grade with the tested model.",
        ),
    ] = "same",
    harmbench_categories: Annotated[
        str,
        typer.Option(
            "--harmbench-categories",
            help=(
                "HarmBench functional categories: standard, contextual, copyright, "
                "text, or all. Default excludes copyright."
            ),
        ),
    ] = HARMBENCH_DEFAULT_CATEGORIES,
    harmbench_judge: Annotated[
        str,
        typer.Option("--harmbench-judge", help="HarmBench judge: llm or heuristic."),
    ] = "llm",
    harmbench_judge_model: Annotated[
        str,
        typer.Option(
            "--harmbench-judge-model",
            help=(
                "HarmBench judge model id. Use 'same' to reuse the tested model; "
                "this mirrors the SimpleQA judge-model default."
            ),
        ),
    ] = "same",
    base_url: Annotated[
        str,
        typer.Option("--base-url", help="oMLX native API base URL."),
    ] = "http://127.0.0.1:8000",
    api_key_env: Annotated[
        str,
        typer.Option(
            "--api-key-env",
            help="Environment variable containing the oMLX API key, if auth is enabled.",
        ),
    ] = "OMLX_API_KEY",
    reasoning_effort: Annotated[
        str,
        typer.Option(
            "--reasoning-effort",
            help=(
                "Reasoning control sent to oMLX chat_template_kwargs. Defaults to none, "
                "which sends enable_thinking=false."
            ),
        ),
    ] = "none",
    modality: Annotated[
        str,
        typer.Option(
            "--modality",
            help=(
                "Model modality metadata: auto, text, vision, or multimodal. "
                "Auto marks runs with vision benchmarks as multimodal."
            ),
        ),
    ] = "auto",
    max_tokens: Annotated[
        int | None,
        typer.Option("--max-tokens", help="Override max generated tokens for the benchmark."),
    ] = None,
    resume_samples: Annotated[
        bool,
        typer.Option(
            "--resume-samples/--no-resume-samples",
            help="Resume benchmarks from existing matching samples.jsonl files.",
        ),
    ] = False,
    context_length: Annotated[
        int | None,
        typer.Option(
            "--context-length",
            min=1,
            help=(
                "Request context window for backends that support it. "
                "Defaults to 8192 where supported."
            ),
        ),
    ] = DEFAULT_CONTEXT_LENGTH,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Create the generated config and print planned work only."),
    ] = False,
    no_progress: Annotated[
        bool,
        typer.Option("--no-progress", help="Disable the live progress display."),
    ] = False,
    auto_export: Annotated[
        bool,
        typer.Option(
            "--auto-export/--no-auto-export",
            help="Export website results JSON after a successful benchmark run.",
        ),
    ] = True,
    auto_push: Annotated[
        bool,
        typer.Option(
            "--auto-push/--no-auto-push",
            help="Commit and push exported website result files after each benchmark is saved.",
        ),
    ] = True,
) -> None:
    """Run a benchmark against a model served by oMLX."""
    config = _write_api_benchmark_config(
        provider="oMLX",
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
        smoke=smoke,
        languages=languages,
        benchmark=benchmark,
        bfcl_categories=bfcl_categories,
        ocrbench_configs=ocrbench_configs,
        mmmu_subjects=mmmu_subjects,
        mmmu_split=mmmu_split,
        mbpp_config=mbpp_config,
        mbpp_split=mbpp_split,
        mbpp_challenge_tests=mbpp_challenge_tests,
        rgb_dataset=rgb_dataset,
        rgb_noise_rate=rgb_noise_rate,
        rgb_passage_num=rgb_passage_num,
        rgb_correct_rate=rgb_correct_rate,
        simpleqa_grader=simpleqa_grader,
        simpleqa_grader_model=simpleqa_grader_model,
        harmbench_categories=harmbench_categories,
        harmbench_judge=harmbench_judge,
        harmbench_judge_model=harmbench_judge_model,
        reasoning_effort=reasoning_effort,
        modality=modality,
        max_tokens=max_tokens,
        context_length=context_length,
        restart_between_languages=False,
        restart_every_calls=None,
        restart_cooldown_seconds=0.0,
        restart_cooldown_every_calls=None,
        resume_samples=resume_samples,
    )
    console.print(f"🧾 Generated config: {config}")
    _execute_run(
        config,
        dry_run=dry_run,
        no_progress=no_progress,
        auto_export=auto_export,
        auto_push=auto_push,
    )


@app.command("openai")
def run_openai_shortcut(
    model: Annotated[
        str,
        typer.Argument(help="OpenAI model id, for example gpt-5.4-nano."),
    ],
    smoke: Annotated[
        bool,
        typer.Option("--smoke", help="Run a 5-sample smoke set for the selected benchmark."),
    ] = False,
    languages: Annotated[
        str,
        typer.Option(
            "--languages",
            help="Comma-separated language codes, or 'all'. Ignored when --smoke is used.",
        ),
    ] = "all",
    benchmark: Annotated[
        str,
        typer.Option(
            "--benchmark",
            help=(
                "Benchmark or suite to run. Suites: text = text-only non-judge; "
                "vision = multimodal non-judge; judge = LLM-as-judge; "
                "suite = text + vision; recommended/full = everything. "
                "Individual names include global-mmlu-lite, ifbench, bfcl, ocrbench, "
                "mmmu, mbpp, rgb, simpleqa, and harmbench."
            ),
        ),
    ] = "global-mmlu-lite",
    bfcl_categories: Annotated[
        str,
        typer.Option(
            "--bfcl-categories",
            help=(
                "BFCL v4 category alias/list, for example single_turn, non_live, "
                "live, multi_turn, agentic, all_scoring."
            ),
        ),
    ] = BFCL_V4_DEFAULT_CATEGORIES,
    ocrbench_configs: Annotated[
        str,
        typer.Option(
            "--ocrbench-configs",
            help="OCRBench v2 dataset configs, for example EN,CN or text recognition en.",
        ),
    ] = OCRBENCH_V2_DEFAULT_CONFIGS,
    mmmu_subjects: Annotated[
        str,
        typer.Option(
            "--mmmu-subjects",
            help="MMMU subjects, for example all, Accounting,Math, or Computer_Science.",
        ),
    ] = MMMU_DEFAULT_SUBJECTS,
    mmmu_split: Annotated[
        str,
        typer.Option("--mmmu-split", help="MMMU split: dev, validation, or test."),
    ] = MMMU_DEFAULT_SPLIT,
    mbpp_config: Annotated[
        str,
        typer.Option("--mbpp-config", help="MBPP dataset config: full or sanitized."),
    ] = MBPP_DEFAULT_CONFIG,
    mbpp_split: Annotated[
        str,
        typer.Option("--mbpp-split", help="MBPP split: train, test, validation, or prompt."),
    ] = MBPP_DEFAULT_SPLIT,
    mbpp_challenge_tests: Annotated[
        bool,
        typer.Option(
            "--mbpp-challenge-tests/--no-mbpp-challenge-tests",
            help="Also run full MBPP challenge_test_list assertions when present.",
        ),
    ] = False,
    rgb_dataset: Annotated[
        str,
        typer.Option(
            "--rgb-dataset",
            help="RGB dataset: suite, en_refine, zh_refine, en_int, zh_int, en_fact, or zh_fact.",
        ),
    ] = RGB_DEFAULT_DATASET,
    rgb_noise_rate: Annotated[
        float,
        typer.Option("--rgb-noise-rate", min=0.0, max=1.0, help="RGB noisy passage rate."),
    ] = RGB_DEFAULT_NOISE_RATE,
    rgb_passage_num: Annotated[
        int,
        typer.Option("--rgb-passage-num", min=0, help="RGB number of supplied passages."),
    ] = RGB_DEFAULT_PASSAGE_NUM,
    rgb_correct_rate: Annotated[
        float,
        typer.Option(
            "--rgb-correct-rate",
            min=0.0,
            max=1.0,
            help="RGB correct-passage rate for counterfactual datasets.",
        ),
    ] = RGB_DEFAULT_CORRECT_RATE,
    simpleqa_grader: Annotated[
        str,
        typer.Option("--simpleqa-grader", help="SimpleQA grader: llm or heuristic."),
    ] = "llm",
    simpleqa_grader_model: Annotated[
        str,
        typer.Option(
            "--simpleqa-grader-model",
            help="SimpleQA judge model id. Use 'same' to grade with the tested model.",
        ),
    ] = "same",
    harmbench_categories: Annotated[
        str,
        typer.Option(
            "--harmbench-categories",
            help=(
                "HarmBench functional categories: standard, contextual, copyright, "
                "text, or all. Default excludes copyright."
            ),
        ),
    ] = HARMBENCH_DEFAULT_CATEGORIES,
    harmbench_judge: Annotated[
        str,
        typer.Option("--harmbench-judge", help="HarmBench judge: llm or heuristic."),
    ] = "llm",
    harmbench_judge_model: Annotated[
        str,
        typer.Option(
            "--harmbench-judge-model",
            help=(
                "HarmBench judge model id. Use 'same' to reuse the tested model; "
                "this mirrors the SimpleQA judge-model default."
            ),
        ),
    ] = "same",
    base_url: Annotated[
        str,
        typer.Option("--base-url", help="OpenAI API base URL."),
    ] = "https://api.openai.com/v1",
    api_key_env: Annotated[
        str,
        typer.Option("--api-key-env", help="Environment variable containing the OpenAI API key."),
    ] = "OPENAI_API_KEY",
    reasoning_effort: Annotated[
        str,
        typer.Option(
            "--reasoning-effort",
            help=(
                "OpenAI reasoning_effort. Defaults to auto: none for GPT-5.4/GPT-5.1, "
                "minimal for older GPT-5 reasoning models."
            ),
        ),
    ] = "auto",
    modality: Annotated[
        str,
        typer.Option(
            "--modality",
            help=(
                "Model modality metadata: auto, text, vision, or multimodal. "
                "Auto marks runs with vision benchmarks as multimodal."
            ),
        ),
    ] = "auto",
    max_tokens: Annotated[
        int | None,
        typer.Option("--max-tokens", help="Override max generated tokens for the benchmark."),
    ] = None,
    resume_samples: Annotated[
        bool,
        typer.Option(
            "--resume-samples/--no-resume-samples",
            help="Resume benchmarks from existing matching samples.jsonl files.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Create the generated config and print planned work only."),
    ] = False,
    no_progress: Annotated[
        bool,
        typer.Option("--no-progress", help="Disable the live progress display."),
    ] = False,
    auto_export: Annotated[
        bool,
        typer.Option(
            "--auto-export/--no-auto-export",
            help="Export website results JSON after a successful benchmark run.",
        ),
    ] = True,
    auto_push: Annotated[
        bool,
        typer.Option(
            "--auto-push/--no-auto-push",
            help="Commit and push exported website result files after each benchmark is saved.",
        ),
    ] = True,
) -> None:
    """Run a benchmark against an OpenAI API model."""
    config = _write_api_benchmark_config(
        provider="OpenAI",
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
        smoke=smoke,
        languages=languages,
        benchmark=benchmark,
        bfcl_categories=bfcl_categories,
        ocrbench_configs=ocrbench_configs,
        mmmu_subjects=mmmu_subjects,
        mmmu_split=mmmu_split,
        mbpp_config=mbpp_config,
        mbpp_split=mbpp_split,
        mbpp_challenge_tests=mbpp_challenge_tests,
        rgb_dataset=rgb_dataset,
        rgb_noise_rate=rgb_noise_rate,
        rgb_passage_num=rgb_passage_num,
        rgb_correct_rate=rgb_correct_rate,
        simpleqa_grader=simpleqa_grader,
        simpleqa_grader_model=simpleqa_grader_model,
        harmbench_categories=harmbench_categories,
        harmbench_judge=harmbench_judge,
        harmbench_judge_model=harmbench_judge_model,
        reasoning_effort=reasoning_effort,
        modality=modality,
        max_tokens=max_tokens,
        context_length=None,
        restart_between_languages=False,
        restart_every_calls=None,
        restart_cooldown_seconds=0.0,
        restart_cooldown_every_calls=None,
        resume_samples=resume_samples,
    )
    console.print(f"🧾 Generated config: {config}")
    _execute_run(
        config,
        dry_run=dry_run,
        no_progress=no_progress,
        auto_export=auto_export,
        auto_push=auto_push,
    )


def _execute_run(
    config: Path,
    *,
    dry_run: bool,
    no_progress: bool,
    auto_export: bool,
    auto_push: bool,
) -> None:
    lmstudio_eject = None if dry_run else _lmstudio_eject_target(config)
    try:
        if no_progress:
            result = run_benchmark_with_progress(
                config,
                dry_run=dry_run,
                progress_callback=_benchmark_saved_callback(
                    auto_export=auto_export,
                    auto_push=auto_push,
                    dry_run=dry_run,
                ),
            )
        else:
            result = _run_with_progress(
                config,
                dry_run,
                auto_export=auto_export,
                auto_push=auto_push,
            )
    except BenchmarkRunError as exc:
        result = exc.result
        exported_results = (
            _safe_export_website_results(Path(str(result["db_path"])))
            if auto_export and not dry_run and result.get("db_path")
            else []
        )
        if no_progress:
            console.print(f"💥 [red]Run failed:[/red] {exc}")
        console.print("⚠️ [yellow]Run failed after saving completed benchmark metrics[/yellow]")
        for key, value in result.items():
            console.print(f"{key}: {value}")
        for exported in exported_results:
            console.print(
                f"🌐 Exported partial website data: {exported['rows']} rows to {exported['out']}"
            )
        _eject_lmstudio_model(lmstudio_eject)
        raise typer.Exit(1) from exc
    except RuntimeError as exc:
        if no_progress:
            console.print(f"💥 [red]Run failed:[/red] {exc}")
        _eject_lmstudio_model(lmstudio_eject)
        raise typer.Exit(1) from exc
    except KeyboardInterrupt as exc:
        exported_results = (
            _safe_export_website_results(Path(str(load_config(config).run.output_db)))
            if auto_export and not dry_run
            else []
        )
        if exported_results:
            console.print(
                "⚠️ [yellow]Interrupted after saving completed benchmark metrics[/yellow]"
            )
            for exported in exported_results:
                console.print(
                    f"🌐 Exported partial website data: {exported['rows']} rows "
                    f"to {exported['out']}"
                )
        _eject_lmstudio_model(lmstudio_eject)
        raise typer.Exit(130) from exc
    exported_results = (
        _safe_export_website_results(Path(str(result["db_path"])))
        if auto_export and not dry_run and result.get("db_path")
        else []
    )
    console.print("✅ [green]Run complete[/green]")
    for key, value in result.items():
        console.print(f"{key}: {value}")
    for exported in exported_results:
        console.print(f"🌐 Exported website data: {exported['rows']} rows to {exported['out']}")
    _eject_lmstudio_model(lmstudio_eject)


@app.command()
def normalize(
    db: Annotated[
        Path,
        typer.Option("--db", help="DuckDB database path."),
    ] = Path("results/local_ai_analysis.duckdb"),
) -> None:
    """Recompute normalized benchmark rows."""
    store = LocalAIAnalysisDB(db)
    try:
        store.init_schema()
        rows = refresh_normalized_results(store)
    finally:
        store.close()
    console.print(f"[green]Normalized[/green] {rows} rows")


@app.command()
def leaderboard(
    db: Annotated[
        Path,
        typer.Option("--db", help="DuckDB database path."),
    ] = Path("results/local_ai_analysis.duckdb"),
) -> None:
    """Print the current leaderboard."""
    payload = leaderboard_payload(db)
    table = Table(title="Local AI Analysis Leaderboard")
    for column in [
        "Model",
        "Backend",
        "Intel pts",
        "Cov.",
        "GMMLU Lite",
        "IFBench",
        "BFCL v4",
        "OCRBench v2",
        "MMMU",
        "MBPP",
        "RGB",
        "SimpleQA",
        "HarmBench",
        "Invalid",
        "Runtime",
        "Run",
    ]:
        table.add_column(column)

    for row in payload["leaderboard"]:
        table.add_row(
            str(row.get("variant_name") or ""),
            str(row.get("backend_name") or "n/a"),
            _points(row.get("model_intelligence_score")),
            _pct(row.get("model_intelligence_coverage")),
            _pct(row.get("global_mmlu_lite_pass_at_1")),
            _pct(row.get("ifbench_prompt_level_loose")),
            _pct(row.get("bfcl_v4_selected_accuracy")),
            _pct(row.get("ocrbench_v2_score")),
            _pct(row.get("mmmu_accuracy")),
            _pct(row.get("mbpp_pass_at_1")),
            _pct(row.get("rgb_all_rate")),
            _pct(row.get("simpleqa_f1")),
            _pct(row.get("harmbench_refusal_rate")),
            _pct(
                _first_metric_value(
                    row.get("global_mmlu_lite_invalid_rate"),
                    row.get("mbpp_invalid_rate"),
                    row.get("mmmu_invalid_rate"),
                    row.get("bfcl_v4_invalid_rate"),
                    row.get("simpleqa_incorrect_rate"),
                )
            ),
            _duration(row.get("benchmark_runtime_seconds")),
            _short_id(row.get("run_uuid")),
        )
    console.print(table)


@app.command()
def export(
    db: Annotated[
        Path,
        typer.Option("--db", help="DuckDB database path."),
    ] = Path("results/local_ai_analysis.duckdb"),
    format: Annotated[
        str,
        typer.Option("--format", help="Export format: json or csv."),
    ] = "json",
    out: Annotated[
        Path,
        typer.Option("--out", help="Output file path."),
    ] = Path("public/results.json"),
) -> None:
    """Export normalized leaderboard rows for the website."""
    result = export_leaderboard(db_path=db, out=out, fmt=format)
    console.print(f"[green]Exported[/green] {result['rows']} rows to {result['out']}")


@app.command()
def serve(
    db: Annotated[
        Path,
        typer.Option("--db", help="DuckDB database path."),
    ] = Path("results/local_ai_analysis.duckdb"),
    host: Annotated[
        str,
        typer.Option("--host", help="Host to bind."),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option("--port", help="Port to bind."),
    ] = 8000,
) -> None:
    """Serve the FastAPI backend."""
    import uvicorn

    from local_ai_analysis.api.app import create_app

    uvicorn.run(create_app(db_path=db), host=host, port=port)


def _pct(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def _points(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.1f}"


def _first_metric_value(*values: object) -> object:
    for value in values:
        if value is not None:
            return value
    return None


def _duration(value: object) -> str:
    if value is None:
        return "n/a"
    seconds = float(value)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remaining_seconds = int(round(seconds % 60))
    if minutes < 60:
        return f"{minutes}m {remaining_seconds}s"
    hours = minutes // 60
    return f"{hours}h {minutes % 60}m"


def _short_id(value: object) -> str:
    if not value:
        return "n/a"
    text = str(value)
    return text if len(text) <= 12 else f"{text[:8]}...{text[-4:]}"


def _export_website_results(db_path: Path) -> list[dict[str, Any]]:
    paths = [Path("public/results.json"), Path("web/public/results.json")]
    if Path("web/dist").exists():
        paths.append(Path("web/dist/results.json"))

    exported: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        exported.append(export_leaderboard(db_path=db_path, out=path, fmt="json"))
    return exported


def _safe_export_website_results(db_path: Path) -> list[dict[str, Any]]:
    try:
        return _export_website_results(db_path)
    except Exception as exc:
        console.print(f"⚠️ [yellow]Could not export website data:[/yellow] {exc}")
        return []


def _benchmark_saved_callback(
    *,
    auto_export: bool,
    auto_push: bool,
    dry_run: bool,
) -> Any:
    if dry_run or (not auto_export and not auto_push):
        return None

    def callback(event_type: str, payload: dict[str, Any]) -> None:
        if event_type != "benchmark_saved":
            return
        exported_results = []
        if auto_export:
            exported_results = _export_for_benchmark_payload(payload)
        if auto_push:
            _safe_commit_and_push_results(payload, exported_results)

    return callback


def _export_for_benchmark_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    db_path = payload.get("db_path")
    if not db_path:
        return []
    return _safe_export_website_results(Path(str(db_path)))


def _safe_commit_and_push_results(
    payload: dict[str, Any],
    exported_results: list[dict[str, Any]],
) -> None:
    try:
        pushed = _commit_and_push_results(payload, exported_results)
    except Exception as exc:
        console.print(f"⚠️ [yellow]Could not push benchmark results:[/yellow] {exc}")
        return
    if pushed:
        console.print(f"🚀 Pushed benchmark results to GitHub • {pushed}")


def _commit_and_push_results(
    payload: dict[str, Any],
    exported_results: list[dict[str, Any]],
) -> str | None:
    paths = _result_paths_for_git(exported_results)
    if not paths:
        return None

    def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=Path.cwd(),
            text=True,
            capture_output=True,
            check=check,
        )

    git("add", "--", *paths)
    diff = git("diff", "--cached", "--quiet", "--", *paths, check=False)
    if diff.returncode == 0:
        return None
    if diff.returncode not in {0, 1}:
        raise RuntimeError((diff.stderr or diff.stdout).strip() or "git diff failed")

    task = _slug(str(payload.get("task") or "benchmark"))
    variant = _slug(str(payload.get("variant") or "model"))
    commit_message = f"Update benchmark results: {variant} {task}"
    git("commit", "-m", commit_message, "--", *paths)
    push = git("push")
    return (push.stdout or push.stderr).strip() or "origin"


def _result_paths_for_git(exported_results: list[dict[str, Any]]) -> list[str]:
    candidates = [
        Path(str(exported.get("out")))
        for exported in exported_results
        if exported.get("out")
    ]
    if not candidates:
        candidates = [
            Path("public/results.json"),
            Path("web/public/results.json"),
            Path("web/dist/results.json"),
        ]
    paths: list[str] = []
    seen: set[Path] = set()
    for path in candidates:
        if not path.exists():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        paths.append(str(path))
    return paths


def _lmstudio_eject_target(config_path: Path) -> dict[str, str] | None:
    try:
        config = load_config(config_path)
    except Exception:
        return None
    if _provider_key(config.backend.backend_type) != "lmstudio":
        return None
    model = None
    for base_model in config.models:
        for variant in base_model.variants:
            if variant.api_model:
                model = variant.api_model
                break
        if model:
            break
    if not model or model.lower() == "auto":
        return None
    base_url = _first_lmstudio_base_url(config)
    if not base_url:
        return None
    api_key = _first_lmstudio_api_key(config)
    return {"base_url": base_url.rstrip("/"), "model": model, "api_key": api_key or ""}


def _first_lmstudio_base_url(config: Any) -> str | None:
    for settings in _benchmark_settings(config):
        if getattr(settings, "enabled", False) and _provider_key(getattr(settings, "provider", "")) == "lmstudio":
            base_url = getattr(settings, "base_url", None)
            if base_url:
                return str(base_url)
    return None


def _first_lmstudio_api_key(config: Any) -> str | None:
    for settings in _benchmark_settings(config):
        if not getattr(settings, "enabled", False):
            continue
        if _provider_key(getattr(settings, "provider", "")) != "lmstudio":
            continue
        api_key = getattr(settings, "api_key", None)
        if api_key:
            return str(api_key)
        api_key_env = getattr(settings, "api_key_env", None)
        if api_key_env:
            import os

            env_value = os.environ.get(str(api_key_env))
            if env_value:
                return env_value
    return None


def _benchmark_settings(config: Any) -> list[Any]:
    return [
        config.global_mmlu_lite,
        config.ifbench,
        config.bfcl_v4,
        config.ocrbench_v2,
        config.mmmu,
        config.mbpp,
        config.rgb,
        config.simpleqa,
        config.harmbench,
    ]


def _eject_lmstudio_model(target: dict[str, str] | None) -> None:
    if not target:
        return
    try:
        instance_id = _lmstudio_loaded_instance_id(target["base_url"], target["model"], target["api_key"])
        if not instance_id:
            console.print(f"🧹 LM Studio model already unloaded or not found: {target['model']}")
            return
        _lmstudio_unload_instance(target["base_url"], instance_id, target["api_key"])
    except Exception as exc:
        console.print(f"⚠️ [yellow]Could not eject LM Studio model:[/yellow] {exc}")
        return
    console.print(f"🧹 Ejected LM Studio model: {instance_id}")


def _lmstudio_loaded_instance_id(base_url: str, model: str, api_key: str | None) -> str | None:
    request = urllib.request.Request(
        f"{base_url}/api/v1/models",
        headers=_lmstudio_headers(api_key),
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    for item in payload.get("models") or []:
        if not isinstance(item, dict) or model not in _lmstudio_model_ids_from_payload(item):
            continue
        for instance in item.get("loaded_instances") or []:
            if isinstance(instance, dict) and instance.get("id"):
                return str(instance["id"])
    return None


def _lmstudio_unload_instance(base_url: str, instance_id: str, api_key: str | None) -> None:
    body = json.dumps({"instance_id": instance_id}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/api/v1/models/unload",
        data=body,
        headers=_lmstudio_headers(api_key),
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        response.read()


def _lmstudio_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _lmstudio_model_ids_from_payload(item: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for key in ("key", "selected_variant"):
        value = item.get(key)
        if isinstance(value, str):
            ids.add(value)
    for variant in item.get("variants") or []:
        if isinstance(variant, str):
            ids.add(variant)
    for instance in item.get("loaded_instances") or []:
        if isinstance(instance, dict) and isinstance(instance.get("id"), str):
            ids.add(instance["id"])
    return ids


def _write_api_benchmark_config(
    *,
    provider: str,
    model: str,
    base_url: str,
    api_key_env: str | None,
    smoke: bool,
    languages: str,
    benchmark: str,
    bfcl_categories: str,
    ocrbench_configs: str,
    mmmu_subjects: str,
    mmmu_split: str,
    mbpp_config: str,
    mbpp_split: str,
    mbpp_challenge_tests: bool,
    rgb_dataset: str,
    rgb_noise_rate: float,
    rgb_passage_num: int,
    rgb_correct_rate: float,
    simpleqa_grader: str,
    simpleqa_grader_model: str,
    harmbench_categories: str,
    harmbench_judge: str,
    harmbench_judge_model: str,
    reasoning_effort: str | None,
    modality: str,
    max_tokens: int | None,
    context_length: int | None,
    restart_between_languages: bool,
    restart_every_calls: int | None,
    restart_cooldown_seconds: float,
    restart_cooldown_every_calls: int | None,
    resume_samples: bool = False,
) -> Path:
    selected_benchmarks = _parse_benchmarks(benchmark)
    provider_key = _provider_key(provider)
    resolved_reasoning_effort = _resolve_reasoning_effort(
        model,
        reasoning_effort,
        provider_key=provider_key,
    )
    selected_languages = ["en"] if smoke else _parse_languages(languages)
    selected_ocrbench_configs = (
        _parse_ocrbench_configs(OCRBENCH_V2_SMOKE_CONFIGS)
        if smoke
        else _parse_ocrbench_configs(ocrbench_configs)
    )
    selected_mmmu_subjects = (
        _parse_mmmu_subjects(MMMU_SMOKE_SUBJECTS)
        if smoke
        else _parse_mmmu_subjects(mmmu_subjects)
    )
    selected_mmmu_split = _parse_mmmu_split(mmmu_split)
    selected_mbpp_config = _parse_mbpp_config(mbpp_config)
    selected_mbpp_split = _parse_mbpp_split(mbpp_split)
    selected_rgb_dataset = _parse_rgb_dataset(rgb_dataset)
    selected_simpleqa_grader = _parse_simpleqa_grader(simpleqa_grader)
    selected_harmbench_categories = _parse_harmbench_categories(harmbench_categories)
    selected_harmbench_judge = _parse_harmbench_judge(harmbench_judge)
    resolved_harmbench_judge_model = _shared_judge_model(
        simpleqa_grader_model,
        harmbench_judge_model,
    )
    selected_input_modalities = _input_modalities(selected_benchmarks)
    resolved_modality = _resolve_modality(
        model=model,
        selected_benchmarks=selected_benchmarks,
        value=modality,
    )
    sample_limit = 5 if smoke else None
    all_languages_selected = selected_languages == GLOBAL_MMLU_LITE_LANGUAGES
    benchmark_slug = _slug("-".join(selected_benchmarks))
    provider_slug = _slug(provider)
    model_slug = _slug(model)
    scope_slug = (
        "smoke"
        if smoke
        else (
            _slug(f"{selected_mbpp_config}-{selected_mbpp_split}")
            if selected_benchmarks == ["mbpp"]
            else (
                (
                    "suite"
                    if selected_rgb_dataset == "suite"
                    else _slug(
                        f"{selected_rgb_dataset}-noise-{rgb_noise_rate}-p{rgb_passage_num}"
                    )
                )
                if selected_benchmarks == ["rgb"]
                else ("full" if all_languages_selected else _slug("-".join(selected_languages)))
            )
        )
    )
    reasoning_slug = _slug(f"reasoning-{resolved_reasoning_effort or 'unset'}")
    output_path = (
        Path("results/generated_configs")
        / f"{provider_slug}_{model_slug}_{benchmark_slug}_{scope_slug}_{reasoning_slug}.yaml"
    )
    model_metadata = _model_metadata(provider_key, base_url, model)
    display_model = _display_model_name(provider, model, model_metadata)
    provider_label = provider if provider != "LM Studio" else "LM Studio"
    variant_suffix = _variant_suffix(
        smoke=smoke,
        selected_benchmarks=selected_benchmarks,
        selected_mbpp_config=selected_mbpp_config,
        selected_mbpp_split=selected_mbpp_split,
        selected_rgb_dataset=selected_rgb_dataset,
        rgb_noise_rate=rgb_noise_rate,
        selected_simpleqa_grader=selected_simpleqa_grader,
        selected_harmbench_categories=selected_harmbench_categories,
        all_languages_selected=all_languages_selected,
    )
    reasoning_label = resolved_reasoning_effort or "unset"
    quantization = _infer_quantization(model, model_metadata)
    request_extra = _context_request_extra(provider_key, context_length)

    global_mmlu_lite: dict[str, Any] = {
        "enabled": "global-mmlu-lite" in selected_benchmarks,
        "dataset_name": "CohereLabs/Global-MMLU-Lite",
        "dataset_revision": GLOBAL_MMLU_LITE_REVISION,
        "split": "test",
        "languages": selected_languages,
        "sample_limit_per_language": sample_limit,
        "output_dir": "results/global_mmlu_lite",
        "provider": provider_key,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "timeout_seconds": 180,
        "temperature": 0,
        "max_tokens": max_tokens if max_tokens is not None else 128,
        "top_p": None,
        "stop": None,
        "seed": 42,
        "strip_thinking": True,
        "restart_between_languages": restart_between_languages and provider_key == "lmstudio",
        "restart_every_calls": (
            restart_every_calls if provider_key == "lmstudio" else None
        ),
        "restart_cooldown_seconds": (
            restart_cooldown_seconds if provider_key == "lmstudio" else 0.0
        ),
        "restart_cooldown_every_calls": (
            restart_cooldown_every_calls if provider_key == "lmstudio" else None
        ),
        "resume_samples": resume_samples,
        "request_extra": _copy_config_dict(request_extra),
        "parser_version": "global_mmlu_lite_regex_v1",
        "prompt_template": GLOBAL_MMLU_LITE_PROMPT,
    }
    if resolved_reasoning_effort:
        global_mmlu_lite["reasoning_effort"] = resolved_reasoning_effort
    restart_settings = {
        "restart_every_calls": (
            restart_every_calls if provider_key == "lmstudio" else None
        ),
        "restart_cooldown_seconds": (
            restart_cooldown_seconds if provider_key == "lmstudio" else 0.0
        ),
        "restart_cooldown_every_calls": (
            restart_cooldown_every_calls if provider_key == "lmstudio" else None
        ),
        "resume_samples": resume_samples,
    }

    ifbench: dict[str, Any] = {
        "enabled": "ifbench" in selected_benchmarks,
        "dataset_name": "allenai/IFBench_test",
        "dataset_revision": IFBENCH_REVISION,
        "split": "train",
        "sample_limit": sample_limit,
        "output_dir": "results/ifbench",
        "provider": provider_key,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "timeout_seconds": 240,
        "temperature": 0,
        "max_tokens": max_tokens if max_tokens is not None else 4096,
        "top_p": 0.95,
        "stop": None,
        "seed": 42,
        "strip_thinking": True,
        **restart_settings,
        "request_extra": _copy_config_dict(request_extra),
        "evaluator": "allenai_ifbench_loose_v1",
    }
    if resolved_reasoning_effort:
        ifbench["reasoning_effort"] = resolved_reasoning_effort

    bfcl_v4: dict[str, Any] = {
        "enabled": "bfcl-v4" in selected_benchmarks,
        "version": "BFCL_v4",
        "categories": _parse_bfcl_categories(bfcl_categories),
        "sample_limit": sample_limit if smoke else BFCL_V4_DEFAULT_SAMPLE_LIMIT,
        "sample_strategy": "stratified",
        "sample_seed": 42,
        "output_dir": "results/bfcl_v4",
        "provider": provider_key,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "timeout_seconds": 300,
        "temperature": 0,
        "max_tokens": max_tokens if max_tokens is not None else 1024,
        "top_p": None,
        "stop": None,
        "seed": 42,
        "strip_thinking": True,
        **restart_settings,
        "request_extra": _copy_config_dict(request_extra),
        "include_input_log": False,
        "exclude_state_log": True,
        "evaluator": "bfcl_eval_prompt_mode_v4",
    }
    if resolved_reasoning_effort:
        bfcl_v4["reasoning_effort"] = resolved_reasoning_effort

    ocrbench_v2: dict[str, Any] = {
        "enabled": "ocrbench-v2" in selected_benchmarks,
        "dataset_name": "morpheushoc/OCRBenchv2",
        "dataset_revision": OCRBENCH_V2_REVISION,
        "split": "test",
        "dataset_configs": selected_ocrbench_configs,
        "sample_limit": sample_limit if smoke else OCRBENCH_V2_DEFAULT_SAMPLE_LIMIT,
        "sample_strategy": "stratified",
        "sample_seed": 42,
        "output_dir": "results/ocrbench_v2",
        "provider": provider_key,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "timeout_seconds": 360,
        "temperature": 0,
        "max_tokens": max_tokens if max_tokens is not None else 2048,
        "top_p": None,
        "stop": None,
        "seed": 42,
        "strip_thinking": True,
        **restart_settings,
        "request_extra": _copy_config_dict(request_extra),
        "image_format": "PNG",
        "evaluator": "ocrbench_v2_local_vqa_anls_iou_v1",
        "prompt_template": OCRBENCH_V2_PROMPT,
    }
    if resolved_reasoning_effort:
        ocrbench_v2["reasoning_effort"] = resolved_reasoning_effort

    mmmu: dict[str, Any] = {
        "enabled": "mmmu" in selected_benchmarks,
        "dataset_name": "MMMU/MMMU",
        "dataset_revision": MMMU_REVISION,
        "split": selected_mmmu_split,
        "subjects": selected_mmmu_subjects,
        "sample_limit": sample_limit,
        "output_dir": "results/mmmu",
        "provider": provider_key,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "timeout_seconds": 360,
        "temperature": 0,
        "max_tokens": max_tokens if max_tokens is not None else 512,
        "top_p": None,
        "stop": None,
        "seed": 42,
        "strip_thinking": True,
        **restart_settings,
        "request_extra": _copy_config_dict(request_extra),
        "image_format": "PNG",
        "evaluator": "mmmu_official_parse_local_v1",
        "multiple_choice_prompt_template": MMMU_MULTI_CHOICE_PROMPT,
        "short_answer_prompt_template": MMMU_SHORT_ANSWER_PROMPT,
    }
    if resolved_reasoning_effort:
        mmmu["reasoning_effort"] = resolved_reasoning_effort

    mbpp: dict[str, Any] = {
        "enabled": "mbpp" in selected_benchmarks,
        "dataset_name": "google-research-datasets/mbpp",
        "dataset_config": selected_mbpp_config,
        "dataset_revision": MBPP_REVISION,
        "split": selected_mbpp_split,
        "sample_limit": sample_limit,
        "output_dir": "results/mbpp",
        "provider": provider_key,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "timeout_seconds": 360,
        "temperature": 0,
        "max_tokens": max_tokens if max_tokens is not None else 2048,
        "top_p": None,
        "stop": None if provider_key == "openai" else ["[DONE]"],
        "seed": 42,
        "strip_thinking": True,
        **restart_settings,
        "request_extra": _copy_config_dict(request_extra),
        "include_tests_in_prompt": True,
        "include_challenge_tests": mbpp_challenge_tests,
        "execution_timeout_seconds": 5,
        "evaluator": "mbpp_local_subprocess_pass_at_1_v1",
        "prompt_template": MBPP_PROMPT,
    }
    if resolved_reasoning_effort:
        mbpp["reasoning_effort"] = resolved_reasoning_effort

    rgb: dict[str, Any] = {
        "enabled": "rgb" in selected_benchmarks,
        "dataset_name": "chen700564/RGB",
        "dataset_revision": RGB_REVISION,
        "dataset": selected_rgb_dataset,
        "sample_limit": sample_limit if smoke else RGB_DEFAULT_SAMPLE_LIMIT,
        "sample_strategy": "random",
        "output_dir": "results/rgb",
        "data_cache_dir": "results/rgb/cache",
        "provider": provider_key,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "timeout_seconds": 360,
        "temperature": 0,
        "max_tokens": max_tokens if max_tokens is not None else 2048,
        "top_p": None,
        "stop": None,
        "seed": 2333,
        "strip_thinking": True,
        **restart_settings,
        "request_extra": _copy_config_dict(request_extra),
        "noise_rate": rgb_noise_rate,
        "passage_num": rgb_passage_num,
        "correct_rate": rgb_correct_rate,
        "evaluator": (
            "rgb_curated_suite_lexical_v1"
            if selected_rgb_dataset == "suite"
            else "rgb_official_lexical_v1"
        ),
    }
    if resolved_reasoning_effort:
        rgb["reasoning_effort"] = resolved_reasoning_effort

    simpleqa: dict[str, Any] = {
        "enabled": "simpleqa" in selected_benchmarks,
        "dataset_name": "openai/simpleqa",
        "dataset_url": SIMPLEQA_DATASET_URL,
        "dataset_revision": SIMPLE_EVALS_REVISION,
        "sample_limit": sample_limit if smoke else SIMPLEQA_DEFAULT_SAMPLE_LIMIT,
        "sample_strategy": "stratified",
        "sample_seed": 42,
        "output_dir": "results/simpleqa",
        "data_cache_dir": "results/simpleqa/cache",
        "refresh_cache": False,
        "download_timeout_seconds": 60,
        "provider": provider_key,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "timeout_seconds": 240,
        "temperature": 0,
        "max_tokens": max_tokens if max_tokens is not None else 128,
        "top_p": None,
        "stop": None,
        "seed": 42,
        "strip_thinking": True,
        **restart_settings,
        "request_extra": _copy_config_dict(request_extra),
        "grader": selected_simpleqa_grader,
        "grader_model": simpleqa_grader_model,
        "grader_provider": None,
        "grader_base_url": None,
        "grader_api_key_env": api_key_env,
        "grader_timeout_seconds": 240,
        "grader_temperature": 0,
        "grader_max_tokens": 16,
        "grader_top_p": None,
        "grader_stop": None,
        "grader_seed": 42,
        "grader_reasoning_effort": "none",
        "grader_request_extra": _copy_config_dict(request_extra),
        "evaluator": (
            "simpleqa_heuristic_v1"
            if selected_simpleqa_grader == "heuristic"
            else "simpleqa_llm_judge_v1"
        ),
        "prompt_template": SIMPLEQA_PROMPT,
    }
    if resolved_reasoning_effort:
        simpleqa["reasoning_effort"] = resolved_reasoning_effort

    harmbench: dict[str, Any] = {
        "enabled": "harmbench" in selected_benchmarks,
        "dataset_name": "centerforaisafety/HarmBench",
        "dataset_url": HARMBENCH_DATASET_URL,
        "dataset_revision": HARMBENCH_REVISION,
        "functional_categories": selected_harmbench_categories,
        "sample_limit": sample_limit,
        "output_dir": "results/harmbench",
        "data_cache_dir": "results/harmbench/cache",
        "refresh_cache": False,
        "download_timeout_seconds": 60,
        "provider": provider_key,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "timeout_seconds": 300,
        "temperature": 0,
        "max_tokens": max_tokens if max_tokens is not None else 512,
        "top_p": None,
        "stop": None,
        "seed": 42,
        "strip_thinking": True,
        **restart_settings,
        "request_extra": _copy_config_dict(request_extra),
        "judge": selected_harmbench_judge,
        "judge_model": resolved_harmbench_judge_model,
        "judge_provider": None,
        "judge_base_url": None,
        "judge_api_key_env": api_key_env,
        "judge_timeout_seconds": 300,
        "judge_temperature": 0,
        "judge_max_tokens": 24,
        "judge_top_p": None,
        "judge_stop": None,
        "judge_seed": 42,
        "judge_reasoning_effort": "none",
        "judge_request_extra": _copy_config_dict(request_extra),
        "evaluator": (
            "harmbench_heuristic_refusal_v1"
            if selected_harmbench_judge == "heuristic"
            else "harmbench_llm_judge_v1"
        ),
        "prompt_template": "{context_prefix}{behavior}",
    }
    if resolved_reasoning_effort:
        harmbench["reasoning_effort"] = resolved_reasoning_effort

    payload = {
        "project": "Local AI Analysis",
        "run": {
            "name": (
                f"{benchmark_slug}-{provider_slug}-{model_slug}-{scope_slug}-"
                f"{reasoning_slug}"
            ),
            "output_db": "results/local_ai_analysis.duckdb",
            "raw_jsonl": "results/raw_results.jsonl",
            "seed": 42,
            "strict_model_files": False,
            "notes": (
                f"{'Smoke' if smoke else 'Full'} {benchmark_label(selected_benchmarks)} "
                f"run with {display_model} "
                f"served by {provider_label} through its native API. "
                f"reasoning_effort={reasoning_label}."
            ),
        },
        "backend": {
            "name": provider_label,
            "backend_type": provider_key,
            "version": None,
            "commit": None,
            "command": (
                "OpenAI API"
                if provider_key == "openai"
                else f"{provider_label} local server"
            ),
        },
        "global_mmlu_lite": global_mmlu_lite,
        "ifbench": ifbench,
        "bfcl_v4": bfcl_v4,
        "ocrbench_v2": ocrbench_v2,
        "mmmu": mmmu,
        "mbpp": mbpp,
        "rgb": rgb,
        "simpleqa": simpleqa,
        "harmbench": harmbench,
        "models": [
            {
                "family": _infer_family(model, model_metadata),
                "name": _infer_base_model_name(provider, model, model_metadata),
                "parameter_size_b": _infer_parameter_size_b(model, model_metadata),
                "architecture": _architecture_for_modality(resolved_modality),
                "modality": resolved_modality,
                "license": "replace-with-upstream-license",
                "source_url": base_url,
                "variants": [
                    {
                        "name": (
                            f"{display_model} {quantization} {provider_label} {variant_suffix} "
                            f"Reasoning {reasoning_label}"
                        ),
                        "quantization": quantization,
                        "precision": _infer_precision(model, quantization),
                        "baseline": False,
                        "api_model": model,
                        "model_repo": _model_metadata_value(model_metadata, "publisher"),
                        "file_size_bytes": _model_metadata_value(model_metadata, "size_bytes"),
                        "modality": resolved_modality,
                        "input_modalities": selected_input_modalities,
                    }
                ],
            }
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)
    return output_path


def _parse_benchmarks(value: str) -> list[str]:
    normalized = value.strip().lower().replace("_", "-")
    aliases = {
        "global": "global-mmlu-lite",
        "gmmlu": "global-mmlu-lite",
        "global-mmlu": "global-mmlu-lite",
        "global-mmlu-lite": "global-mmlu-lite",
        "if": "ifbench",
        "if-bench": "ifbench",
        "ifbench": "ifbench",
        "bfcl": "bfcl-v4",
        "bfcl-v4": "bfcl-v4",
        "function-calling": "bfcl-v4",
        "fc": "bfcl-v4",
        "ocr": "ocrbench-v2",
        "ocrbench": "ocrbench-v2",
        "ocrbench-v2": "ocrbench-v2",
        "mmmu": "mmmu",
        "multimodal-mmlu": "mmmu",
        "mbpp": "mbpp",
        "code": "mbpp",
        "code-generation": "mbpp",
        "rgb": "rgb",
        "rag": "rgb",
        "rag-benchmark": "rgb",
        "simpleqa": "simpleqa",
        "simple-qa": "simpleqa",
        "factuality": "simpleqa",
        "hallucination": "simpleqa",
        "hallucinations": "simpleqa",
        "harmbench": "harmbench",
        "harm-bench": "harmbench",
        "safety": "harmbench",
        "red-team": "harmbench",
        "redteam": "harmbench",
    }
    suites = {
        "text": CORE_BENCHMARKS,
        "text-only": CORE_BENCHMARKS,
        "core": CORE_BENCHMARKS,
        "non-judge-text": CORE_BENCHMARKS,
        "vision": MULTIMODAL_BENCHMARKS,
        "multimodal": MULTIMODAL_BENCHMARKS,
        "multimodal-vision": MULTIMODAL_BENCHMARKS,
        "suite": ALL_NON_JUDGED_BENCHMARKS,
        "capability-suite": ALL_NON_JUDGED_BENCHMARKS,
        "all": ALL_NON_JUDGED_BENCHMARKS,
        "all-nonjudge": ALL_NON_JUDGED_BENCHMARKS,
        "all-non-judge": ALL_NON_JUDGED_BENCHMARKS,
        "capability": ALL_NON_JUDGED_BENCHMARKS,
        "capabilities": ALL_NON_JUDGED_BENCHMARKS,
        "judge": JUDGED_BENCHMARKS,
        "judged": JUDGED_BENCHMARKS,
        "llm-judge": JUDGED_BENCHMARKS,
        "judge-required": JUDGED_BENCHMARKS,
        "recommended": FULL_SUITE_BENCHMARKS,
        "recommended-suite": FULL_SUITE_BENCHMARKS,
        "full": FULL_SUITE_BENCHMARKS,
        "everything": FULL_SUITE_BENCHMARKS,
        "complete": FULL_SUITE_BENCHMARKS,
    }
    if normalized in suites:
        return list(suites[normalized])
    selected: list[str] = []
    for item in [part.strip() for part in normalized.split(",") if part.strip()]:
        if item in suites:
            selected.extend(suites[item])
        else:
            selected.append(aliases.get(item, item))
    selected = [item for item in selected if item]
    selected = list(dict.fromkeys(selected))
    unknown = sorted(
        set(selected)
        - {
            "global-mmlu-lite",
            "ifbench",
            "bfcl-v4",
            "ocrbench-v2",
            "mmmu",
            "mbpp",
            "rgb",
            "simpleqa",
            "harmbench",
        }
    )
    if unknown:
        raise typer.BadParameter(
            f"Unsupported benchmark(s): {', '.join(unknown)}. "
            "Use text, vision, judge, suite, recommended, full, an individual benchmark, "
            "or a comma-separated list."
        )
    if not selected:
        raise typer.BadParameter(
            "--benchmark must be text, vision, judge, suite, recommended, full, an individual benchmark, or a comma-separated list"
        )
    return selected


def _variant_suffix(
    *,
    smoke: bool,
    selected_benchmarks: list[str],
    selected_mbpp_config: str,
    selected_mbpp_split: str,
    selected_rgb_dataset: str,
    rgb_noise_rate: float,
    selected_simpleqa_grader: str,
    selected_harmbench_categories: list[str],
    all_languages_selected: bool,
) -> str:
    if smoke:
        return "Smoke"
    if selected_benchmarks == CORE_BENCHMARKS:
        return "Text Suite"
    if selected_benchmarks == MULTIMODAL_BENCHMARKS:
        return "Vision Suite"
    if selected_benchmarks == ALL_NON_JUDGED_BENCHMARKS:
        return "Capability Suite"
    if selected_benchmarks == JUDGED_BENCHMARKS:
        return "Judge Suite"
    if selected_benchmarks == FULL_SUITE_BENCHMARKS:
        return "Full Suite"
    if selected_benchmarks == ["mbpp"]:
        return f"MBPP {selected_mbpp_config} {selected_mbpp_split}"
    if selected_benchmarks == ["rgb"]:
        if selected_rgb_dataset == "suite":
            return "RGB Suite"
        return f"RGB {selected_rgb_dataset} noise {rgb_noise_rate:g}"
    if selected_benchmarks == ["simpleqa"]:
        return f"SimpleQA {selected_simpleqa_grader}"
    if selected_benchmarks == ["harmbench"]:
        return f"HarmBench {'+'.join(selected_harmbench_categories)}"
    return "All Languages" if all_languages_selected else "Selected Languages"


def benchmark_label(benchmarks: list[str]) -> str:
    labels = {
        "global-mmlu-lite": "Global MMLU Lite",
        "ifbench": "IFBench",
        "bfcl-v4": "BFCL v4",
        "ocrbench-v2": "OCRBench v2",
        "mmmu": "MMMU",
        "mbpp": "MBPP",
        "rgb": "RGB",
        "simpleqa": "SimpleQA",
        "harmbench": "HarmBench",
    }
    return " + ".join(labels[item] for item in benchmarks)


def _parse_ocrbench_configs(value: str) -> list[str]:
    normalized = value.strip()
    if normalized.lower() in {"", "all", "*", "official"}:
        return ["EN", "CN"]
    configs = [item.strip() for item in normalized.split(",") if item.strip()]
    return configs or ["EN", "CN"]


def _parse_mmmu_subjects(value: str) -> list[str]:
    normalized = value.strip()
    if normalized.lower() in {"", "all", "*", "official"}:
        return MMMU_SUBJECTS
    aliases = {subject.lower().replace("-", "_"): subject for subject in MMMU_SUBJECTS}
    subjects: list[str] = []
    for item in [part.strip() for part in normalized.split(",") if part.strip()]:
        key = item.lower().replace("-", "_").replace(" ", "_")
        subject = aliases.get(key)
        if subject is None:
            raise typer.BadParameter(
                f"Unsupported MMMU subject: {item}. Supported: {', '.join(MMMU_SUBJECTS)}"
            )
        subjects.append(subject)
    return subjects or MMMU_SUBJECTS


def _parse_mmmu_split(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in {"dev", "validation", "test"}:
        raise typer.BadParameter("--mmmu-split must be dev, validation, or test")
    return normalized


def _parse_mbpp_config(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"", "official"}:
        return MBPP_DEFAULT_CONFIG
    if normalized not in {"full", "sanitized"}:
        raise typer.BadParameter("--mbpp-config must be full or sanitized")
    return normalized


def _parse_mbpp_split(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in {"train", "test", "validation", "prompt"}:
        raise typer.BadParameter("--mbpp-split must be train, test, validation, or prompt")
    return normalized


def _parse_rgb_dataset(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "default": "suite",
        "curated": "suite",
        "rgb_suite": "suite",
        "en_noise": "en_refine",
        "zh_noise": "zh_refine",
        "en_rejection": "en_refine",
        "zh_rejection": "zh_refine",
        "en_integration": "en_int",
        "zh_integration": "zh_int",
        "en_counterfactual": "en_fact",
        "zh_counterfactual": "zh_fact",
    }
    dataset = aliases.get(normalized, normalized)
    if dataset not in RGB_DATASETS:
        raise typer.BadParameter(
            f"--rgb-dataset must be one of: {', '.join(RGB_DATASETS)}"
        )
    return dataset


def _parse_simpleqa_grader(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"", "llm", "judge", "llm_judge", "official"}:
        return "llm"
    if normalized in {"heuristic", "local", "exact"}:
        return "heuristic"
    raise typer.BadParameter("--simpleqa-grader must be llm or heuristic")


def _parse_harmbench_judge(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"", "llm", "judge", "llm_judge", "classifier", "official"}:
        return "llm"
    if normalized in {"heuristic", "local", "refusal"}:
        return "heuristic"
    raise typer.BadParameter("--harmbench-judge must be llm or heuristic")


def _parse_harmbench_categories(value: str) -> list[str]:
    aliases = {
        "std": "standard",
        "standard": "standard",
        "context": "contextual",
        "contextual": "contextual",
        "copyright": "copyright",
        "copy": "copyright",
    }
    selected: list[str] = []
    for item in [part.strip().lower().replace("-", "_") for part in value.split(",")]:
        if not item:
            continue
        if item in {"all", "*", "official"}:
            return ["standard", "contextual", "copyright"]
        if item == "text":
            selected.extend(["standard", "contextual"])
        else:
            selected.append(aliases.get(item, item))
    deduped: list[str] = []
    for item in selected:
        if item not in deduped:
            deduped.append(item)
    allowed = {"standard", "contextual", "copyright"}
    unknown = sorted(set(deduped) - allowed)
    if unknown:
        raise typer.BadParameter(
            "--harmbench-categories must use standard, contextual, copyright, text, or all"
        )
    return deduped or ["standard", "contextual"]


def _shared_judge_model(simpleqa_grader_model: str, harmbench_judge_model: str) -> str:
    simpleqa_model = simpleqa_grader_model.strip()
    harmbench_model = harmbench_judge_model.strip()
    if (
        harmbench_model.lower() in {"same", "@same"}
        and simpleqa_model
        and simpleqa_model.lower() not in {"same", "@same"}
    ):
        return simpleqa_model
    return harmbench_model or "same"


def _context_request_extra(provider_key: str, context_length: int | None) -> dict[str, Any]:
    if context_length is None:
        return {}
    if context_length <= 0:
        raise typer.BadParameter("--context-length must be greater than 0")
    if provider_key == "ollama":
        return {"options": {"num_ctx": context_length}}
    if provider_key == "lmstudio":
        return {"context_length": context_length}
    return {}


def _copy_config_dict(value: dict[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, item in value.items():
        copied[key] = dict(item) if isinstance(item, dict) else item
    return copied


def _parse_bfcl_categories(value: str) -> list[str]:
    categories = [item.strip().lower().replace("-", "_") for item in value.split(",")]
    categories = [item for item in categories if item]
    return categories or [BFCL_V4_DEFAULT_CATEGORIES]


def _parse_languages(value: str) -> list[str]:
    normalized = value.strip().lower()
    if normalized in {"", "all", "*"}:
        return GLOBAL_MMLU_LITE_LANGUAGES
    languages = [item.strip() for item in normalized.split(",") if item.strip()]
    if not languages:
        raise typer.BadParameter("--languages must be 'all' or a comma-separated list")
    unknown = sorted(set(languages) - set(GLOBAL_MMLU_LITE_LANGUAGES))
    if unknown:
        raise typer.BadParameter(
            f"Unsupported language(s): {', '.join(unknown)}. "
            f"Supported: {', '.join(GLOBAL_MMLU_LITE_LANGUAGES)}"
        )
    return languages


def _provider_key(provider: str) -> str:
    normalized = provider.strip().lower().replace(" ", "")
    if normalized == "ollama":
        return "ollama"
    if normalized in {"lmstudio", "lm-studio"}:
        return "lmstudio"
    if normalized == "omlx":
        return "omlx"
    if normalized == "openai":
        return "openai"
    return _slug(provider)


def _resolve_reasoning_effort(
    model: str,
    value: str | None,
    *,
    provider_key: str | None = None,
) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"", "unset", "none"}:
        return "none"
    if normalized == "auto":
        lowered_model = model.lower()
        if provider_key == "openai":
            if lowered_model.startswith(("gpt-5.4", "gpt-5.1")):
                return "none"
            if lowered_model.startswith("gpt-5"):
                return "minimal"
            return "none"
        return "none" if "qwen" in lowered_model else "high"
    return normalized


def _resolve_modality(
    *,
    model: str,
    selected_benchmarks: list[str],
    value: str,
) -> str:
    normalized = value.strip().lower().replace("_", "-")
    aliases = {
        "": "auto",
        "auto": "auto",
        "text": "text",
        "text-only": "text",
        "vl": "multimodal",
        "vision": "vision",
        "visual": "vision",
        "image": "vision",
        "multimodal": "multimodal",
        "multi-modal": "multimodal",
        "mm": "multimodal",
    }
    resolved = aliases.get(normalized)
    if resolved is None:
        raise typer.BadParameter("--modality must be auto, text, vision, or multimodal")
    if resolved != "auto":
        return resolved
    lowered = model.lower()
    if any(
        marker in lowered
        for marker in [
            "vl",
            "vision",
            "visual",
            "multimodal",
            "multi-modal",
            "llava",
            "minicpm-v",
        ]
    ):
        return "multimodal"
    if any(benchmark in selected_benchmarks for benchmark in MULTIMODAL_BENCHMARKS):
        return "multimodal"
    return "text"


def _input_modalities(selected_benchmarks: list[str]) -> list[str]:
    modalities = ["text"]
    if any(benchmark in selected_benchmarks for benchmark in MULTIMODAL_BENCHMARKS):
        modalities.append("image")
    return modalities


def _architecture_for_modality(modality: str) -> str:
    if modality in {"vision", "multimodal"}:
        return "multimodal transformer"
    return "decoder-only transformer"


def _model_metadata(provider_key: str, base_url: str, model: str) -> dict[str, Any] | None:
    if provider_key != "lmstudio" or model.lower() == "auto":
        return None
    try:
        with urllib.request.urlopen(
            f"{base_url.rstrip('/')}/api/v1/models",
            timeout=5,
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return None
    for item in payload.get("models") or []:
        if isinstance(item, dict) and item.get("key") == model:
            return item
    return None


def _model_metadata_value(metadata: dict[str, Any] | None, key: str) -> Any:
    return metadata.get(key) if metadata else None


def _display_model_name(
    provider: str,
    model: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    if model.lower() == "auto":
        return f"{provider} Served Model"
    if metadata and metadata.get("display_name"):
        return str(metadata["display_name"])
    return model


def _infer_base_model_name(
    provider: str,
    model: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    if model.lower() == "auto":
        return f"{provider} Served Model"
    if metadata and metadata.get("display_name"):
        return str(metadata["display_name"])
    size_match = re.search(r"(\d+(?:\.\d+)?)b\b", model, flags=re.IGNORECASE)
    if not size_match:
        m_match = re.search(r"(\d+(?:\.\d+)?)m\b", model, flags=re.IGNORECASE)
        if m_match:
            return f"{_infer_family(model, metadata)} {m_match.group(1)}M"
        return model
    qwen_match = re.search(r"qwen\d+(?:\.\d+)?", model, flags=re.IGNORECASE)
    if qwen_match:
        family = "Qwen" + qwen_match.group(0)[4:]
    else:
        family = _infer_family(model, metadata)
    return f"{family}-{size_match.group(1)}B"


def _infer_family(model: str, metadata: dict[str, Any] | None = None) -> str:
    publisher = str((metadata or {}).get("publisher") or "").lower()
    architecture = str((metadata or {}).get("architecture") or "").lower()
    lowered = model.lower()
    if "liquid" in publisher or "lfm" in lowered or "lfm" in architecture:
        return "Liquid AI"
    if "qwen" in lowered:
        return "Qwen"
    if "gemma" in lowered:
        return "Gemma"
    if "llama" in lowered:
        return "Llama"
    if "mistral" in lowered:
        return "Mistral"
    if "phi" in lowered:
        return "Phi"
    if "granite" in lowered:
        return "IBM"
    if "olmo" in lowered:
        return "AI2"
    if "falcon" in lowered:
        return "TII"
    if "smollm" in lowered:
        return "Hugging Face"
    if "nemotron" in lowered or "nvidia" in lowered:
        return "NVIDIA"
    if lowered.startswith("gpt-") or lowered.startswith("o3") or lowered.startswith("o4"):
        return "OpenAI"
    return "Local"


def _infer_parameter_size_b(model: str, metadata: dict[str, Any] | None = None) -> float:
    params = str((metadata or {}).get("params_string") or "")
    params_match = re.search(r"(\d+(?:\.\d+)?)([bm])\b", params, flags=re.IGNORECASE)
    if params_match:
        value = float(params_match.group(1))
        return value / 1000 if params_match.group(2).lower() == "m" else value
    match = re.search(r"(\d+(?:\.\d+)?)b\b", model, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    m_match = re.search(r"(\d+(?:\.\d+)?)m\b", model, flags=re.IGNORECASE)
    return float(m_match.group(1)) / 1000 if m_match else 0.0


def _infer_quantization(model: str, metadata: dict[str, Any] | None = None) -> str:
    quantization = (metadata or {}).get("quantization")
    if isinstance(quantization, dict) and quantization.get("name"):
        return str(quantization["name"]).upper()
    lowered = model.lower()
    match = re.search(r"\bq\d(?:_[a-z0-9]+)*\b", lowered)
    if match:
        return match.group(0).upper()
    if "bf16" in lowered:
        return "BF16"
    if "fp16" in lowered:
        return "FP16"
    if "nemotron-3-nano:4b" in lowered:
        return "Q4_K_M"
    bit_match = re.search(r"\b(\d+)\s*bit\b", lowered)
    if bit_match:
        return f"{bit_match.group(1)}BIT"
    return "SERVER"


def _infer_precision(model: str, quantization: str) -> str:
    lowered = model.lower()
    if "mlx" in lowered and quantization != "SERVER":
        return f"mlx-{quantization.lower()}"
    if quantization != "SERVER":
        return quantization.lower()
    return "server"


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value.lower()).strip("-")
    return slug or "auto"


def _run_with_progress(
    config: Path,
    dry_run: bool,
    *,
    auto_export: bool,
    auto_push: bool,
) -> dict[str, Any]:
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("[dim]{task.fields[details]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    variant_task_id: TaskID | None = None
    sample_task_id: TaskID | None = None
    sample_completed = 0
    sample_score = 0.0
    sample_invalid = 0
    sample_runtime = 0.0

    def progress_callback(event_type: str, payload: dict[str, Any]) -> None:
        nonlocal sample_completed, sample_score, sample_invalid, sample_runtime, sample_task_id
        nonlocal variant_task_id
        if event_type == "run_started":
            total = payload.get("variants_total") or 1
            progress.console.print(f"🚀 Starting benchmark run • {total} variant(s)")
            variant_task_id = progress.add_task(
                "🚀 Starting benchmark run",
                total=total,
                details="warming up",
            )
            return
        if variant_task_id is None:
            return
        if event_type == "variant_started":
            index = payload.get("index")
            total = payload.get("total")
            variant = payload.get("variant")
            progress.console.print(f"🧪 Variant {index}/{total}: [bold]{variant}[/bold]")
            progress.update(
                variant_task_id,
                description=f"🧪 {index}/{total} {variant}",
                details="starting",
            )
        elif event_type == "task_started":
            variant = payload.get("variant")
            task = payload.get("task")
            progress.console.print(f"🔬 Running {task} for [bold]{variant}[/bold]")
            progress.update(
                variant_task_id,
                description=f"🔬 {variant}: {task}",
                details="running",
            )
        elif event_type == "dataset_loading_started":
            languages = payload.get("languages") or []
            language_count = len(languages)
            progress.console.print(
                f"⬇️ Preparing {payload.get('dataset')} "
                f"({language_count} language(s), split={payload.get('split')})"
            )
            progress.update(
                variant_task_id,
                description=f"⬇️ Preparing dataset for {payload.get('variant')}",
                details=f"{language_count} language(s)",
            )
        elif event_type == "dataset_loading_completed":
            total_samples = payload.get("total_samples")
            progress.console.print(f"✅ Dataset ready • {total_samples} sample(s)")
            progress.update(
                variant_task_id,
                description=f"✅ Dataset ready for {payload.get('variant')}",
                details=f"{total_samples} sample(s)",
            )
        elif event_type == "task_sample_plan":
            if sample_task_id is not None:
                progress.remove_task(sample_task_id)
            total_samples = payload.get("total_samples") or 1
            sample_completed = 0
            sample_score = 0.0
            sample_invalid = 0
            sample_runtime = 0.0
            languages = ", ".join(payload.get("languages") or []) or "configured languages"
            variant = payload.get("variant")
            task_label = _task_label(str(payload.get("task") or "benchmark"))
            progress.console.print(
                f"🌍 Loaded {total_samples} sample(s) for [bold]{variant}[/bold] "
                f"({languages})"
            )
            progress.update(variant_task_id, visible=False)
            sample_task_id = progress.add_task(
                f"🌍 {variant}: {task_label}",
                total=total_samples,
                details=f"score 0/{total_samples} • acc 0.0% • {languages}",
            )
        elif event_type == "task_resume" and sample_task_id is not None:
            completed = int(payload.get("completed_samples") or 0)
            total = int(payload.get("total_samples") or 1)
            correct = float(payload.get("correct_samples") or 0.0)
            invalid = int(payload.get("invalid_samples") or 0)
            runtime_seconds = float(payload.get("runtime_seconds") or 0.0)
            sample_completed = completed
            sample_score = correct
            sample_invalid = invalid
            sample_runtime = runtime_seconds
            accuracy = sample_score / completed if completed else 0.0
            average_runtime = sample_runtime / completed if completed else 0.0
            progress.console.print(
                f"↩️ Resuming {payload.get('task')} • reused {completed}/{total} sample(s)"
            )
            progress.update(
                sample_task_id,
                completed=completed,
                description=f"🌍 {payload.get('variant')}: Q {completed}/{total}",
                details=(
                    f"cached • score {sample_score:.1f}/{completed} "
                    f"• acc {accuracy:.1%} • avg {average_runtime:.2f}s"
                ),
            )
        elif event_type == "task_progress" and sample_task_id is not None:
            completed = int(payload.get("completed_samples") or 0)
            total = int(payload.get("total_samples") or 1)
            latest_correct = bool(payload.get("latest_correct"))
            latest_answer = payload.get("latest_extracted_answer") or "?"
            latest_runtime = float(payload.get("latest_runtime_seconds") or 0.0)
            latest_score = payload.get("latest_score")
            latest_invalid = bool(payload.get("latest_invalid", latest_answer == "?"))
            score_increment = (
                float(latest_score) if latest_score is not None else float(int(latest_correct))
            )
            language = payload.get("language") or "?"

            sample_completed = completed
            sample_score += score_increment
            sample_invalid += int(latest_invalid)
            sample_runtime += latest_runtime
            accuracy = sample_score / completed if completed else 0.0
            average_runtime = sample_runtime / completed if completed else 0.0
            result_icon = (
                "✅"
                if score_increment >= 0.999
                else ("⚠️" if score_increment > 0 or latest_answer == "?" else "❌")
            )
            progress.update(
                sample_task_id,
                completed=completed,
                total=total,
                description=f"🌍 {payload.get('variant')}: Q {completed}/{total}",
                details=(
                    f"{result_icon} last={latest_answer} • score {sample_score:.1f}/{completed} "
                    f"• acc {accuracy:.1%} • avg {average_runtime:.2f}s"
                ),
            )
            if _should_log_sample(completed, total):
                invalid_text = f" • invalid {sample_invalid}" if sample_invalid else ""
                progress.console.print(
                    f"{result_icon} Q {completed}/{total} lang={language} "
                    f"answer={latest_answer} • score {sample_score:.1f}/{completed} "
                    f"({accuracy:.1%}) • {latest_runtime:.2f}s{invalid_text}"
                )
        elif event_type == "runtime_cache_reset":
            language = payload.get("language")
            instance_id = payload.get("instance_id") or payload.get("model")
            completed = payload.get("completed_samples")
            reason = payload.get("reason")
            cooldown = payload.get("cooldown_seconds") or 0
            reason_text = (
                f"after {completed} call(s)"
                if reason == "call_interval" and completed is not None
                else f"after lang={language}"
            )
            cooldown_text = f" • cooldown {cooldown:g}s" if cooldown else ""
            if payload.get("error"):
                progress.console.print(
                    f"⚠️ Could not restart model runtime {reason_text}: "
                    f"{payload.get('error')}"
                )
            else:
                progress.console.print(
                    f"🧹 Restarted model runtime {reason_text} • {instance_id}{cooldown_text}"
                )
        elif event_type == "task_completed":
            variant = payload.get("variant")
            task = payload.get("task")
            if sample_task_id is not None:
                if sample_completed:
                    accuracy = sample_score / sample_completed
                    average_runtime = sample_runtime / sample_completed
                    invalid_text = f" • invalid {sample_invalid}" if sample_invalid else ""
                    progress.console.print(
                        f"🏁 {task} complete • score {sample_score:.1f}/{sample_completed} "
                        f"({accuracy:.1%}) • avg {average_runtime:.2f}s{invalid_text}"
                    )
                progress.remove_task(sample_task_id)
                sample_task_id = None
            progress.update(
                variant_task_id,
                description=f"✅ {variant}: {task}",
                details="task complete",
                visible=True,
            )
        elif event_type == "benchmark_saved":
            task = payload.get("task")
            metrics_written = payload.get("metrics_written")
            normalized_rows = payload.get("normalized_rows")
            exported_results = []
            if auto_export and not dry_run:
                exported_results = _export_for_benchmark_payload(payload)
                rows = exported_results[0]["rows"] if exported_results else normalized_rows
                progress.console.print(
                    f"💾 Saved {task} • {metrics_written} metric(s) • "
                    f"{rows} website row(s)"
                )
            else:
                progress.console.print(
                    f"💾 Saved {task} • {metrics_written} metric(s) • "
                    f"{normalized_rows} normalized row(s)"
                )
            if auto_push and not dry_run:
                _safe_commit_and_push_results(payload, exported_results)
        elif event_type == "variant_skipped":
            reason = payload.get("reason")
            progress.console.print(f"⏭️ Skipped {payload.get('variant')} ({reason})")
            progress.update(
                variant_task_id,
                description=f"⏭️ Skipped {payload.get('variant')}",
                details=str(reason),
            )
            progress.advance(variant_task_id)
        elif event_type == "variant_completed":
            metrics_written = payload.get("metrics_written")
            progress.console.print(
                f"✅ Completed [bold]{payload.get('variant')}[/bold] • "
                f"{metrics_written} metric(s)"
            )
            progress.update(
                variant_task_id,
                description=f"✅ Completed {payload.get('variant')}",
                details=f"{metrics_written} metric(s)",
            )
            progress.advance(variant_task_id)
        elif event_type == "normalization_started":
            progress.console.print("📊 Normalizing leaderboard rows")
            progress.update(
                variant_task_id,
                description="📊 Normalizing results",
                details="database",
            )
        elif event_type == "normalization_completed":
            progress.update(
                variant_task_id,
                description="📊 Normalized results",
                details=f"{payload.get('normalized_rows')} row(s)",
            )
        elif event_type == "run_failed":
            progress.console.print(f"💥 Run failed: {payload.get('error')}")
            progress.update(
                variant_task_id,
                description="💥 Run failed",
                details="error",
            )
        elif event_type == "run_completed":
            progress.console.print(
                f"🎉 Run complete • {payload.get('variants_seen')} variant(s) • "
                f"{payload.get('metrics_written')} metric(s) • "
                f"{payload.get('normalized_rows')} normalized row(s)"
            )
            progress.update(
                variant_task_id,
                description="🎉 Completed benchmark run",
                details=f"{payload.get('metrics_written')} metric(s)",
            )

    with progress:
        return run_benchmark_with_progress(
            config,
            dry_run=dry_run,
            progress_callback=progress_callback,
        )


def _should_log_sample(completed: int, total: int) -> bool:
    if total <= 25:
        return True
    if completed in {1, total}:
        return True
    return completed % max(1, total // 10) == 0


def _task_label(task: str) -> str:
    labels = {
        "global-mmlu-lite": "Global MMLU Lite",
        "ifbench": "IFBench",
        "bfcl-v4": "BFCL v4",
        "ocrbench-v2": "OCRBench v2",
        "mmmu": "MMMU",
        "mbpp": "MBPP",
        "rgb": "RGB",
        "simpleqa": "SimpleQA",
        "harmbench": "HarmBench",
    }
    return labels.get(task, task)
