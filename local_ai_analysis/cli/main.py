from __future__ import annotations

import re
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
from local_ai_analysis.db import LocalAIAnalysisDB
from local_ai_analysis.export import export_leaderboard, leaderboard_payload
from local_ai_analysis.normalization import refresh_normalized_results
from local_ai_analysis.runner import run_benchmark, run_benchmark_with_progress

app = typer.Typer(
    name="laia",
    help="Local AI Analysis: local API benchmarks for local model servers.",
    no_args_is_help=True,
)
console = Console()

GLOBAL_MMLU_LITE_REVISION = "cbf2f73663ff201d4d56e891c8c2c18467aeea06"
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
) -> None:
    """Run benchmark jobs from a YAML config file."""
    _execute_run(config, dry_run=dry_run, no_progress=no_progress, auto_export=auto_export)


@app.command("ollama")
def run_ollama_shortcut(
    model: Annotated[
        str,
        typer.Argument(help="Ollama model tag, for example qwen3.5:0.8b-mlx-bf16."),
    ],
    smoke: Annotated[
        bool,
        typer.Option("--smoke", help="Run only 5 English questions instead of the full suite."),
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
            help="Benchmark to run: global-mmlu-lite, ifbench, bfcl, or all.",
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
    max_tokens: Annotated[
        int | None,
        typer.Option("--max-tokens", help="Override max generated tokens for the benchmark."),
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
        reasoning_effort=reasoning_effort,
        max_tokens=max_tokens,
    )
    console.print(f"🧾 Generated config: {config}")
    _execute_run(config, dry_run=dry_run, no_progress=no_progress, auto_export=auto_export)


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
        typer.Option("--smoke", help="Run only 5 English questions instead of the full suite."),
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
            help="Benchmark to run: global-mmlu-lite, ifbench, bfcl, or all.",
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
    max_tokens: Annotated[
        int | None,
        typer.Option("--max-tokens", help="Override max generated tokens for the benchmark."),
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
        reasoning_effort=reasoning_effort,
        max_tokens=max_tokens,
    )
    console.print(f"🧾 Generated config: {config}")
    _execute_run(config, dry_run=dry_run, no_progress=no_progress, auto_export=auto_export)


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
        typer.Option("--smoke", help="Run only 5 English questions instead of the full suite."),
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
            help="Benchmark to run: global-mmlu-lite, ifbench, bfcl, or all.",
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
    max_tokens: Annotated[
        int | None,
        typer.Option("--max-tokens", help="Override max generated tokens for the benchmark."),
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
        reasoning_effort=reasoning_effort,
        max_tokens=max_tokens,
    )
    console.print(f"🧾 Generated config: {config}")
    _execute_run(config, dry_run=dry_run, no_progress=no_progress, auto_export=auto_export)


def _execute_run(
    config: Path,
    *,
    dry_run: bool,
    no_progress: bool,
    auto_export: bool,
) -> None:
    try:
        if no_progress:
            result = run_benchmark(config, dry_run=dry_run)
        else:
            result = _run_with_progress(config, dry_run)
    except RuntimeError as exc:
        if no_progress:
            console.print(f"💥 [red]Run failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    exported_results = (
        _export_website_results(Path(str(result["db_path"])))
        if auto_export and not dry_run and result.get("db_path")
        else []
    )
    console.print("✅ [green]Run complete[/green]")
    for key, value in result.items():
        console.print(f"{key}: {value}")
    for exported in exported_results:
        console.print(f"🌐 Exported website data: {exported['rows']} rows to {exported['out']}")


@app.command()
def normalize(
    db: Annotated[
        Path,
        typer.Option("--db", help="DuckDB database path."),
    ] = Path("results/local_ai_analysis.duckdb"),
) -> None:
    """Recompute normalized Global MMLU Lite rows."""
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
        "GMMLU Lite",
        "IFBench",
        "BFCL v4",
        "Invalid",
        "Runtime",
        "Run",
    ]:
        table.add_column(column)

    for row in payload["leaderboard"]:
        table.add_row(
            str(row.get("variant_name") or ""),
            str(row.get("backend_name") or "n/a"),
            _pct(row.get("global_mmlu_lite_pass_at_1")),
            _pct(row.get("ifbench_prompt_level_loose")),
            _pct(row.get("bfcl_v4_selected_accuracy")),
            _pct(row.get("global_mmlu_lite_invalid_rate")),
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
    reasoning_effort: str | None,
    max_tokens: int | None,
) -> Path:
    resolved_reasoning_effort = _resolve_reasoning_effort(model, reasoning_effort)
    selected_benchmarks = _parse_benchmarks(benchmark)
    provider_key = _provider_key(provider)
    selected_languages = ["en"] if smoke else _parse_languages(languages)
    sample_limit = 5 if smoke else None
    all_languages_selected = selected_languages == GLOBAL_MMLU_LITE_LANGUAGES
    benchmark_slug = _slug("-".join(selected_benchmarks))
    provider_slug = _slug(provider)
    model_slug = _slug(model)
    scope_slug = (
        "smoke"
        if smoke
        else ("full" if all_languages_selected else _slug("-".join(selected_languages)))
    )
    reasoning_slug = _slug(f"reasoning-{resolved_reasoning_effort or 'unset'}")
    output_path = (
        Path("results/generated_configs")
        / f"{provider_slug}_{model_slug}_{benchmark_slug}_{scope_slug}_{reasoning_slug}.yaml"
    )
    display_model = _display_model_name(provider, model)
    provider_label = provider if provider != "LM Studio" else "LM Studio"
    variant_suffix = (
        "Smoke"
        if smoke
        else ("All Languages" if all_languages_selected else "Selected Languages")
    )
    reasoning_label = resolved_reasoning_effort or "unset"
    quantization = _infer_quantization(model)

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
        "parser_version": "global_mmlu_lite_regex_v1",
        "prompt_template": GLOBAL_MMLU_LITE_PROMPT,
    }
    if resolved_reasoning_effort:
        global_mmlu_lite["reasoning_effort"] = resolved_reasoning_effort

    ifbench: dict[str, Any] = {
        "enabled": "ifbench" in selected_benchmarks,
        "dataset_name": "allenai/IFBench_test",
        "dataset_revision": None,
        "split": "train",
        "sample_limit": sample_limit,
        "output_dir": "results/ifbench",
        "provider": provider_key,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "timeout_seconds": 240,
        "temperature": 0.01,
        "max_tokens": max_tokens if max_tokens is not None else 4096,
        "top_p": 0.95,
        "stop": None,
        "seed": 42,
        "strip_thinking": True,
        "evaluator": "allenai_ifbench_loose_v1",
    }
    if resolved_reasoning_effort:
        ifbench["reasoning_effort"] = resolved_reasoning_effort

    bfcl_v4: dict[str, Any] = {
        "enabled": "bfcl-v4" in selected_benchmarks,
        "version": "BFCL_v4",
        "categories": _parse_bfcl_categories(bfcl_categories),
        "sample_limit": sample_limit,
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
        "include_input_log": False,
        "exclude_state_log": True,
        "evaluator": "bfcl_eval_prompt_mode_v4",
    }
    if resolved_reasoning_effort:
        bfcl_v4["reasoning_effort"] = resolved_reasoning_effort

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
            "command": f"{provider_label} local server",
        },
        "global_mmlu_lite": global_mmlu_lite,
        "ifbench": ifbench,
        "bfcl_v4": bfcl_v4,
        "models": [
            {
                "family": _infer_family(model),
                "name": _infer_base_model_name(provider, model),
                "parameter_size_b": _infer_parameter_size_b(model),
                "architecture": "decoder-only transformer",
                "license": "replace-with-upstream-license",
                "source_url": base_url,
                "variants": [
                    {
                        "name": (
                            f"{display_model} {provider_label} {variant_suffix} "
                            f"Reasoning {reasoning_label}"
                        ),
                        "quantization": quantization,
                        "precision": _infer_precision(model, quantization),
                        "baseline": False,
                        "api_model": model,
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
    }
    if normalized in {"all", "both"}:
        return ["global-mmlu-lite", "ifbench", "bfcl-v4"]
    selected = [aliases.get(item.strip(), item.strip()) for item in normalized.split(",")]
    selected = [item for item in selected if item]
    unknown = sorted(set(selected) - {"global-mmlu-lite", "ifbench", "bfcl-v4"})
    if unknown:
        raise typer.BadParameter(
            f"Unsupported benchmark(s): {', '.join(unknown)}. "
            "Use global-mmlu-lite, ifbench, bfcl, or all."
        )
    if not selected:
        raise typer.BadParameter("--benchmark must be global-mmlu-lite, ifbench, bfcl, or all")
    return selected


def benchmark_label(benchmarks: list[str]) -> str:
    labels = {
        "global-mmlu-lite": "Global MMLU Lite",
        "ifbench": "IFBench",
        "bfcl-v4": "BFCL v4",
    }
    return " + ".join(labels[item] for item in benchmarks)


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
    return _slug(provider)


def _resolve_reasoning_effort(model: str, value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"", "unset", "none"}:
        return "none"
    if normalized == "auto":
        return "none" if "qwen" in model.lower() else "high"
    return normalized


def _display_model_name(provider: str, model: str) -> str:
    if model.lower() == "auto":
        return f"{provider} Served Model"
    return model


def _infer_base_model_name(provider: str, model: str) -> str:
    if model.lower() == "auto":
        return f"{provider} Served Model"
    size_match = re.search(r"(\d+(?:\.\d+)?)b\b", model, flags=re.IGNORECASE)
    if not size_match:
        return model
    qwen_match = re.search(r"qwen\d+(?:\.\d+)?", model, flags=re.IGNORECASE)
    if qwen_match:
        family = "Qwen" + qwen_match.group(0)[4:]
    else:
        family = _infer_family(model)
    return f"{family}-{size_match.group(1)}B"


def _infer_family(model: str) -> str:
    lowered = model.lower()
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
    return "Local"


def _infer_parameter_size_b(model: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)b\b", model, flags=re.IGNORECASE)
    return float(match.group(1)) if match else 0.0


def _infer_quantization(model: str) -> str:
    lowered = model.lower()
    if "bf16" in lowered:
        return "BF16"
    if "fp16" in lowered:
        return "FP16"
    bit_match = re.search(r"\b(\d+)\s*bit\b", lowered)
    if bit_match:
        return f"{bit_match.group(1)}BIT"
    match = re.search(r"\bq\d(?:_[a-z0-9]+)*\b", lowered)
    return match.group(0).upper() if match else "SERVER"


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


def _run_with_progress(config: Path, dry_run: bool) -> dict[str, Any]:
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
    sample_correct = 0
    sample_invalid = 0
    sample_runtime = 0.0

    def progress_callback(event_type: str, payload: dict[str, Any]) -> None:
        nonlocal sample_completed, sample_correct, sample_invalid, sample_runtime, sample_task_id
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
            sample_correct = 0
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
        elif event_type == "task_progress" and sample_task_id is not None:
            completed = int(payload.get("completed_samples") or 0)
            total = int(payload.get("total_samples") or 1)
            latest_correct = bool(payload.get("latest_correct"))
            latest_answer = payload.get("latest_extracted_answer") or "?"
            latest_runtime = float(payload.get("latest_runtime_seconds") or 0.0)
            language = payload.get("language") or "?"

            sample_completed = completed
            sample_correct += int(latest_correct)
            sample_invalid += int(latest_answer == "?")
            sample_runtime += latest_runtime
            accuracy = sample_correct / completed if completed else 0.0
            average_runtime = sample_runtime / completed if completed else 0.0
            result_icon = "✅" if latest_correct else ("⚠️" if latest_answer == "?" else "❌")
            progress.update(
                sample_task_id,
                completed=completed,
                total=total,
                description=f"🌍 {payload.get('variant')}: Q {completed}/{total}",
                details=(
                    f"{result_icon} last={latest_answer} • score {sample_correct}/{completed} "
                    f"• acc {accuracy:.1%} • avg {average_runtime:.2f}s"
                ),
            )
            if _should_log_sample(completed, total):
                invalid_text = f" • invalid {sample_invalid}" if sample_invalid else ""
                progress.console.print(
                    f"{result_icon} Q {completed}/{total} lang={language} "
                    f"answer={latest_answer} • score {sample_correct}/{completed} "
                    f"({accuracy:.1%}) • {latest_runtime:.2f}s{invalid_text}"
                )
        elif event_type == "task_completed":
            variant = payload.get("variant")
            task = payload.get("task")
            if sample_task_id is not None:
                if sample_completed:
                    accuracy = sample_correct / sample_completed
                    average_runtime = sample_runtime / sample_completed
                    invalid_text = f" • invalid {sample_invalid}" if sample_invalid else ""
                    progress.console.print(
                        f"🏁 {task} complete • score {sample_correct}/{sample_completed} "
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
    }
    return labels.get(task, task)
