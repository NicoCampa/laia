# Local AI Analysis

Global MMLU Lite, IFBench, and BFCL v4 benchmarks for local AI models served through native
Ollama, LM Studio, and oMLX APIs.

The project is intentionally narrow: it benchmarks local API servers with deterministic
generation protocols, records raw prompts and responses, normalizes results into DuckDB,
and publishes a React leaderboard.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[eval]"
```

The `eval` extra installs Hugging Face `datasets`, the pinned AllenAI IFBench
evaluator, and parser helpers used by BFCL.

BFCL's upstream package currently pins an old NumPy that is awkward on Python 3.13.
Install the package itself without dependencies:

```bash
pip install --no-deps bfcl-eval==2026.3.23
```

## Run Benchmarks

Ollama requires an explicit model tag:

```bash
laia ollama qwen3.5:0.8b-mlx-bf16
```

LM Studio can use the first served model, but pinning the model id is recommended for
publishable runs:

```bash
laia lmstudio
laia lmstudio exact-model-id
```

oMLX can reuse your LM Studio MLX model directory. Start oMLX, then benchmark the
first discovered model or pin an exact model id:

```bash
omlx serve --model-dir /Users/nicolocampagnoli/.lmstudio/models
export OMLX_API_KEY=your-omlx-api-key
laia omlx
laia omlx Qwen3.5-9B-4bit
```

Quick checks:

```bash
laia ollama qwen3.5:0.8b-mlx-bf16 --smoke
laia lmstudio --smoke
laia omlx --smoke
```

Useful options:

```bash
laia ollama qwen3.5:0.8b-mlx-bf16 --languages en,it
laia ollama qwen3.5:0.8b-mlx-bf16 --benchmark ifbench --smoke
laia ollama qwen3.5:0.8b-mlx-bf16 --benchmark bfcl --smoke
laia omlx Qwen3.5-9B-4bit --benchmark bfcl --bfcl-categories non_live
laia omlx Qwen3.5-9B-4bit --benchmark ifbench
laia lmstudio exact-model-id --benchmark all
laia ollama qwen3.5:0.8b-mlx-bf16 --reasoning-effort high
laia ollama qwen3.5:0.8b-mlx-bf16 --dry-run --no-auto-export
laia lmstudio exact-model-id --base-url http://127.0.0.1:1234
laia lmstudio exact-model-id --reasoning-effort high
laia omlx exact-model-id --base-url http://127.0.0.1:8000
laia omlx exact-model-id --api-key-env OMLX_API_KEY
```

Shortcut commands generate reproducible YAML configs under `results/generated_configs/`
and then run the normal benchmark pipeline. Successful non-dry runs refresh:

- `public/results.json`
- `web/public/results.json`
- `web/dist/results.json` when `web/dist` exists

## Website

Run the dashboard:

```bash
cd web
npm install
npm run dev
```

The dashboard reads `web/public/results.json` by default. To use the FastAPI backend:

```bash
laia serve --db results/local_ai_analysis.duckdb --port 8000
VITE_API_URL=http://127.0.0.1:8000 npm run dev
```

## CLI

```bash
laia ollama MODEL
laia lmstudio [MODEL]
laia omlx [MODEL]
laia run --config results/generated_configs/some-run.yaml
laia leaderboard --db results/local_ai_analysis.duckdb
laia export --format json --out web/public/results.json
laia normalize --db results/local_ai_analysis.duckdb
laia serve --db results/local_ai_analysis.duckdb
```

`laia run --config` is kept for advanced usage with generated configs. The supported
benchmark tasks are Global MMLU Lite generation pass@1, IFBench instruction following,
and BFCL v4 prompt-mode function calling.

## Outputs

Local benchmark outputs are kept under `results/`:

- DuckDB database: `results/local_ai_analysis.duckdb`
- Raw run events: `results/raw_results.jsonl`
- Per-sample Global MMLU Lite JSONL and summaries: `results/global_mmlu_lite/`
- Per-sample IFBench JSONL and summaries: `results/ifbench/`
- Per-sample BFCL v4 JSONL and summaries: `results/bfcl_v4/`
- Generated configs: `results/generated_configs/`

The repository does not include synthetic benchmark rows or fake sample data.

## Reproducibility

Shortcut commands default to `--reasoning-effort none`. Ollama maps that to native
`think: false`; LM Studio maps it to native `reasoning: "off"` when the model exposes
reasoning controls; oMLX maps it to `chat_template_kwargs.enable_thinking=false`.
Override it with `--reasoning-effort low`, `medium`, `high`, or `auto`.

Every run records:

- provider and native API base URL
- model id or model tag
- dataset name, revision, split, and language list
- prompt template, parser version, decoding parameters, seed, and reasoning control
- raw prompt/output JSONL with per-question runtime and API usage when available
- backend, hardware, command arguments, and run UUID

See [docs/global-mmlu-lite.md](docs/global-mmlu-lite.md), [docs/ifbench.md](docs/ifbench.md),
[docs/bfcl-v4.md](docs/bfcl-v4.md), [docs/data-model.md](docs/data-model.md), and
[docs/reproducibility.md](docs/reproducibility.md).
