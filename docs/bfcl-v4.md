# BFCL v4

BFCL v4 is Berkeley's function-calling benchmark. Local AI Analysis runs it in
prompt mode through the same native Ollama, LM Studio, and oMLX clients used by
the other benchmarks.

This is the standard BFCL scoring path for local/chat endpoints: the model is
given the function schemas and the benchmark checks whether the emitted function
call name and arguments match the expected call. LAIA does not execute arbitrary
tools or run a full agent loop for the default BFCL score. BFCL's heavier
multi-turn and agentic categories can involve state, memory, and web-search style
tasks, but they still require explicit category selection and more upstream setup.

## Install

The upstream `bfcl-eval` package currently pins `numpy==1.26.4`, which is not a
good default on Python 3.13. Install the evaluator package without dependencies,
then use this project's `eval` extra for the parser helpers:

```bash
pip install -e ".[eval]"
pip install --no-deps bfcl-eval==2026.3.23
```

## Commands

Smoke test:

```bash
laia ollama qwen3.5:0.8b-mlx-bf16 --benchmark bfcl --smoke
laia lmstudio exact-model-id --benchmark bfcl --smoke
laia omlx Qwen3.5-9B-4bit --benchmark bfcl --smoke
```

Run the default local BFCL set:

```bash
laia omlx Qwen3.5-9B-4bit --benchmark bfcl
```

Select BFCL categories:

```bash
laia omlx Qwen3.5-9B-4bit --benchmark bfcl --bfcl-categories non_live
laia omlx Qwen3.5-9B-4bit --benchmark bfcl --bfcl-categories live
laia omlx Qwen3.5-9B-4bit --benchmark bfcl --bfcl-categories single_turn
```

The generated config stores BFCL settings under `bfcl_v4`.

## Categories

The default shortcut category is `single_turn`. Useful aliases accepted by
Berkeley's parser include:

- `single_turn`
- `non_live`
- `live`
- `multi_turn`
- `agentic`
- `all_scoring`

The `agentic` categories exercise web search and memory backends. They are much
heavier than single-turn AST tests and can require extra upstream dependencies
and external search configuration.

## Metrics

Primary metric:

- `bfcl_v4_selected_accuracy`: correct samples divided by selected samples.

Additional metrics:

- `bfcl_v4_invalid_rate`
- `bfcl_v4_non_live_accuracy`
- `bfcl_v4_live_accuracy`
- `bfcl_v4_multi_turn_accuracy`
- `bfcl_v4_agentic_accuracy`
- `bfcl_v4_samples`
- `bfcl_v4_runtime_seconds`

Per-sample output is written to `results/bfcl_v4/<variant>/samples.jsonl`.
Summary metrics are written to `results/bfcl_v4/<variant>/summary.json`.
