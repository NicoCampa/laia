# HarmBench

HarmBench is a safety red-team benchmark from the Center for AI Safety. Local AI
Analysis runs the text behavior set against the served model, then classifies each
completion with a local judge model.

Smoke test:

```bash
laia ollama qwen3.5:0.8b-mlx-bf16 --benchmark harmbench --smoke
laia lmstudio exact-model-id --benchmark harmbench --smoke
laia omlx exact-model-id --benchmark harmbench --smoke
```

Full run:

```bash
laia ollama qwen3.5:0.8b-mlx-bf16 --benchmark harmbench
```

By default, HarmBench uses `standard,contextual` functional categories and skips
the copyright slice:

```bash
laia ollama qwen3.5:0.8b-mlx-bf16 --benchmark harmbench --harmbench-categories standard,contextual
```

Opt into every text category only when you explicitly want that scope:

```bash
laia ollama qwen3.5:0.8b-mlx-bf16 --benchmark harmbench --harmbench-categories all
```

The default judge is an LLM judge using the same local model convention as
SimpleQA:

```bash
laia ollama qwen3.5:0.8b-mlx-bf16 --benchmark harmbench --harmbench-judge-model same
```

For publishable rows, serve a stronger pinned judge and use the same judge model
for every judge-based benchmark you compare:

```bash
laia ollama small-model --benchmark harmbench --harmbench-judge-model stronger-local-judge
```

There is also a deterministic heuristic fallback:

```bash
laia ollama qwen3.5:0.8b-mlx-bf16 --benchmark harmbench --harmbench-judge heuristic
```

The heuristic is useful for smoke tests and offline debugging, but it is not the
official HarmBench classifier and should not be used for publishable safety claims.

Leaderboard-facing metrics:

- `harmbench_attack_success_rate`: fraction of behaviors where the judge marked
  the model response as a successful harmful behavior. Lower is better.
- `harmbench_refusal_rate`: `1 - attack_success_rate`. Higher is better.

Supporting metrics include sample count, answer runtime, judge runtime, total
runtime, and token usage when the native backend reports usage.

Summary metrics are written to `results/harmbench/<variant>/summary.json`.
Per-sample records, including behavior IDs, categories, raw model outputs, judge
outputs, runtime, and usage when available, are written to
`results/harmbench/<variant>/samples.jsonl`.

HarmBench is intentionally excluded from `model_intelligence_score`; it is a
safety/refusal measurement, not a capability score.

Primary references:

- [HarmBench paper](https://arxiv.org/abs/2402.04249)
- [HarmBench repository](https://github.com/centerforaisafety/HarmBench)
- [HarmBench behavior CSV](https://github.com/centerforaisafety/HarmBench/blob/main/data/behavior_datasets/harmbench_behaviors_text_all.csv)
