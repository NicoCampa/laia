# SimpleQA

SimpleQA is OpenAI's short-form factuality benchmark. It asks concise,
fact-seeking questions with reference answers and grades model responses as
`correct`, `incorrect`, or `not_attempted`. In this project, `incorrect` is also
reported as `simpleqa_hallucination_rate` because those are attempted factual
answers judged to contradict the reference.

Run a smoke test:

```bash
laia ollama qwen3.5:0.8b-mlx-bf16 --benchmark simpleqa --smoke
laia lmstudio exact-model-id --benchmark simpleqa --smoke
laia omlx exact-model-id --benchmark simpleqa --smoke
```

Full run:

```bash
laia ollama qwen3.5:0.8b-mlx-bf16 --benchmark simpleqa
```

Shortcut-generated full SimpleQA runs use 500 deterministic stratified questions
by topic and answer type (`sample_strategy=stratified`, `sample_seed=42`). Each
question still makes one model call and one grader call when `--simpleqa-grader`
is `llm`. Set `simpleqa.sample_limit: null` in an advanced config when you
intentionally want the full SimpleQA CSV.

By default, SimpleQA uses an LLM judge with the same local model:

```bash
laia ollama qwen3.5:0.8b-mlx-bf16 --benchmark simpleqa --simpleqa-grader-model same
```

For publishable rows, use a stronger pinned judge model served by the same local
backend:

```bash
laia ollama small-model --benchmark simpleqa --simpleqa-grader-model stronger-local-judge
```

You can also run a deterministic lexical fallback:

```bash
laia ollama qwen3.5:0.8b-mlx-bf16 --benchmark simpleqa --simpleqa-grader heuristic
```

The heuristic is useful for smoke tests and offline debugging, but it is not the
official SimpleQA scoring method because it cannot reliably handle aliases,
partial names, numeric tolerances, or semantic equivalence.

Leaderboard-facing metrics:

- `simpleqa_f1`: OpenAI-style F1 over correctness and accuracy-given-attempted.
- `simpleqa_correct_rate`: fraction of all questions graded correct.
- `simpleqa_incorrect_rate`: fraction of all questions graded incorrect.
- `simpleqa_hallucination_rate`: alias of incorrect rate for factuality reporting.
- `simpleqa_not_attempted_rate`: fraction of all questions where the model declined
  or did not provide the reference answer without contradiction.
- `simpleqa_accuracy_given_attempted`: correct divided by attempted answers.

SimpleQA is intentionally excluded from `model_intelligence_score` because it
requires a judge. Treat it as a separate factuality and hallucination diagnostic.

Summary metrics are written to `results/simpleqa/<variant>/summary.json`.
Per-sample records, including questions, gold answers, raw model outputs, grader
outputs, runtime, and usage when available, are written to
`results/simpleqa/<variant>/samples.jsonl`.

Primary references:

- [OpenAI SimpleQA announcement](https://openai.com/index/introducing-simpleqa/)
- [OpenAI simple-evals reference implementation](https://github.com/openai/simple-evals)
