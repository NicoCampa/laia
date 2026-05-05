# Global MMLU Lite Generation Benchmark

Local AI Analysis benchmarks native Ollama, LM Studio, and oMLX API servers with
Global MMLU Lite. The runner sends each multiple-choice question to the served model,
extracts one answer letter, and reports pass@1.

## Protocol

The dataset is loaded with:

```python
load_dataset("CohereLabs/Global-MMLU-Lite", language, split="test")
```

The default full run uses all supported language configs:

```text
ar, bn, cy, de, en, es, fr, hi, id, it, ja, ko, my, pt, sq, sw, yo, zh
```

Smoke mode evaluates 5 English questions.

The prompt asks the model to reply with only `A`, `B`, `C`, or `D`. If
`strip_thinking: true`, `<think>`, `<thinking>`, and `<reasoning>` blocks are removed
before parsing.

## Metrics

Primary metric:

```text
global_mmlu_lite_pass_at_1
```

Additional metrics:

- `global_mmlu_lite_micro_pass_at_1`
- `global_mmlu_lite_invalid_rate`
- `global_mmlu_lite_samples`
- `global_mmlu_lite_runtime_seconds`
- `global_mmlu_lite_pass_at_1_<language>`
- `global_mmlu_lite_pass_at_1_<cultural_sensitivity_label>`

Per-question JSONL records include prompt, raw output, parsed answer, gold answer,
correctness, runtime, raw response, and API usage when the server returns it.

## Commands

```bash
laia ollama qwen3.5:0.8b-mlx-bf16
laia ollama qwen3.5:0.8b-mlx-bf16 --smoke
laia ollama qwen3.5:0.8b-mlx-bf16 --languages en,it
```

```bash
laia lmstudio
laia lmstudio exact-model-id
laia lmstudio --smoke
```

```bash
omlx serve --model-dir /Users/nicolocampagnoli/.lmstudio/models
export OMLX_API_KEY=your-omlx-api-key
laia omlx
laia omlx Qwen3.5-9B-4bit
laia omlx --smoke
```

Ollama requires a model tag and uses `/api/chat`. LM Studio defaults to `auto`, which
uses the first model returned by `/api/v1/models`; pin a model id for publishable runs.
oMLX defaults to `auto`, uses `/v1/models` and `/v1/chat/completions`, and can reuse
the LM Studio model directory when started with `omlx serve --model-dir`.

Shortcut commands default to `--reasoning-effort none`. Override it when needed:

```bash
laia ollama qwen3.5:0.8b-mlx-bf16 --reasoning-effort high
laia lmstudio exact-model-id --reasoning-effort high
laia omlx exact-model-id --reasoning-effort high
```
