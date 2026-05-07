# OCRBench v2

OCRBench v2 is a bilingual visual text benchmark for multimodal models. Local AI
Analysis loads the Hugging Face dataset `morpheushoc/OCRBenchv2`, sends each image
plus question to the selected native local API, stores per-sample predictions, and
normalizes leaderboard metrics.

## Commands

OCRBench v2 requires a vision-capable model. Text-only Qwen, Llama, or Mistral
models will usually fail or ignore the image.

```bash
laia ollama qwen3-vl:8b --benchmark ocrbench --smoke
laia lmstudio exact-vision-model-id --benchmark ocrbench --smoke
laia omlx exact-vision-model-id --benchmark ocrbench --smoke
```

Full OCRBench v2 uses the official English and Chinese aggregate configs:

```bash
laia omlx exact-vision-model-id --benchmark ocrbench --ocrbench-configs EN,CN
```

Run only one subset:

```bash
laia omlx exact-vision-model-id --benchmark ocrbench --ocrbench-configs "text recognition en"
```

Smoke mode always uses 5 samples from `text recognition en`.

## Native Image Payloads

- Ollama receives base64 images on the native `/api/chat` message `images` field.
- LM Studio receives native `/api/v1/chat` input items with `{type: "image", data_url}`.
- oMLX receives OpenAI-style vision content blocks on `/v1/chat/completions`.

## Metrics

Leaderboard-facing metrics:

- `ocrbench_v2_score`: macro average of available English and Chinese group scores.
- `ocrbench_v2_micro_score`: sample-weighted average over selected samples.
- `ocrbench_v2_en_score`
- `ocrbench_v2_cn_score`
- `ocrbench_v2_samples`
- `ocrbench_v2_runtime_seconds`

Per-sample output is written to `results/ocrbench_v2/<variant>/samples.jsonl`.
Summary metrics are written to `results/ocrbench_v2/<variant>/summary.json`.

The local evaluator mirrors the official OCRBench v2 grouping and implements the
common VQA, ANLS, formula, counting, and bounding-box IoU checks. Complex structural
metrics such as table TEDS are represented by the local evaluator version recorded
as `ocrbench_v2_local_vqa_anls_iou_v1`, so compare those rows only against other
rows produced by the same LAIA evaluator.
