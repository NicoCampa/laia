# MMMU

MMMU is a college-level multimodal benchmark covering 30 subjects across six
domains. Local AI Analysis loads `MMMU/MMMU` from Hugging Face, sends the attached
images plus question to the selected native local API, parses the model response,
and stores per-sample predictions.

## Commands

MMMU requires a vision-capable model.

```bash
laia ollama qwen3-vl:8b --benchmark mmmu --smoke
laia lmstudio exact-vision-model-id --benchmark mmmu --smoke
laia omlx exact-vision-model-id --benchmark mmmu --smoke
```

Full local runs default to the official validation split across all 30 subjects:

```bash
laia omlx exact-vision-model-id --benchmark mmmu
```

Run a selected subset:

```bash
laia omlx exact-vision-model-id --benchmark mmmu --mmmu-subjects Accounting,Math
```

Run the released test split locally:

```bash
laia omlx exact-vision-model-id --benchmark mmmu --mmmu-split test
```

Smoke mode always uses 5 samples from `Accounting`.

## Metrics

Leaderboard-facing metrics:

- `mmmu_accuracy`: overall accuracy over selected samples.
- `mmmu_invalid_rate`: multiple-choice responses that required fallback parsing.
- `mmmu_multiple_choice_accuracy`
- `mmmu_open_accuracy`
- `mmmu_samples`
- `mmmu_runtime_seconds`

Per-sample output is written to `results/mmmu/<variant>/samples.jsonl`.
Summary metrics are written to `results/mmmu/<variant>/summary.json`.

The local evaluator follows the official MMMU response parsing and exact-match
logic for multiple-choice and open questions. The evaluator version is recorded as
`mmmu_official_parse_local_v1`.
