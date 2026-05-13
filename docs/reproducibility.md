# Reproducibility

Local AI Analysis treats each row as an auditable local measurement.

Suite aliases are intentionally layered:

- `text`: text-only, non-judge benchmarks.
- `vision`: multimodal non-judge benchmarks.
- `judge`: benchmarks that require an LLM judge.
- `suite`: `text` plus `vision`, without judge-based benchmarks.
- `full`: every benchmark family.

For compatibility, `core` maps to `text`, `all` maps to `suite`, and `judged`
maps to `judge`.

Every run records:

- provider name and native API base URL
- exact model tag or model id
- model/run modality metadata and benchmark input modalities
- dataset name, revision, split, languages, sample limit/strategy/seed,
  OCRBench configs, MMMU subjects, MBPP config/split, RGB dataset/noise/passage
  settings, SimpleQA grader, HarmBench judge/category settings, and BFCL
  categories
- prompt template and parser version
- temperature, max tokens, seed, stop, top-p, reasoning effort, and requested
  context length
- raw prompt, raw output, extracted answer, gold answer, and correctness per question
- per-question runtime and API usage when available
- backend profile, hardware profile, command arguments, and run UUID
- aggregate efficiency metrics derived from per-question logs: input/output/total
  tokens, reasoning tokens when exposed, output tokens per second, average/P50/P95
  latency, truncation rate, tokens per correct answer, and seconds per correct
  answer
- Artificial Analysis-style streaming fields for time to first token,
  inter-token latency, end-to-end latency, and system output throughput. These
  are present in the schema but remain empty until a streaming speed run is used.
- Artificial Analysis-style cost fields for input, output, total, and
  per-correct-answer cost. These remain empty for local runs unless a pricing
  profile is attached.

Completed benchmark metrics are written as soon as each benchmark finishes. If a
later benchmark fails, the run is marked `failed`, but the already completed
benchmark metrics are kept, normalized, and available for website export.

Shortcut-generated configs pin dataset sources to explicit upstream revisions:

- Global MMLU Lite: `cbf2f73663ff201d4d56e891c8c2c18467aeea06`
- IFBench: `2e8a48de45ff3bf41242f927254ca81b59ca3ae2`
- OCRBench v2: `458b55b5f62bfd6eba7b5080da34fbc9a68c2626`
- MMMU: `4619a102cf5ad2da1abf7e220fde1258d2434cb7`
- MBPP: `4bb6404fdc6cacfda99d4ac4205087b89d32030c`
- RGB: `65ec39e40e7dc9abb50e9bf1b4f32be3f6f16615`
- SimpleQA reference implementation: `652c89d0ca9df547706735883097e9537d40dc47`
- HarmBench: `8e1604d1171fe8a48d8febecd22f600e462bdcdd`

For SimpleQA, the official dataset is distributed from OpenAI public blob storage;
the run records the pinned `simple-evals` reference revision and caches the CSV
under a revision-scoped cache directory.

For publishable runs:

- pin the model id instead of relying on `auto`
- for oMLX, record the model directory used to start the server, for example
  `/Users/nicolocampagnoli/.lmstudio/models`
- keep the generated config under `results/generated_configs/`
- keep the raw JSONL and summary under the matching `results/<benchmark>/` directory
- avoid comparing Global MMLU Lite generation pass@1 directly with log-likelihood MMLU
- compare IFBench with IFBench prompt-level loose accuracy unless you explicitly need
  strict or instruction-level metrics
- compare BFCL v4 rows by `bfcl_v4_selected_accuracy`, and record the selected
  BFCL category set, sample limit, strategy, and seed because smoke,
  single-turn, sampled, and all-scoring runs are not equivalent. Shortcut-generated
  full runs use 1,000 seeded stratified samples by default.
- compare OCRBench v2 rows by `ocrbench_v2_score`, and only compare rows that use
  the same evaluator version, dataset config set, sample limit, sample strategy,
  and sample seed. Shortcut-generated full runs use a deterministic 1,000-sample
  stratified subset by default.
- compare MMMU rows by `mmmu_accuracy`, and record the split and subject list
- compare MBPP rows by `mbpp_pass_at_1`, and record whether the run used the
  `full` or `sanitized` config and whether challenge tests were enabled
- compare RGB rows by `rgb_all_rate`; default generated configs use the curated
  English/Chinese RGB suite, while single-dataset runs should record the RGB
  dataset, noise rate, passage count, correct rate, task mode, sample limit, and
  sample strategy. Shortcut-generated full RGB suite runs use 100 seeded random
  rows per suite slice, for 800 RGB calls total.
- compare SimpleQA rows by `simpleqa_f1`, and record whether the grader was an
  LLM judge or the deterministic heuristic. For publishable rows, use a stronger
  pinned judge model instead of an unpinned `same` grader. Shortcut-generated
  full runs use 500 seeded stratified questions by default.
- compare HarmBench rows by `harmbench_attack_success_rate` for attack success
  or `harmbench_refusal_rate` for safety. Record the selected functional
  categories and judge model. Default generated configs reuse the same local
  judge-model convention as SimpleQA.
- compare broad model profiles by `model_intelligence_score`; check
  `model_intelligence_coverage` before ranking rows because missing benchmark
  families are intentionally counted as zero. The stored score is normalized
  from 0 to 1, while the website and terminal leaderboard render it as points
  out of 100.

Default model intelligence weights are Global MMLU Lite 17%, IFBench 17%,
BFCL v4 17%, OCRBench v2 11%, MMMU 17%, MBPP 16%, and RGB 5%.
SimpleQA and HarmBench are reported separately because they require a judge.
