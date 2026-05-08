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
- dataset name, revision, split, languages, OCRBench configs, MMMU subjects,
  MBPP config/split, RGB dataset/noise/passage settings, SimpleQA grader,
  HarmBench judge/category settings, and BFCL categories
- prompt template and parser version
- temperature, max tokens, seed, stop, top-p, and reasoning effort
- raw prompt, raw output, extracted answer, gold answer, and correctness per question
- per-question runtime and API usage when available
- backend profile, hardware profile, command arguments, and run UUID

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
  BFCL category set because smoke, single-turn, and all-scoring runs are not equivalent
- compare OCRBench v2 rows by `ocrbench_v2_score`, and only compare rows that use
  the same evaluator version and dataset config set
- compare MMMU rows by `mmmu_accuracy`, and record the split and subject list
- compare MBPP rows by `mbpp_pass_at_1`, and record whether the run used the
  `full` or `sanitized` config and whether challenge tests were enabled
- compare RGB rows by `rgb_all_rate`, and record the RGB dataset, noise rate,
  passage count, correct rate, and whether the row is noise robustness,
  negative rejection, information integration, or counterfactual robustness
- compare SimpleQA rows by `simpleqa_f1`, and record whether the grader was an
  LLM judge or the deterministic heuristic. For publishable rows, use a stronger
  pinned judge model instead of an unpinned `same` grader.
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
