# Reproducibility

Local AI Analysis treats each row as an auditable local measurement.

Every run records:

- provider name and native API base URL
- exact model tag or model id
- dataset name, revision, split, and languages
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
