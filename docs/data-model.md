# Data Model

The DuckDB schema stores benchmark rows as auditable measurements.

Core entities:

- `base_model`: model family, name, parameter count, architecture, license, source.
- `model_variant`: served model id/tag, quantization label, precision, metadata.
- `benchmark_run`: config path, seed, command arguments, timestamps, run UUID.
- `benchmark_task`: Global MMLU Lite, IFBench, BFCL v4, OCRBench v2, MMMU, MBPP, RGB, SimpleQA, or HarmBench task metadata and decoding settings.
- `benchmark_result`: raw metric values and raw payloads.
- `hardware_profile`: local hardware and OS metadata.
- `backend_profile`: Ollama, LM Studio, or oMLX native server metadata.
- `normalized_result`: leaderboard-facing Global MMLU Lite, IFBench, BFCL v4, OCRBench v2, MMMU, MBPP, RGB, SimpleQA, and HarmBench result rows.

Existing local DB files may still contain older nullable columns from previous experiments.
The current code writes and exports Global MMLU Lite score, IFBench prompt-level loose
score, BFCL v4 selected accuracy, OCRBench v2 score, MMMU accuracy, supporting task
metrics, MBPP pass@1 and compile/error rates, runtime, backend, model id, run id,
RGB all-rate/rejection/fact-check rates, SimpleQA factuality/hallucination rates,
HarmBench attack-success/refusal rates, weighted model intelligence score,
coverage, and metadata.

Model intelligence columns:

- `model_intelligence_score`: full-suite weighted score with missing benchmark
  families counted as zero.
- `model_intelligence_coverage`: sum of the benchmark-family weights present in
  the row.
- `model_intelligence_available_score`: weighted average over the benchmark
  families present in the row.

HarmBench safety metrics are stored in the same row shape but are not included in
the model intelligence score.
