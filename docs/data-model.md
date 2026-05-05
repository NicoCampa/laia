# Data Model

The DuckDB schema stores benchmark rows as auditable measurements.

Core entities:

- `base_model`: model family, name, parameter count, architecture, license, source.
- `model_variant`: served model id/tag, quantization label, precision, metadata.
- `benchmark_run`: config path, seed, command arguments, timestamps, run UUID.
- `benchmark_task`: Global MMLU Lite, IFBench, or BFCL v4 task metadata and decoding settings.
- `benchmark_result`: raw metric values and raw payloads.
- `hardware_profile`: local hardware and OS metadata.
- `backend_profile`: Ollama, LM Studio, or oMLX native server metadata.
- `normalized_result`: leaderboard-facing Global MMLU Lite, IFBench, and BFCL v4 result rows.

Existing local DB files may still contain older nullable columns from previous experiments.
The current code writes and exports Global MMLU Lite score, IFBench prompt-level loose
score, BFCL v4 selected accuracy, supporting task metrics, runtime, backend, model id,
run id, and metadata.
