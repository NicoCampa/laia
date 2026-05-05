# IFBench Instruction-Following Benchmark

Local AI Analysis can run IFBench against the same native Ollama, LM Studio, and
oMLX providers used for Global MMLU Lite.

IFBench evaluates precise instruction following with verifiable constraints. The
current implementation uses the public `allenai/IFBench_test` dataset and the official
AllenAI verification functions pinned through the `eval` extra.

## Protocol

The default run loads:

```python
load_dataset("allenai/IFBench_test", split="train")
```

The full set contains 300 prompts. Smoke mode evaluates the first 5 prompts.

Generation defaults follow the vanilla IFBench harness:

```text
temperature = 0.01
top_p = 0.95
max_tokens = 4096
```

Shortcut commands still default to `--reasoning-effort none`. For oMLX this sends
`chat_template_kwargs.enable_thinking=false`; for Ollama this sends `think=false`.

## Metrics

Primary metric:

```text
ifbench_prompt_level_loose
```

Additional metrics:

- `ifbench_instruction_level_loose`
- `ifbench_prompt_level_strict`
- `ifbench_instruction_level_strict`
- `ifbench_samples`
- `ifbench_runtime_seconds`

The IFBench paper reports prompt-level loose accuracy, so that is the leaderboard-facing
IFBench score.

## Commands

```bash
laia omlx Qwen3.5-9B-4bit --benchmark ifbench --smoke
laia omlx Qwen3.5-9B-4bit --benchmark ifbench
```

```bash
laia ollama qwen3.5:0.8b-mlx-bf16 --benchmark ifbench
laia lmstudio exact-model-id --benchmark ifbench
```

Run both supported benchmarks in one generated config:

```bash
laia omlx Qwen3.5-9B-4bit --benchmark all
```
