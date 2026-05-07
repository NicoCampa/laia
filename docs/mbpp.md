# MBPP

MBPP, Mostly Basic Programming Problems, evaluates short Python program synthesis
from natural-language task descriptions. Local AI Analysis loads
`google-research-datasets/mbpp` from Hugging Face and evaluates pass@1 by running
the generated Python against the dataset assertions in a local subprocess.

Run a smoke test:

```bash
laia ollama qwen3.5:0.8b-mlx-bf16 --benchmark mbpp --smoke
laia lmstudio exact-model-id --benchmark mbpp --smoke
laia omlx exact-model-id --benchmark mbpp --smoke
```

Run the standard full MBPP test split:

```bash
laia ollama qwen3.5:0.8b-mlx-bf16 --benchmark mbpp
```

Use the hand-verified subset:

```bash
laia omlx exact-model-id --benchmark mbpp --mbpp-config sanitized
```

Optional challenge assertions for the `full` config:

```bash
laia ollama qwen3.5:0.8b-mlx-bf16 --benchmark mbpp --mbpp-challenge-tests
```

Leaderboard-facing metrics:

- `mbpp_pass_at_1`: fraction of samples whose generated code passed all selected tests.
- `mbpp_invalid_rate`: fraction of samples with no generated code or syntax errors.
- `mbpp_compile_rate`: fraction of samples that produced syntactically valid executable scripts.
- `mbpp_runtime_error_rate`: fraction of samples that timed out or raised non-assertion runtime errors.

Summary metrics are written to `results/mbpp/<variant>/summary.json`. Per-sample
records, including prompts, raw outputs, extracted code, test results, stderr, and
runtime, are written to `results/mbpp/<variant>/samples.jsonl`.

Safety note: MBPP executes model-generated Python locally. The evaluator runs code
in a separate isolated-mode Python process with a per-sample timeout, but it is not
a hardened security sandbox. Run it only against models and machines where local
code execution is acceptable.

Primary references:

- [Google Research publication](https://research.google/pubs/program-synthesis-with-large-language-models/)
- [Google Research MBPP README](https://github.com/google-research/google-research/tree/master/mbpp)
- [Hugging Face dataset mirror](https://huggingface.co/datasets/google-research-datasets/mbpp)
