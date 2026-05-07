# RGB

RGB is the Retrieval-Augmented Generation Benchmark from Chen et al. It evaluates
four RAG abilities: noise robustness, negative rejection, information integration,
and counterfactual robustness. Local AI Analysis downloads the official JSONL data
from `chen700564/RGB`, builds the document prompt, sends it to the native local API,
and applies the official lexical scoring logic from `evalue.py`.

Run a smoke test:

```bash
laia ollama qwen3.5:0.8b-mlx-bf16 --benchmark rgb --smoke
laia lmstudio exact-model-id --benchmark rgb --smoke
laia omlx exact-model-id --benchmark rgb --smoke
```

Noise robustness, using the refined English set:

```bash
laia ollama qwen3.5:0.8b-mlx-bf16 --benchmark rgb --rgb-dataset en_refine --rgb-noise-rate 0.6
```

Negative rejection, where all supplied passages are noisy:

```bash
laia omlx exact-model-id --benchmark rgb --rgb-dataset en_refine --rgb-noise-rate 1
```

Information integration:

```bash
laia omlx exact-model-id --benchmark rgb --rgb-dataset en_int --rgb-noise-rate 0.2
```

Counterfactual robustness:

```bash
laia omlx exact-model-id --benchmark rgb --rgb-dataset en_fact --rgb-noise-rate 0.2
```

Supported datasets:

- `en_refine`, `zh_refine`: refined noise robustness / negative rejection data.
- `en`, `zh`: original noise robustness / negative rejection data.
- `en_int`, `zh_int`: information integration data.
- `en_fact`, `zh_fact`: counterfactual robustness data.

Leaderboard-facing metrics:

- `rgb_all_rate`: official all-rate, meaning accuracy for noise/integration/counterfactual rows or exact rejection rate when `noise_rate=1`.
- `rgb_accuracy`: answer accuracy for non-rejection RGB rows.
- `rgb_rejection_rate`: exact rejection rate for negative-rejection rows.
- `rgb_fact_check_rate`: exact factual-error detection phrase rate for counterfactual rows.
- `rgb_error_correction_rate`: fraction of fact-detection responses that also contain the correct answer.

Summary metrics are written to `results/rgb/<variant>/summary.json`. Per-sample
records, including selected documents, prompts, raw outputs, labels, and runtime,
are written to `results/rgb/<variant>/samples.jsonl`.

The repo also includes optional ChatGPT judge scripts for relaxed rejection and
counterfactual evaluation. Local AI Analysis does not call an external judge; it
uses the local exact lexical evaluator so runs stay local and reproducible.

Primary references:

- [RGB GitHub repository](https://github.com/chen700564/RGB)
- [RGB paper](https://arxiv.org/abs/2309.01431)
