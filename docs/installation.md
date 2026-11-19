[Docs](README.md) › **Installation**

# Installation

You need Python 3.10 or newer.

```bash
git clone https://github.com/insait-institute/sabermath
cd sabermath
python -m pip install -e .
```

The base install includes Sentence Transformers and the OpenRouter client, so it already runs:

- any [Sentence Transformers](https://sbert.net) model, from a Hugging Face id or an object you loaded;
- any embedding model served by [OpenRouter](https://openrouter.ai) (needs an API key, see [below](#api-keys));
- your own [custom processors](custom-processors.md).

## Optional: vLLM

[vLLM](https://github.com/vllm-project/vllm) runs large embedding models much faster. It needs a CUDA GPU.

```bash
python -m pip install -e ".[vllm]"
```

The extra pins `vllm==0.26.0`. Other vLLM versions may name some options differently.
How to use it: [Built-in processors › vLLM](processors.md#vllm).

## API keys

| Variable | Needed for |
|---|---|
| `OPENROUTER_API_KEY` | `OpenRouterEmbeddingProcessor`, unless you pass `api_key=...` |
| `HF_TOKEN` | optional. The datasets are public, so only set it if you hit Hugging Face rate limits |
| `SABERMATH_VLLM_EXPORT_DIR` | optional. Where vLLM's `export_clean=True` option caches re-exported checkpoints |

```bash
export OPENROUTER_API_KEY=<your-key>
```

Installing also adds the `sabermath` command, see [Command line](cli.md).

## Check the install

Run the main task on 20 queries with a small model (works on CPU):

```python
import sabermath
from sabermath.data import load_data

queries, documents = load_data()           # downloads the benchmark on first use
report = sabermath.evaluate(
    "sentence-transformers/all-MiniLM-L6-v2",
    tasks=["statement-full"],
    queries=queries.select(range(20)),
    documents=documents,
)
print(report.tasks[0].ndcg_at_k)           # a random ranking scores about 0.29
```

---

<table width="100%"><tr>
<td align="left">← <a href="benchmark.md">The benchmark</a></td>
<td align="right"><a href="evaluate.md">Running <code>evaluate()</code></a> →</td>
</tr></table>
