[Docs](README.md) › **Built-in processors**

# Built-in processors

A **processor** is the object `evaluate()` asks for scores. Given one query and its 150 candidate documents,
it returns one score per document. Every processor derives from `ModelProcessor`:

```text
ModelProcessor                        get_scores(query, documents) -> scores      ← subclass for rerankers, BM25, …
└── EmbeddingProcessor                get_scores = encode() + cosine similarity   ← subclass for embedding models
    ├── SentenceTransformersProcessor     processor = "sentence-transformers"
    ├── VLLMProcessor                     processor = "vllm"
    ├── OpenRouterEmbeddingProcessor      processor = "openrouter"
    └── UnknownProcessor                  fallback for unrecognized objects (can't score yet)
```

All classes import from `sabermath.processors`. To write your own, see [Writing a processor](custom-processors.md).

**On this page:** [Building a processor](#building-a-processor) · [Shared embedding options](#options-shared-by-all-embedding-processors) ·
[Sentence Transformers](#sentence-transformers) · [vLLM](#vllm) · [OpenRouter](#openrouter) · [UnknownProcessor](#unknownprocessor)

---

## Building a processor

You can let `evaluate` build the processor from a model id, or build one yourself and pass the instance:

```python
import sabermath
from sabermath.processors import SentenceTransformersProcessor

# 1. evaluate builds it; init_kwargs go to the constructor
sabermath.evaluate("BAAI/bge-m3", init_kwargs={"device": "cuda"})

# 2. you build it; useful to reuse one loaded model across several evaluate() calls
processor = SentenceTransformersProcessor.from_huggingface("BAAI/bge-m3", device="cuda")
sabermath.evaluate(processor)
```

Either way, per-run options go in `scores_kwargs` (see [`init_kwargs` vs `scores_kwargs`](evaluate.md#init_kwargs-and-scores_kwargs)).

## Options shared by all embedding processors

Every `EmbeddingProcessor` scores the same way: all built-ins and every subclass you write embed the query and the documents,
then return the cosine similarity of each document to the query. Its `get_scores` reads these keys from `scores_kwargs`:

| Key | Effect |
|---|---|
| `query_prompt` | text prepended to the query |
| `document_prompt` | text prepended to every document |
| `query_suffix` | text appended to the query |
| `document_suffix` | text appended to every document |
| `query_encode_kwargs` | dict of extra `encode()` arguments, **query side only** |
| `document_encode_kwargs` | dict of extra `encode()` arguments, **document side only** |

Every other key is passed unchanged to `encode()` for both sides.

Without `query_encode_kwargs` and `document_encode_kwargs`, the documents and the query are embedded in **one** `encode()` call, with the query last.
With either of them, the two sides are encoded in two separate calls.

```python
# fixed text prefixes, E5-style (on SABER-Math the paper found E5 does better *without* them)
sabermath.evaluate(
    "intfloat/multilingual-e5-large",
    scores_kwargs={"query_prompt": "query: ", "document_prompt": "passage: "},
)

# a named prompt from the model's Sentence Transformers config, on the query side only
sabermath.evaluate(
    "Qwen/Qwen3-Embedding-0.6B",
    scores_kwargs={"query_encode_kwargs": {"prompt_name": "query"}, "batch_size": 32},
)
```

The paper's per-model input formats are listed in [Matching the paper's protocol](evaluate.md#matching-the-papers-protocol).

## Sentence Transformers

`SentenceTransformersProcessor` is what `evaluate` uses for a model id by default.

```python
from sentence_transformers import SentenceTransformer
from sabermath.processors import SentenceTransformersProcessor

SentenceTransformersProcessor.from_huggingface("BAAI/bge-m3", device="cuda")
SentenceTransformersProcessor(SentenceTransformer("BAAI/bge-m3"), model_name="bge-m3")
```

**Construction:** `init_kwargs` and `from_huggingface(name, **kwargs)` are passed to `SentenceTransformer(name, **kwargs)`
(`device`, `trust_remote_code`, `model_kwargs`, `truncate_dim`, …).
`model_name` sets `report.model`.

**`scores_kwargs`:** the [shared options](#options-shared-by-all-embedding-processors), the arguments of
`SentenceTransformer.encode` (`batch_size`, `prompt_name`, `prompt`, `normalize_embeddings`, …), plus:

| Key | Default | Effect |
|---|---|---|
| `chunk_to_context` | `False` | split texts longer than the context into token chunks, embed each, and average, instead of truncating |
| `context_length` | model's `max_seq_length`, then the tokenizer's `model_max_length` | chunk length in tokens |

## vLLM

`VLLMProcessor` runs embedding models with vLLM. It needs the [`vllm` extra](installation.md#optional-vllm) and a GPU.

```python
sabermath.evaluate(
    "Qwen/Qwen3-Embedding-8B",
    use_vllm=True,
    init_kwargs={"max_model_len": 8192, "tensor_parallel_size": 2},
    scores_kwargs={"batch_size": 256},
)

# or with an LLM you created yourself
from vllm import LLM
sabermath.evaluate(LLM(model="Qwen/Qwen3-Embedding-8B", runner="pooling"))
```

**Construction:** `init_kwargs`, or `VLLMProcessor.from_huggingface(name, ...)`, accepts:

| Key | Effect |
|---|---|
| `pooler_config` | dict of vLLM `PoolerConfig` fields, e.g. `{"pooling_type": "LAST", "normalize": True}`, or a `PoolerConfig` instance. Field names unknown to the installed vLLM are mapped to their aliases or dropped with a message |
| `export_clean` | default `False`. Re-save the checkpoint through `transformers.AutoModel` first and load that copy, for checkpoints vLLM can't read directly. The copy is cached under `$SABERMATH_VLLM_EXPORT_DIR`, else `/scratch/$USER/sabermath_vllm_artifacts` if `/scratch/$USER` exists, else `~/.cache/sabermath/vllm_artifacts` |
| anything else | passed to `vllm.LLM(...)`, overriding the defaults `runner="pooling"`, `gpu_memory_utilization=0.9`, `trust_remote_code=True`, `enforce_eager=True` |

**`scores_kwargs`:** the [shared options](#options-shared-by-all-embedding-processors), plus:

| Key | Default | Effect |
|---|---|---|
| `batch_size` | all texts at once | texts per `LLM.embed` call |
| `truncate_to` | tokenizer's `model_max_length` | token limit per text. vLLM *rejects* over-long inputs instead of truncating them, so set this when you lower `max_model_len` |
| `chunk_to_context` | `False` | as for Sentence Transformers |
| `context_length` | tokenizer's `model_max_length` | as for Sentence Transformers |

Other keys are passed to `LLM.embed`.

> [!NOTE]
> Truncation and chunking keep special tokens that *you* put in the text (for example a `<|im_end|>` suffix), because
> last-token poolers depend on them.

## OpenRouter

`OpenRouterEmbeddingProcessor` calls any embedding model hosted on [OpenRouter](https://openrouter.ai).

```python
from sabermath.processors import OpenRouterEmbeddingProcessor

processor = OpenRouterEmbeddingProcessor("openai/text-embedding-3-large")   # key from OPENROUTER_API_KEY
# or: OpenRouterEmbeddingProcessor("openai/text-embedding-3-large", api_key="sk-or-...")

sabermath.evaluate(processor, scores_kwargs={"max_concurrency": 50})
```

- The model name is the full OpenRouter id (`provider/model`) and is sent as is.
- Without a key, the constructor warns, and requests will probably fail to authenticate.

**`scores_kwargs`:** the [shared options](#options-shared-by-all-embedding-processors), plus:

| Key | Default | Effect |
|---|---|---|
| `max_concurrency` | 20 | parallel requests |
| `retries` | 9 | attempts per text, with exponential back-off (1 s, 2 s, 4 s, …) |
| any `embeddings.create` argument | | e.g. `dimensions` |

> [!WARNING]
> - The processor sends one request per text and caches nothing, so a full three-task run makes about
>   3 × 1,000 × 151 ≈ 450K requests. Try it on a [subset](evaluate.md#queries-and-documents) first, or add a
>   [cache](custom-processors.md#example-caching-embeddings).
> - It runs its own event loop, so calling it from inside a running `asyncio` loop (for example a Jupyter cell that is already awaiting) raises
>   `RuntimeError`.

## UnknownProcessor

`evaluate` wraps objects it doesn't recognize in `UnknownProcessor`:

- An object with a callable `.encode()` or `.get_scores()` is accepted, but scoring **currently raises
  `NotImplementedError`** at the first query.
- An object with neither method raises `ValueError` immediately.

Until this is implemented, wrap your model in a [custom processor](custom-processors.md). It usually takes about 15 lines.

---

<table width="100%"><tr>
<td align="left">← <a href="evaluate.md">Running <code>evaluate()</code></a></td>
<td align="right"><a href="custom-processors.md">Writing a processor</a> →</td>
</tr></table>
