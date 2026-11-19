[Docs](README.md) › **API reference**

# API reference

Signatures and short descriptions of the public API. The guides explain the API in more depth.
Each section below links to the guide that covers it.

**Contents:** [`sabermath`](#sabermath) · [`sabermath.processors`](#sabermathprocessors) ·
[`sabermath.data`](#sabermathdata) · [`sabermath.instructions`](#sabermathinstructions) · [`sabermath.metrics`](#sabermathmetrics)

---

## `sabermath`

### `evaluate`

```python
sabermath.evaluate(
    model,
    tasks: list[Task] | None = None,
    k: int = 10,
    *,
    dcg_variant: Literal["exponent", "linear"] = "exponent",
    use_vllm: bool = False,
    init_kwargs: dict | None = None,
    scores_kwargs: dict | None = None,
    on_progress: Callable[[], None] | None = None,
    instruction: str | None = None,
    instruction_template: str = "canonical",
    verbose: bool = True,
    show_progress_bars: bool = True,
    queries: datasets.Dataset | None = None,
    documents: datasets.Dataset | None = None,
    return_ndcgs: bool = False,
) -> Report | tuple[Report, dict[Task, list[float | None]]]
```

Runs the benchmark and returns a `Report`, or `(report, ndcgs)` when `return_ndcgs=True`.
→ Guide: [Running `evaluate()`](evaluate.md#arguments)

| Raises | When |
|---|---|
| `ValueError` | unknown task name, unknown `dcg_variant`, or unknown `instruction_template` (when `instruction` is set) |
| `RuntimeError` | `use_vllm=True` with a non-string `model`, or `init_kwargs` given with a processor instance |

### Result types

All are dataclasses. → Guide: [The report](evaluate.md#the-report)

```python
@dataclass
class Report:
    model: str                    # processor.model, or "unknown"
    processor: str                # processor.processor, or "unknown"
    dcg_variant: Literal["exponent", "linear"]
    k: int
    tasks: list[TaskResult]
    def to_dict(self) -> dict: ...

@dataclass(frozen=True)
class TaskResult:
    task: Task
    ndcg_at_k: float
    branches: list[BranchResult]  # five, in the order of Branch

@dataclass(frozen=True)
class BranchResult:
    branch: Branch
    ndcg_at_k: float
```

### Type aliases

```python
Task   = Literal["statement-statement", "statement-full", "full-full"]
Branch = Literal["Algebra", "Geometry", "Number Theory", "Combinatorics", "Calculus and Analysis"]
```

`sabermath` also re-exports `EmbeddingProcessor`.

## `sabermath.processors`

→ Guides: [Built-in processors](processors.md) · [Writing a processor](custom-processors.md)

### `ModelProcessor` (abstract)

```python
class ModelProcessor(ABC):
    processor: ClassVar[str | None]          # must be set by subclasses

    @property
    def model(self) -> str | None: ...       # default: None

    @abstractmethod
    def get_scores(self, query: str, documents: list[str], *,
                   show_progress_bar: bool = True, **kwargs) -> list[float] | np.ndarray | None: ...
```

### `EmbeddingProcessor(ModelProcessor)` (abstract)

```python
class EmbeddingProcessor(ModelProcessor):
    @abstractmethod
    def encode(self, texts: list[str], show_progress_bar: bool = True, **kwargs) -> np.ndarray: ...

    def get_scores(self, query, documents, *, show_progress_bar=True,
                   query_prompt=None, document_prompt=None,
                   query_suffix=None, document_suffix=None,
                   query_encode_kwargs=None, document_encode_kwargs=None,
                   **kwargs) -> np.ndarray: ...       # cosine similarity

    def encode_statements(self, ds, show_progress_bar=True, *, statement_column="problem", **kwargs) -> np.ndarray: ...
    def encode_full(self, ds, show_progress_bar=True, *, statement_column="problem",
                    solution_column="solution", **kwargs) -> np.ndarray: ...
```

`encode_statements` and `encode_full` embed a whole dataset's problems, or its `"Problem: …\n\nSolution: …"` texts.
`evaluate` doesn't use them. They are there for your own analyses.

### `SentenceTransformersProcessor(EmbeddingProcessor)`

```python
SentenceTransformersProcessor(st: SentenceTransformer, model_name: str | None = None)
SentenceTransformersProcessor.from_huggingface(model_name: str, **kwargs)   # kwargs → SentenceTransformer(...)

encode(texts, show_progress_bar=True, chunk_to_context=False, context_length=None, **kwargs)
```

`processor = "sentence-transformers"` · [details](processors.md#sentence-transformers)

### `VLLMProcessor(EmbeddingProcessor)`

```python
VLLMProcessor(llm: vllm.LLM, model_name: str | None = None, tokenizer_name: str | None = None)
VLLMProcessor.from_huggingface(model_name: str, *, pooler_config: dict | PoolerConfig | None = None,
                               export_clean: bool = False, **kwargs)        # kwargs → vllm.LLM(...)

encode(texts, show_progress_bar=True, *, batch_size=None, chunk_to_context=False,
       context_length=None, truncate_to=None, **kwargs)                     # kwargs → LLM.embed(...)
```

`processor = "vllm"` · [details](processors.md#vllm)

### `OpenRouterEmbeddingProcessor(EmbeddingProcessor)`

```python
OpenRouterEmbeddingProcessor(model_name: str, api_key: str | None = None)  # key falls back to $OPENROUTER_API_KEY

encode(texts, show_progress_bar=True, *, retries=9, max_concurrency=20, **kwargs)        # kwargs → embeddings.create(...)
async encode_async(texts, show_progress_bar=True, *, max_concurrency=20, retries=3, **kwargs)
```

`processor = "openrouter"` · [details](processors.md#openrouter)

### `UnknownProcessor(EmbeddingProcessor)`

```python
UnknownProcessor(model)   # model must have a callable .encode() or .get_scores(), else ValueError
                          # note: get_scores() currently raises NotImplementedError
```

`processor = None` · [details](processors.md#unknownprocessor)

## `sabermath.data`

```python
sabermath.data.load_data() -> tuple[datasets.Dataset, datasets.Dataset]   # (queries, documents)

TARGETS_DATASET    = "INSAIT-Institute/SaberMath-queries"
CANDIDATES_DATASET = "INSAIT-Institute/SaberMath-documents"
```

→ Columns: [The benchmark › The data](benchmark.md#the-data)

## `sabermath.instructions`

```python
INSTRUCTIONS: dict[str, str | None]          # keys "p0" (None), "p1", "p2", "p3", "pm"
INSTRUCTION_TEMPLATES: dict[str, str]        # {"canonical": "Instruct: {instruction}\nQuery: {query}"}
DEFAULT_INSTRUCTION_TEMPLATE = "canonical"

format_instructed_query(instruction: str, problem: str, template: str = "canonical") -> str
```

→ Guide: [Instructions](evaluate.md#instructions)

## `sabermath.metrics`

```python
compute_ndcg_at_k(relevances, k=10, variant="linear") -> float
dcg_at_k(relevances, k=10, variant="linear") -> float
cosine_similarity(query_emb: np.ndarray, docs_embs: np.ndarray) -> np.ndarray
```

`relevances` are the reference ratings **in the order your model ranked the documents**.
`compute_ndcg_at_k` divides their DCG by the DCG of the same ratings sorted best-first.

> [!NOTE]
> These low-level functions default to `variant="linear"`, but `evaluate()` defaults to `"exponent"`.
> Pass the variant explicitly when you reproduce benchmark numbers by hand.

```python
import numpy as np
from sabermath.metrics import compute_ndcg_at_k

ranked = np.asarray(q["relevance_scores"])[np.argsort(-scores)]
compute_ndcg_at_k(ranked, k=10, variant="exponent")
```

---

<table width="100%"><tr>
<td align="left">← <a href="custom-processors.md">Writing a processor</a></td>
<td align="right"><a href="cli.md">Command line</a> →</td>
</tr></table>
