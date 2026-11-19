<div align="center">
  <img src="images/sabermath_logo.png" alt="SABER-Math logo" width="320">

# SABER-Math

**An automated reranking benchmark for mathematical information retrieval**

[📄 Paper](https://arxiv.org/abs/2606.29894) · [🌐 Website](https://sabermath.insait.ai) · [🤗 Data](https://huggingface.co/collections/INSAIT-Institute/saber-math) · [📚 Docs](docs/README.md)

</div>

Retrieval in math only helps if it returns problems that share a *method*, a *theorem* or a *key idea*
with the problem at hand. Problems that merely look alike don't help.
**SABER-Math** measures exactly that, and this package lets you score any retriever or reranker on it with one call.

- **1,000 queries × 150 candidates.** Each query is a high-school or olympiad problem. Its candidates are solved
  problems picked by topic similarity in a math ontology and by solution-method similarity.
- **Fine-grained labels, no expert annotation.** Every candidate has a relevance rating in [0, 5] from a Swiss-style
  tournament of pairwise LLM judgments, aggregated with Bradley–Terry and validated against human annotators.
- **Three settings, five branches.** Problem→problem, problem→problem+solution (main), problem+solution→problem+solution.
  Results are broken down into Algebra, Geometry, Number Theory, Combinatorics, and Calculus and Analysis.

More in [The benchmark](docs/benchmark.md).

## Installation

```bash
git clone https://github.com/insait-institute/sabermath
cd sabermath
python -m pip install -e .            # Sentence Transformers + OpenRouter
python -m pip install -e ".[vllm]"    # optional: adds vLLM (needs a CUDA GPU)
```

Python 3.10+. Details and API keys: [Installation](docs/installation.md).

## Quickstart

```python
import sabermath

report = sabermath.evaluate("BAAI/bge-m3")      # downloads the data on first use

for task in report.tasks:
    print(task.task, round(task.ndcg_at_k, 4))
```

Or from the shell: `sabermath evaluate BAAI/bge-m3 --task statement-full` (see [Command line](docs/cli.md)).

A model id is loaded with Sentence Transformers, or with vLLM if you pass `use_vllm=True`.
The most common options:

```python
report = sabermath.evaluate(
    "Qwen/Qwen3-Embedding-0.6B",
    tasks=["statement-full"],                          # the main setting; default: all three
    init_kwargs={"device": "cuda"},                    # how to build the model
    scores_kwargs={                                    # passed to every scoring call
        "batch_size": 32,
        "query_encode_kwargs": {"prompt_name": "query"},
    },
)
report.to_dict()                                       # JSON-ready
```

Every argument, the report format, instructions, confidence intervals and the paper's exact protocol are covered in
**[Running `evaluate()`](docs/evaluate.md)**.

## Evaluating your own model

`evaluate()` scores through a **processor**. Three are built in:

| Processor | How to use it |
|---|---|
| `SentenceTransformersProcessor` | `evaluate("model/id")`, or pass a `SentenceTransformer` |
| `VLLMProcessor` | `evaluate("model/id", use_vllm=True)`, or pass a `vllm.LLM` |
| `OpenRouterEmbeddingProcessor` | `evaluate(OpenRouterEmbeddingProcessor("openai/text-embedding-3-large"))` |

For anything else, subclass one of two base classes. An embedding model needs only `encode`:

```python
import numpy as np
from sabermath.processors import EmbeddingProcessor

class MyEmbedder(EmbeddingProcessor):
    processor = "my-backend"                          # required, shown in the report

    def __init__(self, model_name):
        self._model_name = model_name
        self._model = load_my_model(model_name)

    @property
    def model(self):
        return self._model_name

    def encode(self, texts, show_progress_bar=True, **kwargs):
        return np.asarray(self._model.embed(texts))   # (len(texts), dim); cosine scoring is built in

report = sabermath.evaluate(MyEmbedder("my-model"))
```

A reranker, BM25, or any other scorer subclasses `ModelProcessor` and implements
`get_scores(query, documents)`, which returns one score per document (higher = more relevant).

See **[Writing a processor](docs/custom-processors.md)** for the full contract and worked examples:
mean pooling, query/document-specific encoding, caching, cross-encoders, BM25, and how to test a processor.

## Results

Selected nDCG@10 scores from the paper, in the main `statement-full` setting (42 systems are evaluated there):

| System | nDCG@10 |
|---|---|
| ReasonReranker-Qwen3-32B-Rewrite | **0.741** |
| ReasonEmbed-Qwen3-8B | 0.710 |
| Qwen3-Reranker-4B | 0.683 |
| Qwen3-Embedding-4B | 0.615 |
| Text-Embedding-3-Large | 0.557 |
| BGE-m3 | 0.511 |
| BM25 | 0.416 |
| *random ranking* | *≈ 0.29* |

Reasoning-trained retrievers lead, and every system struggles more on symbol-heavy branches (Algebra, Calculus).
See [Reference results](docs/benchmark.md#reference-results) and the paper for the full table.

## Documentation

| | |
|---|---|
| [The benchmark](docs/benchmark.md) | what SABER-Math measures and how it was built |
| [Installation](docs/installation.md) | install options, vLLM, API keys |
| [Running `evaluate()`](docs/evaluate.md) | arguments, report, instructions, recipes, troubleshooting |
| [Built-in processors](docs/processors.md) | Sentence Transformers, vLLM, OpenRouter options |
| [Writing a processor](docs/custom-processors.md) | evaluate any model, reranker or baseline |
| [API reference](docs/api.md) | all public signatures |
| [Command line](docs/cli.md) | `sabermath evaluate`, `download`, `help` |

## License

The benchmark annotations and code are released under [CC BY-SA 4.0](LICENCE). The problem and solution texts keep the
licenses of their original sources (see App. A.8 of the paper).

## Citation

```bibtex
@inproceedings{georgiev2026sabermath,
  title     = {{SABER}-Math: An Automated Reranking Benchmark for Mathematical Information Retrieval},
  author    = {Georgiev, Nikolay and Drencheva, Maria and Ibragimova, Kseniia and Petrov, Ivo and Dimitrov, Dimitar I. and Vechev, Martin},
  booktitle = {The 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026},
  url       = {https://openreview.net/forum?id=Tb2EKLtAS0},
  eprint    = {2606.29894},
  archivePrefix = {arXiv}
}
```
