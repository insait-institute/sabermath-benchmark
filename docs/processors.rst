Built-in processors
===================

A processor is the object ``evaluate`` asks for scores: given one query and its
candidate documents, it returns one score per document. All processors derive
from :class:`~sabermath.processors.ModelProcessor`:

.. code-block:: text

   ModelProcessor                      get_scores(query, documents) -> scores
   └── EmbeddingProcessor              get_scores = encode + cosine similarity
       ├── SentenceTransformersProcessor
       ├── VLLMProcessor
       ├── OpenRouterEmbeddingProcessor
       └── UnknownProcessor

You can let ``evaluate`` build a processor from a model id, or build one
yourself and pass the instance:

.. code-block:: python

   import sabermath
   from sabermath.processors import SentenceTransformersProcessor

   # evaluate builds it
   sabermath.evaluate("BAAI/bge-m3", init_kwargs={"device": "cuda"})

   # you build it
   processor = SentenceTransformersProcessor.from_huggingface("BAAI/bge-m3", device="cuda")
   sabermath.evaluate(processor)

Building it yourself lets you reuse one loaded model across several
``evaluate`` calls.

.. _embedding-options:

Options shared by embedding processors
--------------------------------------

Every :class:`~sabermath.processors.EmbeddingProcessor` (all built-ins, and
your own subclasses) scores a query by embedding it and its documents and
taking cosine similarity. Its ``get_scores`` understands these
``scores_kwargs``:

``query_prompt``, ``document_prompt``
   Text prepended to the query / to every document.

``query_suffix``, ``document_suffix``
   Text appended to the query / to every document.

``query_encode_kwargs``, ``document_encode_kwargs``
   Dicts of extra ``encode`` arguments for one side only. When either is given,
   documents and query are encoded in two separate ``encode`` calls.

Every other key is passed unchanged to ``encode`` for both sides.

.. code-block:: python

   # E5-style text prefixes
   sabermath.evaluate(
       "intfloat/multilingual-e5-large",
       scores_kwargs={"query_prompt": "query: ", "document_prompt": "passage: "},
   )

   # a named prompt from the model's Sentence Transformers config, query side only
   sabermath.evaluate(
       "Qwen/Qwen3-Embedding-0.6B",
       scores_kwargs={"query_encode_kwargs": {"prompt_name": "query"}, "batch_size": 32},
   )

Sentence Transformers
---------------------

:class:`~sabermath.processors.SentenceTransformersProcessor` is the default
for model ids.

.. code-block:: python

   from sentence_transformers import SentenceTransformer
   from sabermath.processors import SentenceTransformersProcessor

   SentenceTransformersProcessor.from_huggingface("BAAI/bge-m3", device="cuda")
   SentenceTransformersProcessor(SentenceTransformer("BAAI/bge-m3"), model_name="bge-m3")

* ``init_kwargs`` / ``from_huggingface(**kwargs)`` go to
  ``SentenceTransformer(...)``.
* ``scores_kwargs`` go to ``SentenceTransformer.encode`` (``batch_size``,
  ``prompt_name``, ``normalize_embeddings``, ...), plus:

  ``chunk_to_context`` (default ``False``)
     Split texts longer than the context into pieces, embed each, and average
     them, instead of truncating.

  ``context_length``
     Piece length in tokens. Defaults to the model's ``max_seq_length``, then
     the tokenizer's ``model_max_length``.

vLLM
----

:class:`~sabermath.processors.VLLMProcessor` runs embedding models with vLLM
(needs the ``vllm`` extra and a GPU).

.. code-block:: python

   sabermath.evaluate(
       "Qwen/Qwen3-Embedding-8B",
       use_vllm=True,
       init_kwargs={"max_model_len": 8192, "tensor_parallel_size": 2},
       scores_kwargs={"batch_size": 256},
   )

   # or, with an LLM you created yourself
   from vllm import LLM
   sabermath.evaluate(LLM(model="Qwen/Qwen3-Embedding-8B", runner="pooling"))

``init_kwargs`` / ``from_huggingface(...)`` arguments:

``pooler_config``
   Dict of vLLM ``PoolerConfig`` fields, e.g.
   ``{"pooling_type": "LAST", "normalize": True}``, or a ``PoolerConfig``.

``export_clean`` (default ``False``)
   Re-save the checkpoint through ``transformers.AutoModel`` first and load
   that copy, for checkpoints vLLM cannot read directly. The copy is cached
   under ``$SABERMATH_VLLM_EXPORT_DIR``, else ``/scratch/$USER/...`` if it
   exists, else ``~/.cache/sabermath/vllm_artifacts``.

anything else
   Passed to ``vllm.LLM``, overriding the defaults ``runner="pooling"``,
   ``gpu_memory_utilization=0.9``, ``trust_remote_code=True``,
   ``enforce_eager=True``.

``scores_kwargs`` accepts the shared options plus:

``batch_size``
   Texts per ``LLM.embed`` call. Default: all at once.

``truncate_to``
   Token limit per text. Defaults to the tokenizer's ``model_max_length``.
   vLLM rejects over-long inputs instead of truncating, so set this when you
   lower ``max_model_len``.

``chunk_to_context``, ``context_length``
   As for Sentence Transformers.

Other keys go to ``LLM.embed``.

OpenRouter
----------

:class:`~sabermath.processors.OpenRouterEmbeddingProcessor` calls any
embedding model hosted on `OpenRouter <https://openrouter.ai>`_.

.. code-block:: python

   from sabermath.processors import OpenRouterEmbeddingProcessor

   processor = OpenRouterEmbeddingProcessor("openai/text-embedding-3-large")
   sabermath.evaluate(processor, scores_kwargs={"max_concurrency": 50})

* The model name is the full OpenRouter id (``provider/model``), sent as is.
* The key comes from ``api_key=...`` or ``OPENROUTER_API_KEY``.
* ``scores_kwargs``: the shared options, ``max_concurrency`` (default 20),
  ``retries`` per text (default 9, with 1 s, 2 s, 4 s, ... back-off), and any
  argument of ``embeddings.create`` such as ``dimensions``.
* It runs its own event loop, so calling it from inside a running ``asyncio``
  loop raises :class:`RuntimeError`.

UnknownProcessor
----------------

``evaluate`` wraps objects it does not recognise in
:class:`~sabermath.processors.UnknownProcessor`. Scoring with it currently
raises :class:`NotImplementedError`, so write a processor instead; see
:doc:`custom_processors`.
