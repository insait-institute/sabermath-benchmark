Writing your own processor
==========================

To evaluate a model that is not covered by the :doc:`built-in processors
<processors>` (your own PyTorch model, an embedding API, a reranker, a lexical
baseline, ...), write a small class and pass an instance of it to
:func:`sabermath.evaluate`:

.. code-block:: python

   report = sabermath.evaluate(MyProcessor(...))

Pick the base class by how your model scores:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Base class
     - Use it when
     - You implement
   * - :class:`~sabermath.processors.EmbeddingProcessor`
     - the model maps a text to a vector
     - ``encode(texts) -> array``
   * - :class:`~sabermath.processors.ModelProcessor`
     - the model scores (query, document) pairs, or anything that is not
       "embed then cosine"
     - ``get_scores(query, documents) -> scores``

Both import from ``sabermath.processors`` (``EmbeddingProcessor`` also from
``sabermath``).

How ``evaluate`` calls a processor
----------------------------------

For each task, and for each of the 1,000 queries, ``evaluate`` makes one call::

   processor.get_scores(query, documents, show_progress_bar=False, **scores_kwargs)

* ``query`` is a ``str``: the problem, or problem + solution for
  ``full-full``, already wrapped in the instruction if one was given.
* ``documents`` is a ``list[str]`` of the query's 150 candidates.
* ``scores_kwargs`` is the dict you passed to ``evaluate`` (empty by default).

The candidates are then ranked by the returned scores, highest first.

The processor contract
----------------------

Every processor, whichever base it uses, has three parts:

``processor`` (class attribute, **required**)
   A short name for the backend, reported as ``Report.processor``. It must be
   defined on your class (it may be ``None``); without it ``evaluate`` fails
   with :class:`AttributeError` when building the report, *after* all
   scoring has run.

``model`` (property, optional)
   The model's name, reported as ``Report.model``. The default returns
   ``None``, which is reported as ``"unknown"``.

``get_scores`` (method)
   Inherited from ``EmbeddingProcessor``, or written by you for
   ``ModelProcessor``.

Configuration belongs in your ``__init__``: ``evaluate`` rejects
``init_kwargs`` when given a processor instance. Per-run options go through
``scores_kwargs``.

Subclassing ``EmbeddingProcessor``
----------------------------------

Implement one method:

.. code-block:: python

   def encode(self, texts: list[str], show_progress_bar: bool = True, **kwargs) -> np.ndarray:
       ...

It must return an array of shape ``(len(texts), dim)``, one row per input text
in input order. Rows must not be all zeros (cosine similarity is undefined and
``evaluate`` raises :class:`ValueError`).

The inherited ``get_scores`` then:

1. adds ``query_prompt`` / ``query_suffix`` / ``document_prompt`` /
   ``document_suffix`` if given in ``scores_kwargs``;
2. calls ``encode(documents + [query], **kwargs)`` once, so the query is the
   *last* row; or, if ``query_encode_kwargs`` or ``document_encode_kwargs`` is
   given, calls ``encode(documents, ...)`` and ``encode([query], ...)``
   separately with those extra arguments;
3. returns the cosine similarity of each document to the query.

``kwargs`` in ``encode`` receives every ``scores_kwargs`` key except the six
options above. Accept ``**kwargs`` so unexpected keys do not break your
method, or name the ones you support.

Example: a Hugging Face model with mean pooling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import torch
   from transformers import AutoModel, AutoTokenizer

   import sabermath
   from sabermath.processors import EmbeddingProcessor


   class MeanPoolingProcessor(EmbeddingProcessor):
       processor = "transformers-mean-pooling"

       def __init__(self, model_name, device="cpu", max_length=512):
           self._model_name = model_name
           self._device = device
           self._max_length = max_length
           self._tokenizer = AutoTokenizer.from_pretrained(model_name)
           self._model = AutoModel.from_pretrained(model_name).to(device).eval()

       @property
       def model(self):
           return self._model_name

       @torch.no_grad()
       def encode(self, texts, show_progress_bar=True, batch_size=32, **kwargs):
           vectors = []
           for start in range(0, len(texts), batch_size):
               batch = self._tokenizer(
                   texts[start : start + batch_size],
                   padding=True,
                   truncation=True,
                   max_length=self._max_length,
                   return_tensors="pt",
               ).to(self._device)
               hidden = self._model(**batch).last_hidden_state
               mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
               pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
               vectors.append(pooled.cpu().numpy())
           return np.concatenate(vectors)


   processor = MeanPoolingProcessor("BAAI/bge-small-en-v1.5", device="cuda")
   report = sabermath.evaluate(processor, scores_kwargs={"batch_size": 64})

``batch_size`` in ``scores_kwargs`` arrives as the ``batch_size`` argument of
``encode``.

Example: different handling for queries and documents
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many embedding APIs want to know whether a text is a query or a document. Give
``encode`` a parameter for it and set it per side with
``query_encode_kwargs`` / ``document_encode_kwargs``:

.. code-block:: python

   class MyAPIProcessor(EmbeddingProcessor):
       processor = "my-api"

       def __init__(self, client, model_name):
           self._client = client
           self._model_name = model_name

       @property
       def model(self):
           return self._model_name

       def encode(self, texts, show_progress_bar=True, input_type="document", **kwargs):
           response = self._client.embed(
               texts, model=self._model_name, input_type=input_type, **kwargs
           )
           return np.asarray(response)


   report = sabermath.evaluate(
       MyAPIProcessor(client, "embed-v1"),
       scores_kwargs={"query_encode_kwargs": {"input_type": "query"}},
   )

Documents are encoded with the default ``input_type="document"``, the query
with ``input_type="query"``.

Example: caching embeddings
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``encode`` is called for every query, and the same document shows up in the
candidate lists of several queries and across tasks. Nothing is cached for
you. If encoding is slow or costs money, cache inside your processor:

.. code-block:: python

   class CachedMeanPoolingProcessor(MeanPoolingProcessor):
       def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)
           self._cache = {}

       def encode(self, texts, show_progress_bar=True, **kwargs):
           options = repr(sorted(kwargs.items()))
           missing = [t for t in dict.fromkeys(texts) if (t, options) not in self._cache]
           if missing:
               vectors = super().encode(missing, show_progress_bar, **kwargs)
               for text, vector in zip(missing, vectors):
                   self._cache[(text, options)] = vector
           return np.stack([self._cache[(t, options)] for t in texts])

The key includes the ``encode`` options so that, for example, query-side and
document-side embeddings of the same text are stored separately.

Changing the similarity function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To score with something other than cosine (say, dot product), override
``get_scores`` and call ``self.encode`` yourself:

.. code-block:: python

   class DotProductProcessor(MeanPoolingProcessor):
       def get_scores(self, query, documents, *, show_progress_bar=True, **kwargs):
           vectors = self.encode(documents + [query], show_progress_bar, **kwargs)
           return vectors[:-1] @ vectors[-1]

An overridden ``get_scores`` no longer handles the prompt, suffix and per-side
options listed above unless you implement them.

Subclassing ``ModelProcessor``
------------------------------

Implement ``get_scores``:

.. code-block:: python

   def get_scores(
       self,
       query: str,
       documents: list[str],
       *,
       show_progress_bar: bool = True,
       **kwargs,
   ) -> list[float] | np.ndarray | None:
       ...

Return one number per document, in the order of ``documents``; higher means
more relevant. A list or a NumPy array both work. ``kwargs`` receives all of
``scores_kwargs``.

Example: a cross-encoder reranker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sentence_transformers import CrossEncoder

   import sabermath
   from sabermath.processors import ModelProcessor


   class CrossEncoderProcessor(ModelProcessor):
       processor = "cross-encoder"

       def __init__(self, model_name, **kwargs):
           self._model_name = model_name
           self._model = CrossEncoder(model_name, **kwargs)

       @property
       def model(self):
           return self._model_name

       def get_scores(self, query, documents, *, show_progress_bar=True, batch_size=32, **kwargs):
           pairs = [(query, document) for document in documents]
           return self._model.predict(
               pairs, batch_size=batch_size, show_progress_bar=show_progress_bar
           )


   report = sabermath.evaluate(
       CrossEncoderProcessor("cross-encoder/ms-marco-MiniLM-L6-v2"),
       scores_kwargs={"batch_size": 150},
   )

Example: a lexical baseline
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A processor does not need a neural model. This BM25 scorer computes term
statistics over the query's 150 candidates:

.. code-block:: python

   import math
   import re
   from collections import Counter

   from sabermath.processors import ModelProcessor


   def tokenize(text):
       return re.findall(r"\w+", text.lower())


   class BM25Processor(ModelProcessor):
       processor = "bm25"

       def __init__(self, k1=1.5, b=0.75):
           self.k1 = k1
           self.b = b

       @property
       def model(self):
           return f"bm25-k1={self.k1}-b={self.b}"

       def get_scores(self, query, documents, *, show_progress_bar=True, **kwargs):
           docs = [tokenize(d) for d in documents]
           n = len(docs)
           avg_len = sum(len(d) for d in docs) / n or 1.0
           df = Counter(term for d in docs for term in set(d))
           query_terms = set(tokenize(query))

           scores = []
           for d in docs:
               tf = Counter(d)
               score = 0.0
               for term in query_terms & tf.keys():
                   idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                   norm = 1 - self.b + self.b * len(d) / avg_len
                   score += idf * tf[term] * (self.k1 + 1) / (tf[term] + self.k1 * norm)
               scores.append(score)
           return scores

.. _skipping-queries:

Skipping a query
~~~~~~~~~~~~~~~~

Return ``None`` from ``get_scores`` when the model cannot score a query (for
example an API refused the request). The query is recorded as ``None`` in the
per-query results and left out of the task and branch averages, so check how
many were skipped before comparing models:

.. code-block:: python

   report, ndcgs = sabermath.evaluate(processor, return_ndcgs=True)
   skipped = sum(v is None for v in ndcgs["statement-full"])

Testing your processor
----------------------

Before a full run, check it on a few queries:

.. code-block:: python

   from sabermath.data import load_data

   queries, documents = load_data()
   report = sabermath.evaluate(
       processor,
       tasks=["statement-full"],
       queries=queries.select(range(20)),
       documents=documents,
   )

A model that ranks at random scores about 0.29 nDCG@10 with the default
exponent gain, on every task; a working retriever should be clearly above
that. On only 20 queries, expect some noise.
