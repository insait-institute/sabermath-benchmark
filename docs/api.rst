API reference
=============

Evaluation
----------

.. autofunction:: sabermath.evaluate

.. autofunction:: sabermath.data.load_data

Results
-------

.. autoclass:: sabermath.Report
   :members:

.. autoclass:: sabermath.TaskResult

.. autoclass:: sabermath.BranchResult

.. py:data:: sabermath.Task

   ``Literal["statement-statement", "statement-full", "full-full"]``

.. py:data:: sabermath.Branch

   ``Literal["Algebra", "Geometry", "Number Theory", "Combinatorics", "Calculus and Analysis"]``

Processors
----------

.. autoclass:: sabermath.processors.ModelProcessor
   :members: model, get_scores
   :undoc-members:

.. autoclass:: sabermath.processors.EmbeddingProcessor
   :members: encode, get_scores, encode_statements, encode_full
   :undoc-members:

.. autoclass:: sabermath.processors.SentenceTransformersProcessor
   :members: from_huggingface, encode
   :undoc-members:

.. autoclass:: sabermath.processors.VLLMProcessor
   :members: from_huggingface, encode
   :undoc-members:

.. autoclass:: sabermath.processors.OpenRouterEmbeddingProcessor
   :members: encode, encode_async
   :undoc-members:

.. autoclass:: sabermath.processors.UnknownProcessor

Metrics
-------

.. autofunction:: sabermath.metrics.compute_ndcg_at_k

.. autofunction:: sabermath.metrics.dcg_at_k

.. autofunction:: sabermath.metrics.cosine_similarity

Instructions
------------

.. py:data:: sabermath.instructions.INSTRUCTIONS
   :type: dict[str, str | None]

   The instructions used in the paper, keyed ``p0``, ``p1``, ``p2``, ``p3``
   and ``pm``. See :ref:`instructions`.

.. py:data:: sabermath.instructions.INSTRUCTION_TEMPLATES
   :type: dict[str, str]

   Named format strings with ``{instruction}`` and ``{query}`` fields. Built
   in: ``"canonical"`` (``"Instruct: {instruction}\nQuery: {query}"``).

.. autofunction:: sabermath.instructions.format_instructed_query
