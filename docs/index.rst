SABER-Math
==========

``sabermath`` evaluates retrieval and reranking models on the SABER-Math
benchmark: for each of 1,000 math problems, the model ranks 150 candidate
solved problems, and the ranking is scored with nDCG@k against graded
relevance labels.

The package has two parts you work with:

* :func:`sabermath.evaluate`, which runs the benchmark and returns a report;
* **processors**, which wrap a model so ``evaluate`` can score with it. Some
  are built in (Sentence Transformers, vLLM, OpenRouter) and you can write
  your own.

.. code-block:: python

   import sabermath

   report = sabermath.evaluate("BAAI/bge-m3", tasks=["statement-full"])
   print(report.tasks[0].ndcg_at_k)

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   evaluate
   processors
   custom_processors

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
