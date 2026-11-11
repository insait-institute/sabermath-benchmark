Running ``evaluate()``
======================

:func:`sabermath.evaluate` is the single entry point. It loads the benchmark,
turns ``model`` into a :doc:`processor <processors>`, scores every query, and
returns a :class:`~sabermath.Report`.

.. code-block:: python

   import sabermath

   report = sabermath.evaluate("BAAI/bge-m3")

   for task in report.tasks:
       print(task.task, round(task.ndcg_at_k, 4))
       for branch in task.branches:
           print("   ", branch.branch, round(branch.ndcg_at_k, 4))

Signature
---------

.. code-block:: python

   sabermath.evaluate(
       model,
       tasks=None,
       k=10,
       *,
       dcg_variant="exponent",
       use_vllm=False,
       init_kwargs=None,
       scores_kwargs=None,
       on_progress=None,
       instruction=None,
       instruction_template="canonical",
       verbose=True,
       show_progress_bars=True,
       queries=None,
       documents=None,
       return_ndcgs=False,
   )

What ``model`` can be
---------------------

.. list-table::
   :header-rows: 1

   * - You pass
     - ``evaluate`` uses
   * - a Hugging Face model id (``str``)
     - :class:`~sabermath.processors.SentenceTransformersProcessor`, or
       :class:`~sabermath.processors.VLLMProcessor` with ``use_vllm=True``
   * - a ``sentence_transformers.SentenceTransformer``
     - :class:`~sabermath.processors.SentenceTransformersProcessor`
   * - a ``vllm.LLM``
     - :class:`~sabermath.processors.VLLMProcessor`
   * - any :class:`~sabermath.processors.ModelProcessor` instance
     - that processor, as is

Anything else (for example a raw PyTorch model) must be wrapped in a processor
first; see :doc:`custom_processors`.

Arguments
---------

``tasks``
   Which tasks to run; ``None`` runs all three. Each task decides what text the
   model sees:

   .. list-table::
      :header-rows: 1

      * - Task
        - Query
        - Documents
      * - ``"statement-statement"``
        - problem
        - problem
      * - ``"statement-full"`` (main)
        - problem
        - problem + solution
      * - ``"full-full"``
        - problem + solution
        - problem + solution

   "Problem + solution" is the string
   ``"Problem: {problem}\n\nSolution: {solution}"``. Tasks run in the order
   above. An unknown name raises :class:`ValueError`.

``k``
   Rank cut-off for nDCG. Default ``10``.

``dcg_variant``
   Gain function: ``"exponent"`` (default, gain :math:`2^r - 1`) or
   ``"linear"`` (gain :math:`r`), where :math:`r` is the relevance label
   (0–5). The discount is :math:`1 / \log_2(\text{rank} + 1)`.

``use_vllm``
   Load a model id with vLLM instead of Sentence Transformers. Only valid when
   ``model`` is a string; otherwise :class:`RuntimeError`.

``init_kwargs``
   Arguments for building the processor, used only when ``evaluate`` builds it
   (``model`` is a string, a ``SentenceTransformer`` or a ``vllm.LLM``). See
   :doc:`processors` for what each backend accepts. Passing ``init_kwargs``
   together with a processor instance raises :class:`RuntimeError`; configure
   the instance yourself instead.

``scores_kwargs``
   Keyword arguments forwarded to **every** ``processor.get_scores(...)`` call.
   This is how per-run options (batch size, prompts, ...) reach the model,
   including your own processors. See :ref:`embedding-options`.

``instruction``, ``instruction_template``
   Wrap every query (never the documents) in an instruction. With the default
   ``"canonical"`` template the query becomes
   ``"Instruct: {instruction}\nQuery: {query}"``. See :ref:`instructions`.

``on_progress``
   A no-argument function called after each query is scored.

``verbose``, ``show_progress_bars``
   Print status messages; show a progress bar over queries.

``queries``, ``documents``
   Use these ``datasets.Dataset`` objects instead of downloading the official
   ones. Both must be given. Handy for quick runs on a subset:

   .. code-block:: python

      from sabermath.data import load_data

      queries, documents = load_data()
      report = sabermath.evaluate(
          "BAAI/bge-m3",
          queries=queries.select(range(50)),
          documents=documents,      # keep it whole: candidates index into it
      )

``return_ndcgs``
   Also return the nDCG of every query. See below.

Return value
------------

A :class:`~sabermath.Report`:

.. code-block:: python

   report.model          # processor.model, or "unknown"
   report.processor      # processor.processor, or "unknown"
   report.dcg_variant
   report.k
   report.tasks          # list[TaskResult], one per task that was run

   task = report.tasks[0]
   task.task             # e.g. "statement-full"
   task.ndcg_at_k        # mean nDCG@k over the task's queries
   task.branches         # list[BranchResult], one per branch of mathematics

   report.to_dict()      # plain dicts and lists, ready for json.dump

Branches are Algebra, Geometry, Number Theory, Combinatorics, and Calculus and
Analysis. A branch score is the mean over the queries tagged with that branch;
a query can carry two branches and then counts for both.

With ``return_ndcgs=True`` you get ``(report, ndcgs)``. ``ndcgs`` maps each
task name to a list with one value per query, in dataset order:

.. code-block:: python

   report, ndcgs = sabermath.evaluate("BAAI/bge-m3", return_ndcgs=True)
   ndcgs["statement-full"][:3]   # [0.52, 0.31, None, ...]

``None`` means the processor returned no scores for that query (see
:ref:`skipping-queries`). Such queries are left out of every mean; a branch
with no scored queries gets ``0.0``.

.. _instructions:

Instructions
------------

:data:`sabermath.instructions.INSTRUCTIONS` holds the instructions used in the
paper, under the keys ``p0`` (``None``, no instruction), ``p1``, ``p2``,
``p3`` and ``pm``:

.. code-block:: python

   from sabermath.instructions import INSTRUCTIONS

   report = sabermath.evaluate(
       "Qwen/Qwen3-Embedding-0.6B",
       instruction=INSTRUCTIONS["pm"],
   )

To use another format, register a template with ``{instruction}`` and
``{query}`` fields:

.. code-block:: python

   from sabermath.instructions import INSTRUCTION_TEMPLATES

   INSTRUCTION_TEMPLATES["plain"] = "{instruction} {query}"
   report = sabermath.evaluate(model, instruction="...", instruction_template="plain")

The instruction is applied by ``evaluate`` before the processor sees the
query, so it works with every processor, including your own.
