Installation
============

SABER-Math needs Python 3.10 or newer.

.. code-block:: bash

   git clone https://github.com/insait-institute/sabermath
   cd sabermath
   python -m pip install -e .

The base install covers Sentence Transformers models and embedding models
served through OpenRouter.

vLLM
----

To run embedding models with `vLLM <https://github.com/vllm-project/vllm>`_,
install the ``vllm`` extra. This needs a CUDA GPU.

.. code-block:: bash

   python -m pip install -e ".[vllm]"

The extra pins ``vllm==0.26.0``. Other versions may name their options
differently.

API keys
--------

OpenRouter models need an API key, read from the ``OPENROUTER_API_KEY``
environment variable unless you pass ``api_key=...`` explicitly:

.. code-block:: bash

   export OPENROUTER_API_KEY=<your-key>

The datasets are public; set ``HF_TOKEN`` only if you hit Hugging Face rate
limits.

Building these docs
-------------------

.. code-block:: bash

   python -m pip install -r docs/requirements.txt
   sphinx-build -b html docs docs/_build/html
