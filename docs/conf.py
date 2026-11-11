import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

from sabermath.__version__ import __version__  # noqa: E402

project = "SABER-Math"
author = "INSAIT"
copyright = "2026, INSAIT"
release = __version__
version = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

autodoc_member_order = "bysource"
autodoc_typehints = "signature"
autodoc_mock_imports = ["vllm", "openai"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

html_theme = "furo"
html_title = f"SABER-Math {release}"
html_logo = "../images/sabermath_logo.png"

exclude_patterns = ["_build"]

# Third-party classes that appear in signatures but have no intersphinx inventory.
nitpick_ignore = [
    ("py:class", "datasets.arrow_dataset.Dataset"),
    ("py:class", "sentence_transformers.SentenceTransformer"),
    ("py:class", "vllm.LLM"),
]
