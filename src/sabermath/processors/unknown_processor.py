import numpy as np

from .embedding_processor import EmbeddingProcessor


class UnknownProcessor(EmbeddingProcessor):
    processor = None

    def __init__(self, model):
        self._has_encode = callable(getattr(model, "encode", None))
        has_get_scores = callable(getattr(model, "get_scores", None))

        if not self._has_encode and not has_get_scores:
            raise ValueError(
                "Input model must have a callable .encode() or .get_scores()"
            )

        self._model = model

    def encode(self, texts, show_progress_bar=True, **kwargs):
        if not self._has_encode:
            raise NotImplementedError("This model does not support .encode()")
        return np.asarray(self._model.encode(texts, **kwargs))

    def get_scores(self, query, documents, *, show_progress_bar=True, **kwargs):
        raise NotImplementedError
