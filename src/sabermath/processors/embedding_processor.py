from abc import abstractmethod

from datasets import Dataset
import numpy as np

from .base import ModelProcessor

AFFIX_KEYS = (
    "query_prompt",
    "document_prompt",
    "query_suffix",
    "document_suffix",
    "query_encode_kwargs",
    "document_encode_kwargs",
)


def split_affix_kwargs(kwargs: dict) -> tuple[dict, dict]:
    affixes = {k: v for k, v in kwargs.items() if k in AFFIX_KEYS}
    rest = {k: v for k, v in kwargs.items() if k not in AFFIX_KEYS}
    return affixes, rest


def apply_affixes(
    texts: list[str], prompt: str | None, suffix: str | None
) -> list[str]:
    if not prompt and not suffix:
        return list(texts)
    head = prompt or ""
    tail = suffix or ""
    return [f"{head}{text}{tail}" for text in texts]


class EmbeddingProcessor(ModelProcessor):
    @classmethod
    def from_huggingface(cls, model_name: str):
        raise NotImplementedError

    def _cosine_similarity(
        self, query_emb: np.ndarray, docs_embs: np.ndarray
    ) -> np.ndarray:
        q_norm = np.linalg.norm(query_emb)
        d_norms = np.linalg.norm(docs_embs, axis=1, keepdims=True)

        if q_norm == 0:
            raise ValueError("Query embedding has zero norm")

        if np.any(d_norms == 0):
            raise ValueError("At least one document embedding has zero norm")

        query_emb = query_emb / q_norm
        docs_embs = docs_embs / d_norms
        return docs_embs @ query_emb

    def get_scores(
        self,
        query: str,
        documents: list[str],
        *,
        show_progress_bar: bool = True,
        query_prompt: str | None = None,
        document_prompt: str | None = None,
        query_suffix: str | None = None,
        document_suffix: str | None = None,
        query_encode_kwargs: dict | None = None,
        document_encode_kwargs: dict | None = None,
        **kwargs,
    ) -> np.ndarray:
        query = apply_affixes([query], query_prompt, query_suffix)[0]
        documents = apply_affixes(documents, document_prompt, document_suffix)

        if query_encode_kwargs or document_encode_kwargs:
            document_embeddings = self.encode(
                documents,
                show_progress_bar=show_progress_bar,
                **{**kwargs, **(document_encode_kwargs or {})},
            )
            query_embedding = self.encode(
                [query],
                show_progress_bar=show_progress_bar,
                **{**kwargs, **(query_encode_kwargs or {})},
            )[0]
        else:
            N_d = len(documents)
            embeddings = self.encode(
                documents + [query], show_progress_bar=show_progress_bar, **kwargs
            )
            query_embedding = embeddings[N_d]
            document_embeddings = embeddings[:N_d]

        return self._cosine_similarity(
            np.asarray(query_embedding), np.asarray(document_embeddings)
        )

    @abstractmethod
    def encode(
        self, texts: list[str], show_progress_bar: bool = True, **kwargs
    ) -> np.ndarray:
        pass

    def encode_statements(
        self,
        ds: Dataset,
        show_progress_bar: bool = True,
        *,
        statement_column: str = "problem",
        **kwargs,
    ) -> np.ndarray:
        statements = list(ds[statement_column])

        return self.encode(
            statements,
            show_progress_bar=show_progress_bar,
            **kwargs,
        )

    def encode_full(
        self,
        ds: Dataset,
        show_progress_bar: bool = True,
        *,
        statement_column: str = "problem",
        solution_column: str = "solution",
        **kwargs,
    ) -> np.ndarray:
        statements = list(ds[statement_column])
        solutions = list(ds[solution_column])

        full_texts = [
            f"Problem: {statement}\n\nSolution: {solution}"
            for statement, solution in zip(statements, solutions)
        ]

        return self.encode(full_texts, show_progress_bar=show_progress_bar, **kwargs)
