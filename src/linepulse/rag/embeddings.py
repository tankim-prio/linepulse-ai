"""CPU-friendly multilingual embeddings for LinePulse RAG."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from linepulse.rag.chunks import RetrievalChunk


EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"

EMBEDDING_MODEL_REVISION = (
    "614241f622f53c4eeff9890bdc4f31cfecc418b3"
)

EMBEDDING_DIMENSION = 384

QUERY_PREFIX = "query: "

PASSAGE_PREFIX = "passage: "


class EmbeddingError(RuntimeError):
    """Raised when embedding generation violates its contract."""


@dataclass(frozen=True)
class EmbeddingBatch:
    """Normalized vectors produced by one embedding operation."""

    model_name: str
    model_revision: str
    dimension: int
    vectors: np.ndarray


class MultilingualE5Embedder:
    """CPU-only multilingual E5 embedding service."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        batch_size: int = 8,
    ) -> None:

        if device != "cpu":
            raise ValueError(
                "LinePulse currently supports CPU embeddings only."
            )

        if batch_size < 1:
            raise ValueError(
                "batch_size must be at least 1."
            )

        self.device = device
        self.batch_size = batch_size

    @cached_property
    def model(
        self,
    ) -> SentenceTransformer:
        """Load the pinned embedding model lazily."""

        model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            revision=EMBEDDING_MODEL_REVISION,
            device=self.device,
        )

        dimension = (
            model.get_embedding_dimension()
        )

        if dimension != EMBEDDING_DIMENSION:
            raise EmbeddingError(
                "Unexpected embedding dimension: "
                f"{dimension}; expected "
                f"{EMBEDDING_DIMENSION}."
            )

        return model

    def _encode(
        self,
        texts: Sequence[str],
        *,
        prefix: str,
    ) -> EmbeddingBatch:
        """Encode text and enforce the embedding contract."""

        if not texts:
            raise ValueError(
                "At least one text is required."
            )

        prepared: list[str] = []

        for index, value in enumerate(
            texts
        ):

            if not isinstance(
                value,
                str,
            ):
                raise TypeError(
                    "Embedding text must be str; "
                    f"item {index} is "
                    f"{type(value).__name__}."
                )

            stripped = value.strip()

            if not stripped:
                raise ValueError(
                    "Embedding text cannot be empty; "
                    f"item {index}."
                )

            prepared.append(
                prefix
                + stripped
            )

        vectors = self.model.encode(
            prepared,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        vectors = np.asarray(
            vectors,
            dtype=np.float32,
        )

        expected_shape = (
            len(prepared),
            EMBEDDING_DIMENSION,
        )

        if vectors.shape != expected_shape:
            raise EmbeddingError(
                "Unexpected embedding shape: "
                f"{vectors.shape}; expected "
                f"{expected_shape}."
            )

        if not np.isfinite(
            vectors
        ).all():
            raise EmbeddingError(
                "Embedding vectors contain "
                "non-finite values."
            )

        norms = np.linalg.norm(
            vectors,
            axis=1,
        )

        if not np.allclose(
            norms,
            1.0,
            atol=1e-5,
        ):
            raise EmbeddingError(
                "Embedding vectors are not normalized."
            )

        return EmbeddingBatch(
            model_name=EMBEDDING_MODEL_NAME,
            model_revision=EMBEDDING_MODEL_REVISION,
            dimension=EMBEDDING_DIMENSION,
            vectors=vectors,
        )

    def embed_queries(
        self,
        queries: Sequence[str],
    ) -> EmbeddingBatch:
        """Embed search queries with the E5 query prefix."""

        return self._encode(
            queries,
            prefix=QUERY_PREFIX,
        )

    def embed_passages(
        self,
        passages: Sequence[str],
    ) -> EmbeddingBatch:
        """Embed evidence passages with the E5 passage prefix."""

        return self._encode(
            passages,
            prefix=PASSAGE_PREFIX,
        )

    def embed_chunks(
        self,
        chunks: Sequence[
            RetrievalChunk
        ],
    ) -> EmbeddingBatch:
        """Embed deterministic LinePulse retrieval chunks."""

        if not chunks:
            raise ValueError(
                "At least one retrieval chunk is required."
            )

        return self.embed_passages(
            [
                chunk.text
                for chunk in chunks
            ]
        )
