"""Deterministic in-memory retrieval for LinePulse RAG."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Sequence

import numpy as np

from linepulse.rag.chunks import RetrievalChunk
from linepulse.rag.embeddings import (
    EMBEDDING_DIMENSION,
    MultilingualE5Embedder,
)


DEFAULT_TOP_K = 3


class RetrievalError(RuntimeError):
    """Raised when retrieval data violates its contract."""


@dataclass(frozen=True)
class RetrievalResult:
    """One ranked evidence result for a user query."""

    rank: int
    score: float
    chunk: RetrievalChunk


class InMemoryRetriever:
    """Cosine-equivalent retrieval over normalized E5 vectors."""

    def __init__(
        self,
        chunks: Sequence[RetrievalChunk],
        *,
        embedder: MultilingualE5Embedder | None = None,
    ) -> None:

        if not chunks:
            raise ValueError(
                "At least one retrieval chunk is required."
            )

        self.chunks = tuple(
            chunks
        )

        chunk_ids = [
            chunk.chunk_id
            for chunk in self.chunks
        ]

        if len(
            set(
                chunk_ids
            )
        ) != len(
            chunk_ids
        ):
            raise ValueError(
                "Retrieval chunk IDs must be unique."
            )

        self.embedder = (
            embedder
            if embedder is not None
            else MultilingualE5Embedder()
        )

    @property
    def chunk_count(
        self,
    ) -> int:
        """Return the number of indexed chunks."""

        return len(
            self.chunks
        )

    @cached_property
    def passage_vectors(
        self,
    ) -> np.ndarray:
        """Embed and cache the indexed chunks."""

        batch = self.embedder.embed_chunks(
            self.chunks
        )

        vectors = np.asarray(
            batch.vectors,
            dtype=np.float32,
        )

        expected_shape = (
            self.chunk_count,
            EMBEDDING_DIMENSION,
        )

        if vectors.shape != expected_shape:
            raise RetrievalError(
                "Unexpected passage embedding shape: "
                f"{vectors.shape}; expected "
                f"{expected_shape}."
            )

        if not np.isfinite(
            vectors
        ).all():
            raise RetrievalError(
                "Passage embeddings contain "
                "non-finite values."
            )

        return vectors

    def search(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_TOP_K,
    ) -> tuple[RetrievalResult, ...]:
        """Return deterministically ranked evidence for one query."""

        if not isinstance(
            query,
            str,
        ):
            raise TypeError(
                "query must be str."
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "query cannot be empty."
            )

        if not isinstance(
            top_k,
            int,
        ) or isinstance(
            top_k,
            bool,
        ):
            raise TypeError(
                "top_k must be int."
            )

        if top_k < 1:
            raise ValueError(
                "top_k must be at least 1."
            )

        if top_k > self.chunk_count:
            raise ValueError(
                "top_k cannot exceed "
                f"the chunk count ({self.chunk_count})."
            )

        query_batch = (
            self.embedder.embed_queries(
                [
                    query,
                ]
            )
        )

        query_vectors = np.asarray(
            query_batch.vectors,
            dtype=np.float32,
        )

        expected_query_shape = (
            1,
            EMBEDDING_DIMENSION,
        )

        if (
            query_vectors.shape
            != expected_query_shape
        ):
            raise RetrievalError(
                "Unexpected query embedding shape: "
                f"{query_vectors.shape}; expected "
                f"{expected_query_shape}."
            )

        if not np.isfinite(
            query_vectors
        ).all():
            raise RetrievalError(
                "Query embedding contains "
                "non-finite values."
            )

        scores = (
            self.passage_vectors
            @ query_vectors[0]
        )

        if scores.shape != (
            self.chunk_count,
        ):
            raise RetrievalError(
                "Unexpected retrieval score shape."
            )

        if not np.isfinite(
            scores
        ).all():
            raise RetrievalError(
                "Retrieval scores contain "
                "non-finite values."
            )

        ranked_indices = sorted(
            range(
                self.chunk_count
            ),
            key=lambda index: (
                -float(
                    scores[
                        index
                    ]
                ),
                self.chunks[
                    index
                ].chunk_id,
            ),
        )

        results = []

        for rank, index in enumerate(
            ranked_indices[
                :top_k
            ],
            start=1,
        ):

            results.append(
                RetrievalResult(
                    rank=rank,
                    score=float(
                        scores[
                            index
                        ]
                    ),
                    chunk=self.chunks[
                        index
                    ],
                )
            )

        return tuple(
            results
        )
