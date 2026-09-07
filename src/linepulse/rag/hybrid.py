"""Deterministic CPU-friendly lexical and hybrid RAG retrieval.

This module deliberately uses result contracts that are distinct from the
v1 dense RetrievalResult contract.

In particular, RRF fusion scores are NOT cosine similarities and must never
be passed to the frozen v1 AnswerabilityGate as if they were dense scores.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import unicodedata
from typing import Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from linepulse.rag.chunks import RetrievalChunk
from linepulse.rag.embeddings import MultilingualE5Embedder
from linepulse.rag.retrieval import InMemoryRetriever


LEXICAL_ANALYZER = "char_wb"
LEXICAL_NGRAM_RANGE = (3, 5)
DEFAULT_RRF_K = 60


class HybridRetrievalError(RuntimeError):
    """Raised when lexical or hybrid retrieval violates its contract."""


@dataclass(frozen=True)
class LexicalRetrievalResult:
    """One lexical retrieval result.

    lexical_score is TF-IDF cosine similarity and is intentionally not named
    ``score`` so this result cannot accidentally satisfy the frozen v1
    EvidenceService retrieval contract.
    """

    rank: int
    lexical_score: float
    chunk: RetrievalChunk

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


@dataclass(frozen=True)
class HybridRetrievalResult:
    """One dense + lexical RRF result."""

    rank: int
    fusion_score: float
    dense_rank: int
    dense_score: float
    lexical_rank: int
    lexical_score: float
    chunk: RetrievalChunk

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


def _normalize_lexical_text(
    value: str,
) -> str:
    """Normalize one lexical string deterministically."""

    if not isinstance(
        value,
        str,
    ):
        raise TypeError(
            "lexical text must be str."
        )

    value = unicodedata.normalize(
        "NFKC",
        value,
    )

    value = value.casefold()

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def _chunk_lexical_text(
    chunk: RetrievalChunk,
) -> str:
    """Build lexical text without filenames, IDs, or provenance noise."""

    parts = (
        chunk.document_title,
        chunk.section_title,
        chunk.english_text,
        chunk.bangla_text,
    )

    return _normalize_lexical_text(
        "\n".join(parts)
    )


def _validate_search_input(
    query: str,
    top_k: int,
    chunk_count: int,
) -> str:
    """Validate common lexical/hybrid query inputs."""

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

    if top_k > chunk_count:
        raise ValueError(
            "top_k cannot exceed "
            f"the chunk count ({chunk_count})."
        )

    return query


class LexicalRetriever:
    """Sparse multilingual character TF-IDF retrieval.

    Character n-grams avoid language-specific tokenizers and work naturally
    across English, Bangla, and code-mixed text.

    The corpus matrix is fitted once at construction time. Query-time work
    only transforms the query and performs sparse similarity scoring.
    """

    def __init__(
        self,
        chunks: Sequence[RetrievalChunk],
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
            set(chunk_ids)
        ) != len(chunk_ids):
            raise ValueError(
                "Retrieval chunk IDs must be unique."
            )

        self.vectorizer = TfidfVectorizer(
            analyzer=LEXICAL_ANALYZER,
            ngram_range=LEXICAL_NGRAM_RANGE,
            lowercase=False,
            norm="l2",
            sublinear_tf=True,
            dtype=np.float32,
        )

        corpus_text = [
            _chunk_lexical_text(
                chunk
            )
            for chunk in self.chunks
        ]

        matrix = self.vectorizer.fit_transform(
            corpus_text
        )

        matrix = sparse.csr_matrix(
            matrix,
            dtype=np.float32,
        )

        if matrix.shape[0] != self.chunk_count:
            raise HybridRetrievalError(
                "Unexpected lexical matrix row count."
            )

        if matrix.shape[1] < 1:
            raise HybridRetrievalError(
                "Lexical vocabulary is empty."
            )

        if not np.isfinite(
            matrix.data
        ).all():
            raise HybridRetrievalError(
                "Lexical matrix contains non-finite values."
            )

        self.matrix = matrix

    @property
    def chunk_count(
        self,
    ) -> int:
        return len(
            self.chunks
        )

    @property
    def vocabulary_size(
        self,
    ) -> int:
        return int(
            self.matrix.shape[1]
        )

    @property
    def matrix_nnz(
        self,
    ) -> int:
        return int(
            self.matrix.nnz
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
    ) -> tuple[LexicalRetrievalResult, ...]:
        """Return deterministic lexical ranking."""

        query = _validate_search_input(
            query,
            top_k,
            self.chunk_count,
        )

        normalized_query = (
            _normalize_lexical_text(
                query
            )
        )

        query_vector = (
            self.vectorizer.transform(
                [
                    normalized_query,
                ]
            )
        )

        query_vector = sparse.csr_matrix(
            query_vector,
            dtype=np.float32,
        )

        if query_vector.shape != (
            1,
            self.vocabulary_size,
        ):
            raise HybridRetrievalError(
                "Unexpected lexical query-vector shape."
            )

        raw_scores = (
            self.matrix
            @ query_vector.T
        )

        scores = np.asarray(
            raw_scores.toarray(),
            dtype=np.float32,
        ).reshape(-1)

        if scores.shape != (
            self.chunk_count,
        ):
            raise HybridRetrievalError(
                "Unexpected lexical score shape."
            )

        if not np.isfinite(
            scores
        ).all():
            raise HybridRetrievalError(
                "Lexical scores contain non-finite values."
            )

        ranked_indices = sorted(
            range(
                self.chunk_count
            ),
            key=lambda index: (
                -float(
                    scores[index]
                ),
                self.chunks[
                    index
                ].chunk_id,
            ),
        )

        results = []

        for rank, index in enumerate(
            ranked_indices[:top_k],
            start=1,
        ):

            results.append(
                LexicalRetrievalResult(
                    rank=rank,
                    lexical_score=float(
                        scores[index]
                    ),
                    chunk=self.chunks[
                        index
                    ],
                )
            )

        return tuple(
            results
        )


class HybridRetriever:
    """Fuse dense E5 and lexical rankings with reciprocal-rank fusion.

    RRF operates on ranks rather than combining incomparable raw dense and
    lexical score scales.

    The default RRF constant is an engineering baseline and is not tuned from
    the old v1 held-out evaluation set.
    """

    def __init__(
        self,
        chunks: Sequence[RetrievalChunk],
        *,
        embedder: MultilingualE5Embedder | None = None,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:

        if not isinstance(
            rrf_k,
            int,
        ) or isinstance(
            rrf_k,
            bool,
        ):
            raise TypeError(
                "rrf_k must be int."
            )

        if rrf_k < 1:
            raise ValueError(
                "rrf_k must be at least 1."
            )

        self.chunks = tuple(
            chunks
        )

        if not self.chunks:
            raise ValueError(
                "At least one retrieval chunk is required."
            )

        self.rrf_k = rrf_k

        self.dense_retriever = (
            InMemoryRetriever(
                self.chunks,
                embedder=embedder,
            )
        )

        self.lexical_retriever = (
            LexicalRetriever(
                self.chunks
            )
        )

    @property
    def chunk_count(
        self,
    ) -> int:
        return len(
            self.chunks
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
    ) -> tuple[HybridRetrievalResult, ...]:
        """Return deterministic dense + lexical RRF ranking."""

        query = _validate_search_input(
            query,
            top_k,
            self.chunk_count,
        )

        dense_results = (
            self.dense_retriever.search(
                query,
                top_k=self.chunk_count,
            )
        )

        lexical_results = (
            self.lexical_retriever.search(
                query,
                top_k=self.chunk_count,
            )
        )

        dense_by_id = {
            result.chunk.chunk_id:
                result
            for result in dense_results
        }

        lexical_by_id = {
            result.chunk.chunk_id:
                result
            for result in lexical_results
        }

        expected_ids = {
            chunk.chunk_id
            for chunk in self.chunks
        }

        if set(
            dense_by_id
        ) != expected_ids:
            raise HybridRetrievalError(
                "Dense ranking does not cover all chunks."
            )

        if set(
            lexical_by_id
        ) != expected_ids:
            raise HybridRetrievalError(
                "Lexical ranking does not cover all chunks."
            )

        fused = []

        for chunk in self.chunks:

            dense = dense_by_id[
                chunk.chunk_id
            ]

            lexical = lexical_by_id[
                chunk.chunk_id
            ]

            fusion_score = (
                1.0
                / (
                    self.rrf_k
                    + dense.rank
                )
                + 1.0
                / (
                    self.rrf_k
                    + lexical.rank
                )
            )

            if not math.isfinite(
                fusion_score
            ):
                raise HybridRetrievalError(
                    "Fusion score is non-finite."
                )

            fused.append(
                HybridRetrievalResult(
                    rank=0,
                    fusion_score=float(
                        fusion_score
                    ),
                    dense_rank=dense.rank,
                    dense_score=float(
                        dense.score
                    ),
                    lexical_rank=lexical.rank,
                    lexical_score=float(
                        lexical.lexical_score
                    ),
                    chunk=chunk,
                )
            )

        fused.sort(
            key=lambda result: (
                -result.fusion_score,
                result.chunk.chunk_id,
            )
        )

        ranked = []

        for rank, result in enumerate(
            fused[:top_k],
            start=1,
        ):

            ranked.append(
                HybridRetrievalResult(
                    rank=rank,
                    fusion_score=result.fusion_score,
                    dense_rank=result.dense_rank,
                    dense_score=result.dense_score,
                    lexical_rank=result.lexical_rank,
                    lexical_score=result.lexical_score,
                    chunk=result.chunk,
                )
            )

        return tuple(
            ranked
        )
