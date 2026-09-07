from __future__ import annotations

import numpy as np
import pytest

from linepulse.rag import (
    DEFAULT_TOP_K,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_REVISION,
    EmbeddingBatch,
    InMemoryRetriever,
    RetrievalError,
    RetrievalResult,
    load_knowledge_corpus,
)


def _batch(
    vectors: np.ndarray,
) -> EmbeddingBatch:
    return EmbeddingBatch(
        model_name=EMBEDDING_MODEL_NAME,
        model_revision=EMBEDDING_MODEL_REVISION,
        dimension=EMBEDDING_DIMENSION,
        vectors=np.asarray(
            vectors,
            dtype=np.float32,
        ),
    )


class FakeEmbedder:
    def __init__(
        self,
        passage_vectors: np.ndarray,
        query_vectors: np.ndarray,
    ) -> None:
        self.passage_vectors = np.asarray(
            passage_vectors,
            dtype=np.float32,
        )

        self.query_vectors = np.asarray(
            query_vectors,
            dtype=np.float32,
        )

        self.passage_calls = 0
        self.query_calls = 0

    def embed_chunks(
        self,
        chunks,
    ) -> EmbeddingBatch:
        self.passage_calls += 1

        return _batch(
            self.passage_vectors
        )

    def embed_queries(
        self,
        queries,
    ) -> EmbeddingBatch:
        self.query_calls += 1

        return _batch(
            self.query_vectors
        )


def _zero_vectors(
    count: int,
) -> np.ndarray:
    return np.zeros(
        (
            count,
            EMBEDDING_DIMENSION,
        ),
        dtype=np.float32,
    )


def _basic_fake_embedder(
    chunk_count: int,
) -> FakeEmbedder:
    passages = _zero_vectors(
        chunk_count
    )

    passages[:, 1] = 1.0
    passages[0, :] = 0.0
    passages[0, 0] = 1.0

    query = _zero_vectors(
        1
    )

    query[0, 0] = 1.0

    return FakeEmbedder(
        passages,
        query,
    )


def test_retrieval_public_contract():
    assert DEFAULT_TOP_K == 3

    field_names = [
        field
        for field in (
            RetrievalResult.__dataclass_fields__
        )
    ]

    assert field_names == [
        "rank",
        "score",
        "chunk",
    ]


def test_retriever_rejects_empty_and_duplicate_chunks():
    with pytest.raises(
        ValueError,
        match="At least one",
    ):
        InMemoryRetriever(
            []
        )

    corpus = load_knowledge_corpus()

    duplicate = (
        corpus.chunks[0],
        corpus.chunks[0],
    )

    with pytest.raises(
        ValueError,
        match="unique",
    ):
        InMemoryRetriever(
            duplicate
        )


def test_retriever_validates_query():
    corpus = load_knowledge_corpus()

    embedder = _basic_fake_embedder(
        len(
            corpus.chunks
        )
    )

    retriever = InMemoryRetriever(
        corpus.chunks,
        embedder=embedder,
    )

    with pytest.raises(
        TypeError,
        match="query must be str",
    ):
        retriever.search(
            123
        )

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        retriever.search(
            "   "
        )


def test_retriever_validates_top_k():
    corpus = load_knowledge_corpus()

    embedder = _basic_fake_embedder(
        len(
            corpus.chunks
        )
    )

    retriever = InMemoryRetriever(
        corpus.chunks,
        embedder=embedder,
    )

    for invalid in [
        True,
        1.5,
        "3",
    ]:
        with pytest.raises(
            TypeError,
            match="top_k must be int",
        ):
            retriever.search(
                "query",
                top_k=invalid,
            )

    with pytest.raises(
        ValueError,
        match="at least 1",
    ):
        retriever.search(
            "query",
            top_k=0,
        )

    with pytest.raises(
        ValueError,
        match="cannot exceed",
    ):
        retriever.search(
            "query",
            top_k=(
                len(
                    corpus.chunks
                )
                + 1
            ),
        )


def test_default_top_three_preserves_evidence_metadata():
    corpus = load_knowledge_corpus()

    count = len(
        corpus.chunks
    )

    passages = _zero_vectors(
        count
    )

    # Score 1.0
    passages[0, 0] = 1.0

    # Score 0.8
    passages[1, 0] = 0.8
    passages[1, 1] = 0.6

    # Score 0.6
    passages[2, 0] = 0.6
    passages[2, 1] = 0.8

    # Remaining chunks have score 0.
    for index in range(
        3,
        count,
    ):
        passages[index, 2] = 1.0

    query = _zero_vectors(
        1
    )

    query[0, 0] = 1.0

    embedder = FakeEmbedder(
        passages,
        query,
    )

    retriever = InMemoryRetriever(
        corpus.chunks,
        embedder=embedder,
    )

    results = retriever.search(
        "test evidence query"
    )

    assert len(results) == 3

    assert [
        result.rank
        for result in results
    ] == [
        1,
        2,
        3,
    ]

    assert [
        result.chunk.chunk_id
        for result in results
    ] == [
        corpus.chunks[0].chunk_id,
        corpus.chunks[1].chunk_id,
        corpus.chunks[2].chunk_id,
    ]

    assert np.allclose(
        [
            result.score
            for result in results
        ],
        [
            1.0,
            0.8,
            0.6,
        ],
        atol=1e-6,
    )

    first = results[0]

    assert (
        first.chunk.document_id
        == corpus.chunks[0].document_id
    )

    assert (
        first.chunk.section_id
        == corpus.chunks[0].section_id
    )

    assert (
        first.chunk.source_id
        == corpus.chunks[0].source_id
    )

    assert (
        first.chunk.source_path
        == corpus.chunks[0].source_path
    )

    assert (
        first.chunk.content_sha256
        == corpus.chunks[0].content_sha256
    )


def test_top_k_one_returns_only_best_result():
    corpus = load_knowledge_corpus()

    embedder = _basic_fake_embedder(
        len(
            corpus.chunks
        )
    )

    retriever = InMemoryRetriever(
        corpus.chunks,
        embedder=embedder,
    )

    results = retriever.search(
        "query",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].rank == 1

    assert (
        results[0].chunk.chunk_id
        == corpus.chunks[0].chunk_id
    )


def test_equal_scores_use_chunk_id_tie_break():
    corpus = load_knowledge_corpus()

    count = len(
        corpus.chunks
    )

    passages = _zero_vectors(
        count
    )

    passages[:, 1] = 1.0

    passages[0, :] = 0.0
    passages[1, :] = 0.0

    passages[0, 0] = 1.0
    passages[1, 0] = 1.0

    query = _zero_vectors(
        1
    )

    query[0, 0] = 1.0

    embedder = FakeEmbedder(
        passages,
        query,
    )

    retriever = InMemoryRetriever(
        corpus.chunks,
        embedder=embedder,
    )

    results = retriever.search(
        "tie query",
        top_k=2,
    )

    expected_ids = sorted(
        [
            corpus.chunks[0].chunk_id,
            corpus.chunks[1].chunk_id,
        ]
    )

    actual_ids = [
        result.chunk.chunk_id
        for result in results
    ]

    assert actual_ids == expected_ids

    assert np.allclose(
        [
            result.score
            for result in results
        ],
        [
            1.0,
            1.0,
        ],
        atol=1e-6,
    )


def test_passage_vectors_are_cached_across_searches():
    corpus = load_knowledge_corpus()

    embedder = _basic_fake_embedder(
        len(
            corpus.chunks
        )
    )

    retriever = InMemoryRetriever(
        corpus.chunks,
        embedder=embedder,
    )

    retriever.search(
        "first query"
    )

    retriever.search(
        "second query"
    )

    assert embedder.passage_calls == 1
    assert embedder.query_calls == 2

    first = retriever.passage_vectors
    second = retriever.passage_vectors

    assert first is second


def test_invalid_passage_embedding_shape_is_rejected():
    corpus = load_knowledge_corpus()

    bad_passages = np.zeros(
        (
            len(
                corpus.chunks
            ),
            EMBEDDING_DIMENSION - 1,
        ),
        dtype=np.float32,
    )

    query = _zero_vectors(
        1
    )

    query[0, 0] = 1.0

    embedder = FakeEmbedder(
        bad_passages,
        query,
    )

    retriever = InMemoryRetriever(
        corpus.chunks,
        embedder=embedder,
    )

    with pytest.raises(
        RetrievalError,
        match="passage embedding shape",
    ):
        retriever.search(
            "query"
        )


def test_invalid_query_embedding_shape_is_rejected():
    corpus = load_knowledge_corpus()

    passages = _zero_vectors(
        len(
            corpus.chunks
        )
    )

    passages[:, 0] = 1.0

    bad_query = np.zeros(
        (
            2,
            EMBEDDING_DIMENSION,
        ),
        dtype=np.float32,
    )

    embedder = FakeEmbedder(
        passages,
        bad_query,
    )

    retriever = InMemoryRetriever(
        corpus.chunks,
        embedder=embedder,
    )

    with pytest.raises(
        RetrievalError,
        match="query embedding shape",
    ):
        retriever.search(
            "query"
        )
