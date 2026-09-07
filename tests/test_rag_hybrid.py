from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from linepulse.rag import (
    DEFAULT_RRF_K,
    EMBEDDING_DIMENSION,
    HybridRetriever,
    LexicalRetriever,
    load_knowledge_corpus,
)


class FakeEmbedder:
    """Small deterministic embedder for unit tests only."""

    def __init__(self) -> None:
        self.chunk_calls = 0
        self.query_calls = 0

    def embed_chunks(
        self,
        chunks,
    ):
        self.chunk_calls += 1

        vectors = np.zeros(
            (
                len(chunks),
                EMBEDDING_DIMENSION,
            ),
            dtype=np.float32,
        )

        for index in range(
            len(chunks)
        ):
            vectors[
                index,
                index,
            ] = 1.0

        return SimpleNamespace(
            vectors=vectors
        )

    def embed_queries(
        self,
        queries,
    ):
        self.query_calls += 1

        vectors = np.zeros(
            (
                len(queries),
                EMBEDDING_DIMENSION,
            ),
            dtype=np.float32,
        )

        vectors[
            :,
            0,
        ] = 1.0

        return SimpleNamespace(
            vectors=vectors
        )


@pytest.fixture
def chunks():
    corpus = load_knowledge_corpus()

    assert len(
        corpus.chunks
    ) == 16

    return corpus.chunks


def test_lexical_retriever_builds_sparse_index(
    chunks,
) -> None:

    retriever = LexicalRetriever(
        chunks
    )

    assert retriever.chunk_count == 16
    assert retriever.vocabulary_size > 0
    assert retriever.matrix_nnz > 0

    assert retriever.matrix.shape[0] == 16


def test_lexical_retrieval_is_deterministic(
    chunks,
) -> None:

    retriever = LexicalRetriever(
        chunks
    )

    query = (
        "hourly production target actual "
        "output downtime defects"
    )

    first = retriever.search(
        query,
        top_k=5,
    )

    second = retriever.search(
        query,
        top_k=5,
    )

    assert first == second

    assert [
        result.rank
        for result in first
    ] == [
        1,
        2,
        3,
        4,
        5,
    ]

    assert len(
        {
            result.chunk_id
            for result in first
        }
    ) == 5

    assert all(
        np.isfinite(
            result.lexical_score
        )
        for result in first
    )


def test_lexical_zero_match_is_still_deterministic(
    chunks,
) -> None:

    retriever = LexicalRetriever(
        chunks
    )

    results = retriever.search(
        "?????",
        top_k=3,
    )

    assert all(
        result.lexical_score == 0.0
        for result in results
    )

    expected_ids = sorted(
        chunk.chunk_id
        for chunk in chunks
    )[:3]

    assert [
        result.chunk_id
        for result in results
    ] == expected_ids


def test_lexical_input_validation(
    chunks,
) -> None:

    retriever = LexicalRetriever(
        chunks
    )

    with pytest.raises(
        TypeError
    ):
        retriever.search(
            123,
        )

    with pytest.raises(
        ValueError
    ):
        retriever.search(
            "   ",
        )

    with pytest.raises(
        TypeError
    ):
        retriever.search(
            "query",
            top_k=True,
        )

    with pytest.raises(
        ValueError
    ):
        retriever.search(
            "query",
            top_k=0,
        )

    with pytest.raises(
        ValueError
    ):
        retriever.search(
            "query",
            top_k=17,
        )


def test_hybrid_result_contract_and_rrf_formula(
    chunks,
) -> None:

    fake = FakeEmbedder()

    retriever = HybridRetriever(
        chunks,
        embedder=fake,
    )

    results = retriever.search(
        "hourly production output downtime",
        top_k=5,
    )

    assert len(results) == 5

    assert [
        result.rank
        for result in results
    ] == [
        1,
        2,
        3,
        4,
        5,
    ]

    for result in results:

        expected = (
            1.0
            / (
                DEFAULT_RRF_K
                + result.dense_rank
            )
            + 1.0
            / (
                DEFAULT_RRF_K
                + result.lexical_rank
            )
        )

        assert (
            result.fusion_score
            == pytest.approx(
                expected,
                abs=1e-15,
            )
        )

        assert np.isfinite(
            result.dense_score
        )

        assert np.isfinite(
            result.lexical_score
        )

        assert np.isfinite(
            result.fusion_score
        )


def test_hybrid_retrieval_is_deterministic(
    chunks,
) -> None:

    fake = FakeEmbedder()

    retriever = HybridRetriever(
        chunks,
        embedder=fake,
    )

    query = (
        "quality defects corrective action"
    )

    first = retriever.search(
        query,
        top_k=8,
    )

    second = retriever.search(
        query,
        top_k=8,
    )

    assert first == second


def test_dense_passage_embeddings_are_cached(
    chunks,
) -> None:

    fake = FakeEmbedder()

    retriever = HybridRetriever(
        chunks,
        embedder=fake,
    )

    retriever.search(
        "first query",
        top_k=3,
    )

    retriever.search(
        "second query",
        top_k=3,
    )

    assert fake.chunk_calls == 1
    assert fake.query_calls == 2


def test_hybrid_input_validation(
    chunks,
) -> None:

    fake = FakeEmbedder()

    with pytest.raises(
        TypeError
    ):
        HybridRetriever(
            chunks,
            embedder=fake,
            rrf_k=True,
        )

    with pytest.raises(
        ValueError
    ):
        HybridRetriever(
            chunks,
            embedder=fake,
            rrf_k=0,
        )

    retriever = HybridRetriever(
        chunks,
        embedder=fake,
    )

    with pytest.raises(
        ValueError
    ):
        retriever.search(
            "",
        )

    with pytest.raises(
        ValueError
    ):
        retriever.search(
            "query",
            top_k=17,
        )


def test_hybrid_score_cannot_masquerade_as_v1_cosine_score(
    chunks,
) -> None:
    """Protect the frozen v1 answerability-score contract."""

    fake = FakeEmbedder()

    retriever = HybridRetriever(
        chunks,
        embedder=fake,
    )

    result = retriever.search(
        "production reporting",
        top_k=1,
    )[0]

    assert not hasattr(
        result,
        "score",
    )

    assert hasattr(
        result,
        "fusion_score",
    )
