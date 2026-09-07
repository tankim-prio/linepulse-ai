from __future__ import annotations

import numpy as np
import pytest

from linepulse.rag import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_REVISION,
    MultilingualE5Embedder,
    load_knowledge_corpus,
)


class FakeEmbeddingModel:
    """Small deterministic substitute for unit tests."""

    def __init__(self) -> None:
        self.seen_texts: list[str] = []

    def encode(
        self,
        texts,
        *,
        batch_size,
        normalize_embeddings,
        convert_to_numpy,
        show_progress_bar,
    ):
        self.seen_texts = list(
            texts
        )

        assert batch_size >= 1
        assert normalize_embeddings is True
        assert convert_to_numpy is True
        assert show_progress_bar is False

        vectors = np.zeros(
            (
                len(self.seen_texts),
                EMBEDDING_DIMENSION,
            ),
            dtype=np.float32,
        )

        vectors[:, 0] = 1.0

        return vectors


def _embedder_with_fake_model(
    *,
    batch_size: int = 8,
):
    embedder = MultilingualE5Embedder(
        batch_size=batch_size
    )

    fake_model = FakeEmbeddingModel()

    embedder.__dict__[
        "model"
    ] = fake_model

    return (
        embedder,
        fake_model,
    )


def test_embedding_metadata_is_pinned():
    assert (
        EMBEDDING_MODEL_NAME
        == "intfloat/multilingual-e5-small"
    )

    assert (
        EMBEDDING_MODEL_REVISION
        == "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    )

    assert (
        EMBEDDING_DIMENSION
        == 384
    )


def test_embedder_rejects_non_cpu_device():
    with pytest.raises(
        ValueError,
        match="CPU",
    ):
        MultilingualE5Embedder(
            device="cuda"
        )


def test_embedder_rejects_invalid_batch_size():
    with pytest.raises(
        ValueError,
        match="batch_size",
    ):
        MultilingualE5Embedder(
            batch_size=0
        )


def test_embedder_rejects_empty_input():
    embedder, _ = (
        _embedder_with_fake_model()
    )

    with pytest.raises(
        ValueError,
        match="At least one",
    ):
        embedder.embed_queries(
            []
        )


def test_embedder_rejects_invalid_text_values():
    embedder, _ = (
        _embedder_with_fake_model()
    )

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        embedder.embed_queries(
            [
                "   ",
            ]
        )

    with pytest.raises(
        TypeError,
        match="must be str",
    ):
        embedder.embed_queries(
            [
                123,
            ]
        )


def test_query_embedding_contract_and_prefix():
    embedder, fake_model = (
        _embedder_with_fake_model(
            batch_size=3
        )
    )

    batch = embedder.embed_queries(
        [
            "What should be done?",
            "কী করতে হবে?",
            "What action নিতে হবে?",
        ]
    )

    assert fake_model.seen_texts == [
        "query: What should be done?",
        "query: কী করতে হবে?",
        "query: What action নিতে হবে?",
    ]

    assert (
        batch.model_name
        == EMBEDDING_MODEL_NAME
    )

    assert (
        batch.model_revision
        == EMBEDDING_MODEL_REVISION
    )

    assert (
        batch.dimension
        == EMBEDDING_DIMENSION
    )

    assert (
        batch.vectors.shape
        == (
            3,
            384,
        )
    )

    assert (
        batch.vectors.dtype
        == np.float32
    )

    assert np.isfinite(
        batch.vectors
    ).all()

    assert np.allclose(
        np.linalg.norm(
            batch.vectors,
            axis=1,
        ),
        1.0,
        atol=1e-5,
    )


def test_chunk_embedding_uses_passage_prefix():
    corpus = (
        load_knowledge_corpus()
    )

    assert (
        len(
            corpus.chunks
        )
        == 16
    )

    embedder, fake_model = (
        _embedder_with_fake_model(
            batch_size=8
        )
    )

    batch = embedder.embed_chunks(
        corpus.chunks
    )

    assert (
        len(
            fake_model.seen_texts
        )
        == 16
    )

    assert all(
        text.startswith(
            "passage: "
        )
        for text
        in fake_model.seen_texts
    )

    assert (
        fake_model.seen_texts[0]
        == (
            "passage: "
            + corpus.chunks[0].text
        )
    )

    assert (
        batch.vectors.shape
        == (
            16,
            384,
        )
    )

    assert np.allclose(
        np.linalg.norm(
            batch.vectors,
            axis=1,
        ),
        1.0,
        atol=1e-5,
    )
