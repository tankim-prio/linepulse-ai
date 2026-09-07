from __future__ import annotations

from dataclasses import fields

import pytest

from linepulse.rag import (
    EvidenceBundle,
    EvidenceError,
    EvidenceRecord,
    EvidenceService,
    RetrievalResult,
    load_knowledge_corpus,
)


class FakeRetriever:
    """Deterministic retriever substitute for evidence tests."""

    def __init__(
        self,
        results,
        *,
        honor_top_k: bool = True,
    ) -> None:
        self.results = tuple(
            results
        )

        self.honor_top_k = (
            honor_top_k
        )

        self.calls = []

    def search(
        self,
        query: str,
        *,
        top_k: int,
    ):
        self.calls.append(
            (
                query,
                top_k,
            )
        )

        if self.honor_top_k:
            return self.results[
                :top_k
            ]

        return self.results


def _result(
    *,
    rank: int,
    score: float,
    chunk_index: int,
) -> RetrievalResult:

    corpus = load_knowledge_corpus()

    return RetrievalResult(
        rank=rank,
        score=score,
        chunk=corpus.chunks[
            chunk_index
        ],
    )


def _three_results():
    return (
        _result(
            rank=1,
            score=0.90,
            chunk_index=0,
        ),
        _result(
            rank=2,
            score=0.80,
            chunk_index=1,
        ),
        _result(
            rank=3,
            score=0.70,
            chunk_index=2,
        ),
    )


def test_evidence_public_contract():
    record_fields = [
        field.name
        for field in fields(
            EvidenceRecord
        )
    ]

    bundle_fields = [
        field.name
        for field in fields(
            EvidenceBundle
        )
    ]

    assert record_fields == [
        "rank",
        "score",
        "chunk_id",
        "chunk_schema_version",
        "source_id",
        "document_id",
        "section_id",
        "document_title",
        "section_title",
        "document_type",
        "document_version",
        "document_language",
        "filename",
        "source_path",
        "status",
        "document_provenance",
        "section_provenance",
        "content_sha256",
        "english_text",
        "bangla_text",
        "text",
    ]

    assert bundle_fields == [
        "query",
        "top_k",
        "evidence",
    ]


def test_collect_normalizes_query_and_forwards_top_k():
    retriever = FakeRetriever(
        _three_results()
    )

    service = EvidenceService(
        retriever
    )

    bundle = service.collect(
        "   production status question   ",
        top_k=3,
    )

    assert (
        bundle.query
        == "production status question"
    )

    assert bundle.top_k == 3

    assert retriever.calls == [
        (
            "production status question",
            3,
        )
    ]


def test_collect_preserves_source_metadata_and_text():
    results = _three_results()

    retriever = FakeRetriever(
        results
    )

    service = EvidenceService(
        retriever
    )

    bundle = service.collect(
        "question"
    )

    assert len(
        bundle.evidence
    ) == 3

    source = results[0].chunk
    evidence = bundle.evidence[0]

    assert evidence.rank == 1
    assert evidence.score == 0.90

    assert (
        evidence.chunk_id
        == source.chunk_id
    )

    assert (
        evidence.chunk_schema_version
        == source.chunk_schema_version
    )

    assert (
        evidence.source_id
        == source.source_id
    )

    assert (
        evidence.document_id
        == source.document_id
    )

    assert (
        evidence.section_id
        == source.section_id
    )

    assert (
        evidence.document_title
        == source.document_title
    )

    assert (
        evidence.section_title
        == source.section_title
    )

    assert (
        evidence.document_type
        == source.document_type
    )

    assert (
        evidence.document_version
        == source.document_version
    )

    assert (
        evidence.document_language
        == source.document_language
    )

    assert (
        evidence.filename
        == source.filename
    )

    assert (
        evidence.source_path
        == source.source_path
    )

    assert (
        evidence.status
        == source.status
    )

    assert (
        evidence.document_provenance
        == source.document_provenance
    )

    assert (
        evidence.section_provenance
        == source.section_provenance
    )

    assert (
        evidence.content_sha256
        == source.content_sha256
    )

    assert (
        evidence.english_text
        == source.english_text
    )

    assert (
        evidence.bangla_text
        == source.bangla_text
    )

    assert (
        evidence.text
        == source.text
    )


def test_collect_rejects_invalid_query():
    service = EvidenceService(
        FakeRetriever(
            _three_results()
        )
    )

    with pytest.raises(
        TypeError,
        match="query must be str",
    ):
        service.collect(
            123
        )

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        service.collect(
            "   "
        )


def test_collect_rejects_wrong_result_count():
    retriever = FakeRetriever(
        (
            _result(
                rank=1,
                score=0.90,
                chunk_index=0,
            ),
            _result(
                rank=2,
                score=0.80,
                chunk_index=1,
            ),
        ),
        honor_top_k=False,
    )

    service = EvidenceService(
        retriever
    )

    with pytest.raises(
        EvidenceError,
        match="unexpected number",
    ):
        service.collect(
            "question",
            top_k=3,
        )


def test_collect_rejects_noncontiguous_ranks():
    retriever = FakeRetriever(
        (
            _result(
                rank=1,
                score=0.90,
                chunk_index=0,
            ),
            _result(
                rank=3,
                score=0.80,
                chunk_index=1,
            ),
            _result(
                rank=4,
                score=0.70,
                chunk_index=2,
            ),
        )
    )

    service = EvidenceService(
        retriever
    )

    with pytest.raises(
        EvidenceError,
        match="not contiguous",
    ):
        service.collect(
            "question"
        )


def test_collect_rejects_score_order_violation():
    retriever = FakeRetriever(
        (
            _result(
                rank=1,
                score=0.90,
                chunk_index=0,
            ),
            _result(
                rank=2,
                score=0.70,
                chunk_index=1,
            ),
            _result(
                rank=3,
                score=0.80,
                chunk_index=2,
            ),
        )
    )

    service = EvidenceService(
        retriever
    )

    with pytest.raises(
        EvidenceError,
        match="ordered descending",
    ):
        service.collect(
            "question"
        )


def test_custom_top_k_returns_matching_bundle():
    retriever = FakeRetriever(
        _three_results()
    )

    service = EvidenceService(
        retriever
    )

    bundle = service.collect(
        "question",
        top_k=2,
    )

    assert bundle.top_k == 2

    assert len(
        bundle.evidence
    ) == 2

    assert [
        item.rank
        for item in bundle.evidence
    ] == [
        1,
        2,
    ]

    assert retriever.calls == [
        (
            "question",
            2,
        )
    ]
