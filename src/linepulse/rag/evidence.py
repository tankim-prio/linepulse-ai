"""Structured evidence packaging for LinePulse RAG."""

from __future__ import annotations

from dataclasses import dataclass

from linepulse.rag.retrieval import (
    DEFAULT_TOP_K,
    InMemoryRetriever,
    RetrievalResult,
)


class EvidenceError(RuntimeError):
    """Raised when retrieved evidence violates its contract."""


@dataclass(frozen=True)
class EvidenceRecord:
    """One citations-ready retrieved evidence record."""

    rank: int
    score: float

    chunk_id: str
    chunk_schema_version: str
    source_id: str

    document_id: str
    section_id: str

    document_title: str
    section_title: str

    document_type: str
    document_version: str
    document_language: str

    filename: str
    source_path: str

    status: str

    document_provenance: str
    section_provenance: str

    content_sha256: str

    english_text: str
    bangla_text: str
    text: str


@dataclass(frozen=True)
class EvidenceBundle:
    """Structured retrieved evidence for one user query."""

    query: str
    top_k: int
    evidence: tuple[EvidenceRecord, ...]


class EvidenceService:
    """Convert ranked retrieval results into grounded evidence."""

    def __init__(
        self,
        retriever: InMemoryRetriever,
    ) -> None:

        self.retriever = retriever

    @staticmethod
    def _to_evidence(
        result: RetrievalResult,
    ) -> EvidenceRecord:

        chunk = result.chunk

        return EvidenceRecord(
            rank=result.rank,
            score=float(
                result.score
            ),
            chunk_id=chunk.chunk_id,
            chunk_schema_version=(
                chunk.chunk_schema_version
            ),
            source_id=chunk.source_id,
            document_id=chunk.document_id,
            section_id=chunk.section_id,
            document_title=(
                chunk.document_title
            ),
            section_title=(
                chunk.section_title
            ),
            document_type=(
                chunk.document_type
            ),
            document_version=(
                chunk.document_version
            ),
            document_language=(
                chunk.document_language
            ),
            filename=chunk.filename,
            source_path=chunk.source_path,
            status=chunk.status,
            document_provenance=(
                chunk.document_provenance
            ),
            section_provenance=(
                chunk.section_provenance
            ),
            content_sha256=(
                chunk.content_sha256
            ),
            english_text=(
                chunk.english_text
            ),
            bangla_text=(
                chunk.bangla_text
            ),
            text=chunk.text,
        )

    def collect(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_TOP_K,
    ) -> EvidenceBundle:
        """Retrieve and package evidence without judging answerability."""

        if not isinstance(
            query,
            str,
        ):
            raise TypeError(
                "query must be str."
            )

        normalized_query = (
            query.strip()
        )

        if not normalized_query:
            raise ValueError(
                "query cannot be empty."
            )

        results = self.retriever.search(
            normalized_query,
            top_k=top_k,
        )

        evidence = tuple(
            self._to_evidence(
                result
            )
            for result in results
        )

        if len(evidence) != top_k:
            raise EvidenceError(
                "Retriever returned an unexpected "
                "number of evidence records."
            )

        expected_ranks = tuple(
            range(
                1,
                top_k + 1,
            )
        )

        actual_ranks = tuple(
            item.rank
            for item in evidence
        )

        if actual_ranks != expected_ranks:
            raise EvidenceError(
                "Evidence ranks are not contiguous."
            )

        if any(
            evidence[index].score
            < evidence[index + 1].score
            for index in range(
                len(evidence) - 1
            )
        ):
            raise EvidenceError(
                "Evidence scores are not "
                "ordered descending."
            )

        return EvidenceBundle(
            query=normalized_query,
            top_k=top_k,
            evidence=evidence,
        )
