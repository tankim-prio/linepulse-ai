"""Core document models for the LinePulse RAG knowledge base."""

from __future__ import annotations

from dataclasses import dataclass


class RagDataError(ValueError):
    """Raised when RAG source data violates its expected contract."""


@dataclass(frozen=True)
class KnowledgeDocument:
    """Catalog metadata for one knowledge document."""

    document_id: str
    title: str
    document_type: str
    version: str
    effective_from: str
    effective_to: str | None
    language: str
    filename: str
    source_path: str
    status: str
    dataset_provenance: str


@dataclass(frozen=True)
class KnowledgeSection:
    """One bilingual source section belonging to a knowledge document."""

    document_id: str
    section_id: str
    section_title: str
    english_text: str
    bangla_text: str
    dataset_provenance: str
