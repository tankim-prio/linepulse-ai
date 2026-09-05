"""Deterministic section-level chunks for LinePulse RAG."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from linepulse.rag.documents import (
    KnowledgeDocument,
    KnowledgeSection,
)


CHUNK_SCHEMA_VERSION = "bilingual-section-v1"


@dataclass(frozen=True)
class RetrievalChunk:
    """Retrieval-ready bilingual section with traceable metadata."""

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
    english_text: str
    bangla_text: str
    text: str
    content_sha256: str


def _render_chunk_text(
    document: KnowledgeDocument,
    section: KnowledgeSection,
) -> str:
    """Render deterministic bilingual retrieval text."""

    return (
        f"Document: {document.title}\n"
        f"Section: {section.section_title}\n"
        f"English: {section.english_text}\n"
        f"বাংলা: {section.bangla_text}"
    )


def build_retrieval_chunk(
    document: KnowledgeDocument,
    section: KnowledgeSection,
) -> RetrievalChunk:
    """Build one stable chunk from one pre-segmented bilingual section."""

    if section.document_id != document.document_id:
        raise ValueError(
            "Section document_id does not match document metadata."
        )

    text = _render_chunk_text(
        document,
        section,
    )

    source_id = (
        f"{document.document_id}:"
        f"{section.section_id}:"
        f"{document.version}"
    )

    identity_payload = {
        "chunk_schema_version": CHUNK_SCHEMA_VERSION,
        "document_id": document.document_id,
        "section_id": section.section_id,
        "document_title": document.title,
        "section_title": section.section_title,
        "document_type": document.document_type,
        "document_version": document.version,
        "document_language": document.language,
        "filename": document.filename,
        "status": document.status,
        "english_text": section.english_text,
        "bangla_text": section.bangla_text,
    }

    canonical = json.dumps(
        identity_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    content_sha256 = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()

    chunk_id = (
        "CHK-"
        + content_sha256[:24]
    )

    return RetrievalChunk(
        chunk_id=chunk_id,
        chunk_schema_version=CHUNK_SCHEMA_VERSION,
        source_id=source_id,
        document_id=document.document_id,
        section_id=section.section_id,
        document_title=document.title,
        section_title=section.section_title,
        document_type=document.document_type,
        document_version=document.version,
        document_language=document.language,
        filename=document.filename,
        source_path=document.source_path,
        status=document.status,
        document_provenance=document.dataset_provenance,
        section_provenance=section.dataset_provenance,
        english_text=section.english_text,
        bangla_text=section.bangla_text,
        text=text,
        content_sha256=content_sha256,
    )
