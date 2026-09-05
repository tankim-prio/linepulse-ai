"""Retrieval-augmented generation foundations for LinePulse AI."""

from linepulse.rag.chunks import (
    CHUNK_SCHEMA_VERSION,
    RetrievalChunk,
    build_retrieval_chunk,
)
from linepulse.rag.documents import (
    KnowledgeDocument,
    KnowledgeSection,
    RagDataError,
)
from linepulse.rag.loader import (
    DEFAULT_DATA_DIR,
    KnowledgeCorpus,
    build_retrieval_chunks,
    load_documents,
    load_knowledge_corpus,
    load_sections,
    validate_knowledge_integrity,
)


__all__ = [
    "CHUNK_SCHEMA_VERSION",
    "DEFAULT_DATA_DIR",
    "KnowledgeCorpus",
    "KnowledgeDocument",
    "KnowledgeSection",
    "RagDataError",
    "RetrievalChunk",
    "build_retrieval_chunk",
    "build_retrieval_chunks",
    "load_documents",
    "load_knowledge_corpus",
    "load_sections",
    "validate_knowledge_integrity",
]
