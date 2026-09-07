"""Retrieval-augmented generation foundations for LinePulse AI."""

from linepulse.rag.chunks import (
    CHUNK_SCHEMA_VERSION,
    RetrievalChunk,
    build_retrieval_chunk,
)
from linepulse.rag.embeddings import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_REVISION,
    EmbeddingBatch,
    EmbeddingError,
    MultilingualE5Embedder,
)
from linepulse.rag.documents import (
    KnowledgeDocument,
    KnowledgeSection,
    RagDataError,
)
from linepulse.rag.retrieval import (
    DEFAULT_TOP_K,
    InMemoryRetriever,
    RetrievalError,
    RetrievalResult,
)
from linepulse.rag.evidence import (
    EvidenceBundle,
    EvidenceError,
    EvidenceRecord,
    EvidenceService,
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
    "EvidenceService",
    "EvidenceRecord",
    "EvidenceError",
    "EvidenceBundle",
    "RetrievalResult",
    "RetrievalError",
    "InMemoryRetriever",
    "DEFAULT_TOP_K",
    "MultilingualE5Embedder",
    "EmbeddingError",
    "EmbeddingBatch",
    "EMBEDDING_MODEL_REVISION",
    "EMBEDDING_MODEL_NAME",
    "EMBEDDING_DIMENSION",
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
