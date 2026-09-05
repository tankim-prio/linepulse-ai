"""Validated loading of the LinePulse bilingual RAG knowledge corpus."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from linepulse.rag.chunks import (
    RetrievalChunk,
    build_retrieval_chunk,
)
from linepulse.rag.documents import (
    KnowledgeDocument,
    KnowledgeSection,
    RagDataError,
)


DEFAULT_DATA_DIR = Path(
    "data/linepulse"
)

DOCUMENTS_RELATIVE_PATH = Path(
    "knowledge/documents.csv"
)

SECTIONS_RELATIVE_PATH = Path(
    "knowledge/document_sections.csv"
)


DOCUMENT_COLUMNS = (
    "document_id",
    "title",
    "document_type",
    "version",
    "effective_from",
    "effective_to",
    "language",
    "filename",
    "status",
    "dataset_provenance",
)

SECTION_COLUMNS = (
    "document_id",
    "section_id",
    "section_title",
    "english_text",
    "bangla_text",
    "dataset_provenance",
)


@dataclass(frozen=True)
class KnowledgeCorpus:
    """Validated knowledge documents, sections, and retrieval chunks."""

    documents: tuple[KnowledgeDocument, ...]
    sections: tuple[KnowledgeSection, ...]
    chunks: tuple[RetrievalChunk, ...]


def _read_csv(
    path: Path,
) -> pd.DataFrame:
    if not path.is_file():
        raise RagDataError(
            f"Required RAG data file not found: {path}"
        )

    try:
        return pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
        )

    except Exception as exc:
        raise RagDataError(
            f"Could not read RAG data file: {path}"
        ) from exc


def _require_columns(
    frame: pd.DataFrame,
    required: tuple[str, ...],
    *,
    dataset_name: str,
) -> None:
    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise RagDataError(
            f"{dataset_name} is missing required columns: {missing}"
        )


def _required_text(
    value: object,
    *,
    field: str,
    identity: str,
) -> str:
    text = str(value).strip()

    if not text:
        raise RagDataError(
            f"{identity} has an empty required field: {field}"
        )

    return text


def _optional_text(
    value: object,
) -> str | None:
    text = str(value).strip()

    return (
        text
        if text
        else None
    )


def load_documents(
    data_dir: Path | str = DEFAULT_DATA_DIR,
) -> tuple[KnowledgeDocument, ...]:
    """Load document catalog metadata without changing source content."""

    data_root = Path(
        data_dir
    )

    frame = _read_csv(
        data_root
        / DOCUMENTS_RELATIVE_PATH
    )

    _require_columns(
        frame,
        DOCUMENT_COLUMNS,
        dataset_name="documents",
    )

    if frame["document_id"].duplicated().any():
        duplicates = sorted(
            frame.loc[
                frame[
                    "document_id"
                ].duplicated(
                    keep=False
                ),
                "document_id",
            ].unique()
        )

        raise RagDataError(
            "Duplicate document_id values: "
            f"{duplicates}"
        )

    if frame["filename"].duplicated().any():
        duplicates = sorted(
            frame.loc[
                frame[
                    "filename"
                ].duplicated(
                    keep=False
                ),
                "filename",
            ].unique()
        )

        raise RagDataError(
            "Duplicate document filenames: "
            f"{duplicates}"
        )

    documents: list[
        KnowledgeDocument
    ] = []

    for row in frame.to_dict(
        orient="records"
    ):
        document_id = _required_text(
            row["document_id"],
            field="document_id",
            identity="documents row",
        )

        filename = _required_text(
            row["filename"],
            field="filename",
            identity=document_id,
        )

        filename_path = Path(
            filename
        )

        if (
            filename_path.name != filename
            or filename_path.is_absolute()
        ):
            raise RagDataError(
                f"{document_id} has an unsafe filename: {filename}"
            )

        source_path = (
            Path("knowledge")
            / filename
        ).as_posix()

        documents.append(
            KnowledgeDocument(
                document_id=document_id,
                title=_required_text(
                    row["title"],
                    field="title",
                    identity=document_id,
                ),
                document_type=_required_text(
                    row["document_type"],
                    field="document_type",
                    identity=document_id,
                ),
                version=_required_text(
                    row["version"],
                    field="version",
                    identity=document_id,
                ),
                effective_from=_required_text(
                    row["effective_from"],
                    field="effective_from",
                    identity=document_id,
                ),
                effective_to=_optional_text(
                    row["effective_to"]
                ),
                language=_required_text(
                    row["language"],
                    field="language",
                    identity=document_id,
                ),
                filename=filename,
                source_path=source_path,
                status=_required_text(
                    row["status"],
                    field="status",
                    identity=document_id,
                ),
                dataset_provenance=_required_text(
                    row["dataset_provenance"],
                    field="dataset_provenance",
                    identity=document_id,
                ),
            )
        )

    return tuple(
        sorted(
            documents,
            key=lambda item: item.document_id,
        )
    )


def load_sections(
    data_dir: Path | str = DEFAULT_DATA_DIR,
) -> tuple[KnowledgeSection, ...]:
    """Load bilingual section data."""

    data_root = Path(
        data_dir
    )

    frame = _read_csv(
        data_root
        / SECTIONS_RELATIVE_PATH
    )

    _require_columns(
        frame,
        SECTION_COLUMNS,
        dataset_name="document_sections",
    )

    if frame["section_id"].duplicated().any():
        duplicates = sorted(
            frame.loc[
                frame[
                    "section_id"
                ].duplicated(
                    keep=False
                ),
                "section_id",
            ].unique()
        )

        raise RagDataError(
            "Duplicate section_id values: "
            f"{duplicates}"
        )

    sections: list[
        KnowledgeSection
    ] = []

    for row in frame.to_dict(
        orient="records"
    ):
        section_id = _required_text(
            row["section_id"],
            field="section_id",
            identity="document_sections row",
        )

        sections.append(
            KnowledgeSection(
                document_id=_required_text(
                    row["document_id"],
                    field="document_id",
                    identity=section_id,
                ),
                section_id=section_id,
                section_title=_required_text(
                    row["section_title"],
                    field="section_title",
                    identity=section_id,
                ),
                english_text=_required_text(
                    row["english_text"],
                    field="english_text",
                    identity=section_id,
                ),
                bangla_text=_required_text(
                    row["bangla_text"],
                    field="bangla_text",
                    identity=section_id,
                ),
                dataset_provenance=_required_text(
                    row["dataset_provenance"],
                    field="dataset_provenance",
                    identity=section_id,
                ),
            )
        )

    return tuple(
        sorted(
            sections,
            key=lambda item: (
                item.document_id,
                item.section_id,
            ),
        )
    )


def validate_knowledge_integrity(
    documents: tuple[KnowledgeDocument, ...],
    sections: tuple[KnowledgeSection, ...],
    *,
    data_dir: Path | str = DEFAULT_DATA_DIR,
) -> None:
    """Validate catalog, section, and Markdown-source integrity."""

    data_root = Path(
        data_dir
    )

    document_ids = [
        document.document_id
        for document in documents
    ]

    if len(document_ids) != len(
        set(document_ids)
    ):
        raise RagDataError(
            "Knowledge documents contain duplicate document_id values."
        )

    section_ids = [
        section.section_id
        for section in sections
    ]

    if len(section_ids) != len(
        set(section_ids)
    ):
        raise RagDataError(
            "Knowledge sections contain duplicate section_id values."
        )

    document_map = {
        document.document_id:
            document
        for document in documents
    }

    unknown_references = sorted(
        {
            section.document_id
            for section in sections
            if section.document_id
            not in document_map
        }
    )

    if unknown_references:
        raise RagDataError(
            "Knowledge sections reference unknown document_id values: "
            f"{unknown_references}"
        )

    documents_with_sections = {
        section.document_id
        for section in sections
    }

    without_sections = sorted(
        set(document_map)
        - documents_with_sections
    )

    if without_sections:
        raise RagDataError(
            "Knowledge documents have no sections: "
            f"{without_sections}"
        )

    source_text_by_document: dict[
        str,
        str,
    ] = {}

    for document in documents:
        source_file = (
            data_root
            / document.source_path
        )

        if not source_file.is_file():
            raise RagDataError(
                "Missing knowledge source file for "
                f"{document.document_id}: {source_file}"
            )

        try:
            source_text = source_file.read_text(
                encoding="utf-8"
            )

        except Exception as exc:
            raise RagDataError(
                "Could not read knowledge source file for "
                f"{document.document_id}: {source_file}"
            ) from exc

        source_text_by_document[
            document.document_id
        ] = source_text

    for section in sections:
        source_text = (
            source_text_by_document[
                section.document_id
            ]
        )

        if section.english_text not in source_text:
            raise RagDataError(
                f"{section.section_id} English text "
                "does not match its Markdown source."
            )

        if section.bangla_text not in source_text:
            raise RagDataError(
                f"{section.section_id} Bangla text "
                "does not match its Markdown source."
            )


def build_retrieval_chunks(
    documents: tuple[KnowledgeDocument, ...],
    sections: tuple[KnowledgeSection, ...],
) -> tuple[RetrievalChunk, ...]:
    """Build one deterministic retrieval chunk per source section."""

    document_map = {
        document.document_id:
            document
        for document in documents
    }

    chunks: list[
        RetrievalChunk
    ] = []

    for section in sections:
        document = document_map.get(
            section.document_id
        )

        if document is None:
            raise RagDataError(
                f"{section.section_id} references "
                f"unknown document_id {section.document_id}."
            )

        chunks.append(
            build_retrieval_chunk(
                document,
                section,
            )
        )

    chunks.sort(
        key=lambda item: (
            item.document_id,
            item.section_id,
        )
    )

    chunk_ids = [
        chunk.chunk_id
        for chunk in chunks
    ]

    if len(chunk_ids) != len(
        set(chunk_ids)
    ):
        raise RagDataError(
            "Deterministic chunk IDs are not unique."
        )

    return tuple(
        chunks
    )


def load_knowledge_corpus(
    data_dir: Path | str = DEFAULT_DATA_DIR,
) -> KnowledgeCorpus:
    """Load and validate the complete retrieval-ready corpus."""

    documents = load_documents(
        data_dir
    )

    sections = load_sections(
        data_dir
    )

    validate_knowledge_integrity(
        documents,
        sections,
        data_dir=data_dir,
    )

    chunks = build_retrieval_chunks(
        documents,
        sections,
    )

    return KnowledgeCorpus(
        documents=documents,
        sections=sections,
        chunks=chunks,
    )
