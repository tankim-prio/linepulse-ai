from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from linepulse.rag import (
    CHUNK_SCHEMA_VERSION,
    RagDataError,
    load_knowledge_corpus,
    validate_knowledge_integrity,
)


DATA_DIR = Path(
    "data/linepulse"
)


def test_loads_verified_bilingual_knowledge_corpus():
    corpus = load_knowledge_corpus(
        DATA_DIR
    )

    assert len(
        corpus.documents
    ) == 8

    assert len(
        corpus.sections
    ) == 16

    assert len(
        corpus.chunks
    ) == 16

    assert {
        document.document_id
        for document in corpus.documents
    } == {
        f"DOC-{number:03d}"
        for number in range(
            1,
            9,
        )
    }

    assert all(
        document.status == "active"
        for document
        in corpus.documents
    )

    assert all(
        document.language
        == "bilingual_en_bn"
        for document
        in corpus.documents
    )


def test_chunk_ids_are_unique_and_deterministic():
    first = load_knowledge_corpus(
        DATA_DIR
    )

    second = load_knowledge_corpus(
        DATA_DIR
    )

    first_ids = [
        chunk.chunk_id
        for chunk in first.chunks
    ]

    second_ids = [
        chunk.chunk_id
        for chunk in second.chunks
    ]

    assert first_ids == second_ids

    assert len(first_ids) == len(
        set(first_ids)
    )

    assert all(
        chunk.chunk_schema_version
        == CHUNK_SCHEMA_VERSION
        for chunk in first.chunks
    )

    assert all(
        chunk.chunk_id.startswith(
            "CHK-"
        )
        for chunk in first.chunks
    )

    assert all(
        len(
            chunk.content_sha256
        ) == 64
        for chunk in first.chunks
    )

    assert all(
        hashlib.sha256(
            bytes.fromhex(
                chunk.content_sha256
            )
        ).hexdigest()
        for chunk in first.chunks
    )


def test_chunks_preserve_source_text_and_metadata():
    corpus = load_knowledge_corpus(
        DATA_DIR
    )

    document_map = {
        document.document_id:
            document
        for document in corpus.documents
    }

    section_map = {
        section.section_id:
            section
        for section in corpus.sections
    }

    for chunk in corpus.chunks:
        document = document_map[
            chunk.document_id
        ]

        section = section_map[
            chunk.section_id
        ]

        assert (
            chunk.document_id
            == section.document_id
        )

        assert (
            chunk.document_title
            == document.title
        )

        assert (
            chunk.document_version
            == document.version
        )

        assert (
            chunk.filename
            == document.filename
        )

        assert (
            chunk.english_text
            == section.english_text
        )

        assert (
            chunk.bangla_text
            == section.bangla_text
        )

        assert (
            section.english_text
            in chunk.text
        )

        assert (
            section.bangla_text
            in chunk.text
        )

        assert (
            DATA_DIR
            / chunk.source_path
        ).is_file()


def test_one_chunk_maps_to_each_presegmented_section():
    corpus = load_knowledge_corpus(
        DATA_DIR
    )

    chunk_section_ids = {
        chunk.section_id
        for chunk in corpus.chunks
    }

    source_section_ids = {
        section.section_id
        for section in corpus.sections
    }

    assert (
        chunk_section_ids
        == source_section_ids
    )

    assert len(
        chunk_section_ids
    ) == 16


def test_all_gold_references_exist_in_retrieval_corpus():
    corpus = load_knowledge_corpus(
        DATA_DIR
    )

    gold = pd.read_csv(
        DATA_DIR
        / "evaluation"
        / "rag_gold_questions.csv",
        dtype=str,
        keep_default_na=False,
    )

    chunk_document_ids = {
        chunk.document_id
        for chunk in corpus.chunks
    }

    chunk_section_ids = {
        chunk.section_id
        for chunk in corpus.chunks
    }

    referenced_documents = {
        value
        for value
        in gold[
            "relevant_document_id"
        ]
        if value
    }

    referenced_sections = {
        value
        for value
        in gold[
            "relevant_section_id"
        ]
        if value
    }

    assert (
        referenced_documents
        <= chunk_document_ids
    )

    assert (
        referenced_sections
        <= chunk_section_ids
    )

    assert len(
        referenced_documents
    ) == 8

    assert len(
        referenced_sections
    ) == 16


def test_unknown_document_reference_is_rejected():
    corpus = load_knowledge_corpus(
        DATA_DIR
    )

    broken_section = replace(
        corpus.sections[0],
        document_id="DOC-MISSING",
    )

    sections = (
        broken_section,
        *corpus.sections[1:],
    )

    with pytest.raises(
        RagDataError,
        match="unknown document_id",
    ):
        validate_knowledge_integrity(
            corpus.documents,
            sections,
            data_dir=DATA_DIR,
        )
