from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from linepulse.rag import (
    ANSWERABILITY_POLICY_ID,
    ANSWERABILITY_THRESHOLD,
    AnswerabilityDecision,
    AnswerabilityError,
    AnswerabilityGate,
    EvidenceBundle,
    EvidenceRecord,
    load_knowledge_corpus,
)

from linepulse.rag.answerability import (
    ANSWERABILITY_CALIBRATION_NEGATIVE_MAX,
    ANSWERABILITY_CALIBRATION_POSITIVE_MIN,
    ANSWERABILITY_CALIBRATION_ROW_COUNT,
    ANSWERABILITY_CALIBRATION_SHA256,
    ANSWERABILITY_CALIBRATION_SUPPORTED_COUNT,
    ANSWERABILITY_CALIBRATION_INSUFFICIENT_COUNT,
    ANSWERABILITY_CHUNK_SCHEMA_VERSION,
    ANSWERABILITY_MODEL_NAME,
    ANSWERABILITY_MODEL_REVISION,
    INSUFFICIENT_EVIDENCE_STATUS,
    SUPPORTED_STATUS,
)


CALIBRATION_PATH = Path(
    "data/linepulse/evaluation/"
    "rag_answerability_calibration.csv"
)


def _record(
    score: float,
    *,
    rank: int = 1,
) -> EvidenceRecord:

    chunk = load_knowledge_corpus().chunks[0]

    return EvidenceRecord(
        rank=rank,
        score=score,
        chunk_id=chunk.chunk_id,
        chunk_schema_version=chunk.chunk_schema_version,
        source_id=chunk.source_id,
        document_id=chunk.document_id,
        section_id=chunk.section_id,
        document_title=chunk.document_title,
        section_title=chunk.section_title,
        document_type=chunk.document_type,
        document_version=chunk.document_version,
        document_language=chunk.document_language,
        filename=chunk.filename,
        source_path=chunk.source_path,
        status=chunk.status,
        document_provenance=chunk.document_provenance,
        section_provenance=chunk.section_provenance,
        content_sha256=chunk.content_sha256,
        english_text=chunk.english_text,
        bangla_text=chunk.bangla_text,
        text=chunk.text,
    )


def _bundle(
    score: float,
) -> EvidenceBundle:

    return EvidenceBundle(
        query="test question",
        top_k=1,
        evidence=(
            _record(score),
        ),
    )


def test_answerability_public_contract() -> None:

    assert ANSWERABILITY_POLICY_ID == "e5-top1-midpoint-v1"

    assert isinstance(
        ANSWERABILITY_THRESHOLD,
        float,
    )

    gate = AnswerabilityGate()

    assert gate.policy_id == ANSWERABILITY_POLICY_ID

    assert gate.threshold == ANSWERABILITY_THRESHOLD

    decision = gate.decide(
        _bundle(
            ANSWERABILITY_THRESHOLD
        )
    )

    assert isinstance(
        decision,
        AnswerabilityDecision,
    )


def test_calibration_metadata_is_frozen() -> None:

    expected_midpoint = (
        ANSWERABILITY_CALIBRATION_NEGATIVE_MAX
        + ANSWERABILITY_CALIBRATION_POSITIVE_MIN
    ) / 2.0

    assert (
        ANSWERABILITY_CALIBRATION_NEGATIVE_MAX
        < ANSWERABILITY_THRESHOLD
        <= ANSWERABILITY_CALIBRATION_POSITIVE_MIN
    )

    assert ANSWERABILITY_THRESHOLD == pytest.approx(
        expected_midpoint,
        abs=1e-15,
    )

    assert ANSWERABILITY_CALIBRATION_ROW_COUNT == 46

    assert (
        ANSWERABILITY_CALIBRATION_SUPPORTED_COUNT
        == 23
    )

    assert (
        ANSWERABILITY_CALIBRATION_INSUFFICIENT_COUNT
        == 23
    )

    actual_sha256 = hashlib.sha256(
        CALIBRATION_PATH.read_bytes()
    ).hexdigest()

    assert (
        ANSWERABILITY_CALIBRATION_SHA256
        == actual_sha256
    )

    assert (
        ANSWERABILITY_MODEL_NAME
        == "intfloat/multilingual-e5-small"
    )

    assert (
        ANSWERABILITY_MODEL_REVISION
        == "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    )

    assert (
        ANSWERABILITY_CHUNK_SCHEMA_VERSION
        == "bilingual-section-v1"
    )


def test_score_below_threshold_is_insufficient() -> None:

    gate = AnswerabilityGate()

    decision = gate.decide(
        _bundle(
            ANSWERABILITY_THRESHOLD
            - 0.000001
        )
    )

    assert decision.answerable is False

    assert (
        decision.status
        == INSUFFICIENT_EVIDENCE_STATUS
    )

    assert decision.margin_to_threshold < 0


def test_score_equal_to_threshold_is_supported() -> None:

    gate = AnswerabilityGate()

    decision = gate.decide(
        _bundle(
            ANSWERABILITY_THRESHOLD
        )
    )

    assert decision.answerable is True

    assert (
        decision.status
        == SUPPORTED_STATUS
    )

    assert decision.margin_to_threshold == pytest.approx(
        0.0,
        abs=1e-15,
    )


def test_score_above_threshold_is_supported() -> None:

    gate = AnswerabilityGate()

    decision = gate.decide(
        _bundle(
            ANSWERABILITY_THRESHOLD
            + 0.000001
        )
    )

    assert decision.answerable is True

    assert (
        decision.status
        == SUPPORTED_STATUS
    )

    assert decision.margin_to_threshold > 0


def test_gate_rejects_empty_evidence() -> None:

    bundle = EvidenceBundle(
        query="test question",
        top_k=1,
        evidence=(),
    )

    with pytest.raises(
        AnswerabilityError,
        match="Evidence count",
    ):
        AnswerabilityGate().decide(
            bundle
        )


def test_gate_rejects_nonfinite_score() -> None:

    bundle = EvidenceBundle(
        query="test question",
        top_k=1,
        evidence=(
            _record(
                float("nan")
            ),
        ),
    )

    with pytest.raises(
        AnswerabilityError,
        match="finite",
    ):
        AnswerabilityGate().decide(
            bundle
        )


def test_gate_rejects_wrong_chunk_schema() -> None:

    record = replace(
        _record(
            ANSWERABILITY_THRESHOLD
        ),
        chunk_schema_version="future-schema-v2",
    )

    bundle = EvidenceBundle(
        query="test question",
        top_k=1,
        evidence=(
            record,
        ),
    )

    with pytest.raises(
        AnswerabilityError,
        match="chunk schema",
    ):
        AnswerabilityGate().decide(
            bundle
        )
