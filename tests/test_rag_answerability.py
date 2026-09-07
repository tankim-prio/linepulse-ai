from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from linepulse.rag import (
    ANSWERABILITY_POLICY_FINGERPRINT,
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
    ANSWERABILITY_CALIBRATION_GAP,
    ANSWERABILITY_CALIBRATION_INSUFFICIENT_COUNT,
    ANSWERABILITY_CALIBRATION_NEGATIVE_MAX,
    ANSWERABILITY_CALIBRATION_POSITIVE_MIN,
    ANSWERABILITY_CALIBRATION_ROW_COUNT,
    ANSWERABILITY_CALIBRATION_SHA256,
    ANSWERABILITY_CALIBRATION_SUPPORTED_COUNT,
    ANSWERABILITY_CHUNK_SCHEMA_VERSION,
    ANSWERABILITY_MODEL_NAME,
    ANSWERABILITY_MODEL_REVISION,
    CALIBRATION_BOUNDARY_WARNING,
    INSUFFICIENT_EVIDENCE_STATUS,
    INSUFFICIENT_REASON_CODE,
    SUPPORTED_REASON_CODE,
    SUPPORTED_STATUS,
    _computed_policy_fingerprint,
)


CALIBRATION_PATH = Path(
    "data/linepulse/evaluation/"
    "rag_answerability_calibration.csv"
)


def _record(
    score: float,
    *,
    rank: int = 1,
    chunk_index: int = 0,
) -> EvidenceRecord:

    chunk = load_knowledge_corpus().chunks[
        chunk_index
    ]

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
            _record(
                score
            ),
        ),
    )


def _bundle_two(
    first_score: float = 0.85,
    second_score: float = 0.80,
) -> EvidenceBundle:

    return EvidenceBundle(
        query="test question",
        top_k=2,
        evidence=(
            _record(
                first_score,
                rank=1,
                chunk_index=0,
            ),
            _record(
                second_score,
                rank=2,
                chunk_index=1,
            ),
        ),
    )


def test_public_policy_contract_is_unchanged() -> None:

    assert (
        ANSWERABILITY_POLICY_ID
        == "e5-top1-midpoint-v1"
    )

    assert (
        ANSWERABILITY_THRESHOLD
        == 0.8190949857234955
    )

    gate = AnswerabilityGate()

    assert (
        gate.policy_id
        == ANSWERABILITY_POLICY_ID
    )

    assert (
        gate.threshold
        == ANSWERABILITY_THRESHOLD
    )

    assert (
        gate.policy_fingerprint
        == ANSWERABILITY_POLICY_FINGERPRINT
    )

    decision = gate.decide(
        _bundle(
            ANSWERABILITY_THRESHOLD
        )
    )

    assert isinstance(
        decision,
        AnswerabilityDecision,
    )


def test_policy_fingerprint_is_frozen() -> None:

    assert (
        ANSWERABILITY_POLICY_FINGERPRINT
        == (
            "8b3fb0ebde2632490790a44cc92cc44a"
            "1c1f0c86603cd653c38e0d3190034dea"
        )
    )

    assert (
        _computed_policy_fingerprint()
        == ANSWERABILITY_POLICY_FINGERPRINT
    )

    assert len(
        ANSWERABILITY_POLICY_FINGERPRINT
    ) == 64


def test_calibration_metadata_is_frozen() -> None:

    expected_midpoint = (
        ANSWERABILITY_CALIBRATION_NEGATIVE_MAX
        + ANSWERABILITY_CALIBRATION_POSITIVE_MIN
    ) / 2.0

    assert (
        ANSWERABILITY_THRESHOLD
        == pytest.approx(
            expected_midpoint,
            abs=1e-15,
        )
    )

    assert (
        ANSWERABILITY_CALIBRATION_NEGATIVE_MAX
        < ANSWERABILITY_THRESHOLD
        <= ANSWERABILITY_CALIBRATION_POSITIVE_MIN
    )

    assert (
        ANSWERABILITY_CALIBRATION_GAP
        == pytest.approx(
            0.0015088915824890137,
            abs=1e-15,
        )
    )

    assert (
        ANSWERABILITY_CALIBRATION_ROW_COUNT
        == 46
    )

    assert (
        ANSWERABILITY_CALIBRATION_SUPPORTED_COUNT
        == 23
    )

    assert (
        ANSWERABILITY_CALIBRATION_INSUFFICIENT_COUNT
        == 23
    )

    actual_sha = hashlib.sha256(
        CALIBRATION_PATH.read_bytes()
    ).hexdigest()

    assert (
        actual_sha
        == ANSWERABILITY_CALIBRATION_SHA256
    )

    assert (
        ANSWERABILITY_MODEL_NAME
        == "intfloat/multilingual-e5-small"
    )

    assert (
        ANSWERABILITY_MODEL_REVISION
        == (
            "614241f622f53c4eeff9890bdc4f31cfecc418b3"
        )
    )

    assert (
        ANSWERABILITY_CHUNK_SCHEMA_VERSION
        == "bilingual-section-v1"
    )


def test_below_threshold_is_insufficient() -> None:

    decision = AnswerabilityGate().decide(
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

    assert (
        decision.reason_code
        == INSUFFICIENT_REASON_CODE
    )

    assert (
        decision.margin_to_threshold
        < 0
    )


def test_equal_threshold_is_supported() -> None:

    decision = AnswerabilityGate().decide(
        _bundle(
            ANSWERABILITY_THRESHOLD
        )
    )

    assert decision.answerable is True

    assert (
        decision.status
        == SUPPORTED_STATUS
    )

    assert (
        decision.reason_code
        == SUPPORTED_REASON_CODE
    )

    assert (
        decision.margin_to_threshold
        == pytest.approx(
            0.0,
            abs=1e-15,
        )
    )


def test_above_threshold_is_supported() -> None:

    decision = AnswerabilityGate().decide(
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

    assert (
        decision.reason_code
        == SUPPORTED_REASON_CODE
    )


def test_decision_contains_traceability_fields() -> None:

    bundle = _bundle_two(
        0.85,
        0.80,
    )

    decision = AnswerabilityGate().decide(
        bundle
    )

    assert (
        decision.top1_chunk_id
        == bundle.evidence[0].chunk_id
    )

    assert (
        decision.evidence_count
        == 2
    )

    assert (
        decision.policy_id
        == ANSWERABILITY_POLICY_ID
    )

    assert (
        decision.policy_fingerprint
        == ANSWERABILITY_POLICY_FINGERPRINT
    )


def test_threshold_score_reports_boundary_warning() -> None:

    decision = AnswerabilityGate().decide(
        _bundle(
            ANSWERABILITY_THRESHOLD
        )
    )

    assert (
        decision.calibration_boundary_zone
        is True
    )

    assert (
        decision.warnings
        == (
            CALIBRATION_BOUNDARY_WARNING,
        )
    )


def test_score_outside_gap_has_no_boundary_warning() -> None:

    decision = AnswerabilityGate().decide(
        _bundle(
            ANSWERABILITY_CALIBRATION_POSITIVE_MIN
            + 0.01
        )
    )

    assert (
        decision.calibration_boundary_zone
        is False
    )

    assert decision.warnings == ()


def test_blank_query_is_rejected() -> None:

    bundle = EvidenceBundle(
        query="   ",
        top_k=1,
        evidence=(
            _record(
                0.85
            ),
        ),
    )

    with pytest.raises(
        AnswerabilityError,
        match="must not be blank",
    ):
        AnswerabilityGate().decide(
            bundle
        )


def test_empty_evidence_is_rejected() -> None:

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


def test_evidence_count_must_match_top_k() -> None:

    bundle = EvidenceBundle(
        query="test question",
        top_k=2,
        evidence=(
            _record(
                0.85
            ),
        ),
    )

    with pytest.raises(
        AnswerabilityError,
        match="count does not match",
    ):
        AnswerabilityGate().decide(
            bundle
        )


def test_ranks_must_be_contiguous() -> None:

    bundle = EvidenceBundle(
        query="test question",
        top_k=2,
        evidence=(
            _record(
                0.85,
                rank=1,
                chunk_index=0,
            ),
            _record(
                0.80,
                rank=3,
                chunk_index=1,
            ),
        ),
    )

    with pytest.raises(
        AnswerabilityError,
        match="ranks must be contiguous",
    ):
        AnswerabilityGate().decide(
            bundle
        )


def test_nonfinite_score_anywhere_is_rejected() -> None:

    bundle = EvidenceBundle(
        query="test question",
        top_k=2,
        evidence=(
            _record(
                0.85,
                rank=1,
                chunk_index=0,
            ),
            _record(
                float("nan"),
                rank=2,
                chunk_index=1,
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


def test_out_of_range_cosine_score_is_rejected() -> None:

    with pytest.raises(
        AnswerabilityError,
        match=r"\[-1, 1\]",
    ):
        AnswerabilityGate().decide(
            _bundle(
                1.01
            )
        )


def test_scores_must_be_non_increasing() -> None:

    bundle = _bundle_two(
        0.80,
        0.85,
    )

    with pytest.raises(
        AnswerabilityError,
        match="non-increasing",
    ):
        AnswerabilityGate().decide(
            bundle
        )


def test_duplicate_chunks_are_rejected() -> None:

    bundle = EvidenceBundle(
        query="test question",
        top_k=2,
        evidence=(
            _record(
                0.85,
                rank=1,
                chunk_index=0,
            ),
            _record(
                0.80,
                rank=2,
                chunk_index=0,
            ),
        ),
    )

    with pytest.raises(
        AnswerabilityError,
        match="Duplicate",
    ):
        AnswerabilityGate().decide(
            bundle
        )


def test_wrong_schema_on_non_top1_record_is_rejected() -> None:

    second = replace(
        _record(
            0.80,
            rank=2,
            chunk_index=1,
        ),
        chunk_schema_version="future-schema-v2",
    )

    bundle = EvidenceBundle(
        query="test question",
        top_k=2,
        evidence=(
            _record(
                0.85,
                rank=1,
                chunk_index=0,
            ),
            second,
        ),
    )

    with pytest.raises(
        AnswerabilityError,
        match="chunk schema",
    ):
        AnswerabilityGate().decide(
            bundle
        )
