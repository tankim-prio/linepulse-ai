from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import linepulse.rag as rag

from linepulse.rag.answerability_v2 import (
    DenseTop1AnswerabilityPolicyV2,
    INSUFFICIENT_EVIDENCE_STATUS,
    RAG_V2_AUDIT_SHA256,
    RAG_V2_CORPUS_FINGERPRINT,
    RAG_V2_DENSE_TOP1_THRESHOLD,
    RAG_V2_DEVELOPMENT_SHA256,
    RAG_V2_POLICY_FINGERPRINT,
    RAG_V2_POLICY_ID,
    RAG_V2_POLICY_SCHEMA_VERSION,
    SUPPORTED_STATUS,
)


ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    ROOT
    / "data"
    / "linepulse"
    / "evaluation"
    / "rag_answerability_v2_policy.json"
)


def load_manifest():
    return json.loads(
        MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )


def test_v2_policy_constants_match_manifest():
    manifest = load_manifest()

    assert (
        manifest["policy_schema_version"]
        == RAG_V2_POLICY_SCHEMA_VERSION
    )

    assert (
        manifest["policy_id"]
        == RAG_V2_POLICY_ID
    )

    assert (
        manifest["threshold"]
        == RAG_V2_DENSE_TOP1_THRESHOLD
    )

    assert (
        manifest["development_dataset_sha256"]
        == RAG_V2_DEVELOPMENT_SHA256
    )

    assert (
        manifest["audit_dataset_sha256"]
        == RAG_V2_AUDIT_SHA256
    )

    assert (
        manifest["corpus_fingerprint"]
        == RAG_V2_CORPUS_FINGERPRINT
    )

    assert manifest["audit_used_for_selection"] is False
    assert manifest["audit_evaluated"] is False


def test_v2_policy_fingerprint_matches_manifest():
    manifest = load_manifest()

    expected_fingerprint = manifest.pop(
        "policy_fingerprint"
    )

    canonical = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode(
        "utf-8"
    )

    actual_fingerprint = hashlib.sha256(
        canonical
    ).hexdigest()

    assert (
        actual_fingerprint
        == expected_fingerprint
        == RAG_V2_POLICY_FINGERPRINT
    )


def test_v2_policy_threshold_is_inclusive():
    policy = DenseTop1AnswerabilityPolicyV2()

    decision = policy.decide(
        RAG_V2_DENSE_TOP1_THRESHOLD
    )

    assert decision.is_supported is True
    assert decision.status == SUPPORTED_STATUS
    assert decision.margin == pytest.approx(0.0)


def test_v2_policy_below_threshold_is_insufficient():
    policy = DenseTop1AnswerabilityPolicyV2()

    score = (
        RAG_V2_DENSE_TOP1_THRESHOLD
        - 1e-12
    )

    decision = policy.decide(
        score
    )

    assert decision.is_supported is False

    assert (
        decision.status
        == INSUFFICIENT_EVIDENCE_STATUS
    )

    assert decision.margin < 0.0


def test_v2_policy_above_threshold_is_supported():
    policy = DenseTop1AnswerabilityPolicyV2()

    score = (
        RAG_V2_DENSE_TOP1_THRESHOLD
        + 1e-12
    )

    decision = policy.decide(
        score
    )

    assert decision.is_supported is True
    assert decision.status == SUPPORTED_STATUS
    assert decision.margin > 0.0


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_v2_policy_rejects_non_finite_scores(
    value,
):
    policy = DenseTop1AnswerabilityPolicyV2()

    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        policy.decide(
            value
        )


def test_v2_policy_is_exported_from_package():
    assert (
        rag.DenseTop1AnswerabilityPolicyV2
        is DenseTop1AnswerabilityPolicyV2
    )

    assert (
        rag.RAG_V2_DENSE_TOP1_THRESHOLD
        == RAG_V2_DENSE_TOP1_THRESHOLD
    )

    assert (
        rag.RAG_V2_POLICY_FINGERPRINT
        == RAG_V2_POLICY_FINGERPRINT
    )
