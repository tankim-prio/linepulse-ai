from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from linepulse.rag import (
    RAG_V2_AUDIT_SHA256,
    RAG_V2_DENSE_TOP1_THRESHOLD,
    RAG_V2_POLICY_FINGERPRINT,
    RAG_V2_POLICY_ID,
)


ROOT = Path(__file__).resolve().parents[1]

SUMMARY_PATH = (
    ROOT
    / "reports"
    / "rag_answerability_v2_audit_summary.json"
)

PREDICTIONS_PATH = (
    ROOT
    / "reports"
    / "rag_answerability_v2_audit_predictions.csv"
)


def load_summary():
    return json.loads(
        SUMMARY_PATH.read_text(
            encoding="utf-8"
        )
    )


def test_audit_summary_is_bound_to_frozen_policy():
    summary = load_summary()

    assert (
        summary["policy_id"]
        == RAG_V2_POLICY_ID
    )

    assert (
        summary["policy_fingerprint"]
        == RAG_V2_POLICY_FINGERPRINT
    )

    assert (
        summary["threshold"]
        == RAG_V2_DENSE_TOP1_THRESHOLD
    )

    assert (
        summary["audit_dataset_sha256"]
        == RAG_V2_AUDIT_SHA256
    )


def test_audit_was_not_used_for_tuning():
    summary = load_summary()

    assert (
        summary["threshold_fitting_on_audit"]
        is False
    )

    assert (
        summary["model_training_on_audit"]
        is False
    )

    assert (
        summary["feature_selection_on_audit"]
        is False
    )

    assert (
        summary["candidate_selection_on_audit"]
        is False
    )

    assert (
        summary["policy_changed_after_audit"]
        is False
    )

    assert (
        summary["audit_used_for_selection"]
        is False
    )

    assert (
        summary["audit_evaluated"]
        is True
    )


def test_audit_predictions_hash_and_row_count():
    summary = load_summary()

    actual_sha256 = hashlib.sha256(
        PREDICTIONS_PATH.read_bytes()
    ).hexdigest()

    assert (
        actual_sha256
        == summary["predictions_sha256"]
    )

    with PREDICTIONS_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:

        rows = list(
            csv.DictReader(
                handle
            )
        )

    assert len(rows) == 48

    assert (
        sum(
            row["expected_status"]
            == "supported"
            for row in rows
        )
        == 32
    )

    assert (
        sum(
            row["expected_status"]
            == "insufficient_evidence"
            for row in rows
        )
        == 16
    )


def test_audit_confusion_matrix_matches_predictions():
    summary = load_summary()

    with PREDICTIONS_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:

        rows = list(
            csv.DictReader(
                handle
            )
        )

    tp = sum(
        row["expected_status"] == "supported"
        and row["predicted_status"] == "supported"
        for row in rows
    )

    fn = sum(
        row["expected_status"] == "supported"
        and row["predicted_status"] == "insufficient_evidence"
        for row in rows
    )

    tn = sum(
        row["expected_status"] == "insufficient_evidence"
        and row["predicted_status"] == "insufficient_evidence"
        for row in rows
    )

    fp = sum(
        row["expected_status"] == "insufficient_evidence"
        and row["predicted_status"] == "supported"
        for row in rows
    )

    metrics = summary[
        "overall_metrics"
    ]

    assert tp == metrics["tp"]
    assert fn == metrics["fn"]
    assert tn == metrics["tn"]
    assert fp == metrics["fp"]


def test_audit_summary_fingerprint():
    summary = load_summary()

    expected = summary.pop(
        "evaluation_fingerprint"
    )

    canonical = json.dumps(
        summary,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode(
        "utf-8"
    )

    actual = hashlib.sha256(
        canonical
    ).hexdigest()

    assert actual == expected
