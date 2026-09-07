"""Frozen RAG v2 answerability policy.

This module contains the development-selected policy frozen before
the v2 audit dataset is evaluated.

Hybrid RRF remains the evidence-ranking mechanism.  This policy uses
the raw dense top-1 cosine score only for answerability classification.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


RAG_V2_POLICY_SCHEMA_VERSION = "rag-answerability-v2-policy-v1"
RAG_V2_POLICY_ID = "dense-top1-development-fit-v2"

RAG_V2_DENSE_TOP1_THRESHOLD = 0.8415164947509766

RAG_V2_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
RAG_V2_EMBEDDING_MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
RAG_V2_EMBEDDING_DIMENSION = 384

RAG_V2_CORPUS_FINGERPRINT = "59f725c9b3b67abce85fb38b4860c1dbe7c5d9f5f4094a8b5ae60e26f31fa1a4"

RAG_V2_DEVELOPMENT_SHA256 = "68b973e4b04fb001379fc265de0d40cd0ff4bf6b5e5e3c085d0cf0a6b15c6b8a"
RAG_V2_AUDIT_SHA256 = "2e9ccd4c20e023a5a52081eb3a8e84f1a8eb0a3a6b0921447c3adb4eab3495c9"

RAG_V2_POLICY_FINGERPRINT = "59e9ab2bdf28abf157b3981a768ce2a646724dca294aff7525e1131d8e8f189f"

SUPPORTED_STATUS = "supported"
INSUFFICIENT_EVIDENCE_STATUS = "insufficient_evidence"


@dataclass(frozen=True)
class AnswerabilityDecisionV2:
    """Deterministic answerability decision."""

    status: str
    dense_top1_score: float
    threshold: float
    margin: float
    policy_id: str
    policy_fingerprint: str
    is_supported: bool


class DenseTop1AnswerabilityPolicyV2:
    """Frozen v2 dense-top1 answerability gate."""

    policy_id = RAG_V2_POLICY_ID
    policy_fingerprint = RAG_V2_POLICY_FINGERPRINT
    threshold = RAG_V2_DENSE_TOP1_THRESHOLD

    def decide(
        self,
        dense_top1_score: float,
    ) -> AnswerabilityDecisionV2:
        """Classify one finite dense top-1 cosine score."""

        score = float(
            dense_top1_score
        )

        if not math.isfinite(
            score
        ):
            raise ValueError(
                "dense_top1_score must be finite"
            )

        supported = (
            score
            >= self.threshold
        )

        status = (
            SUPPORTED_STATUS
            if supported
            else INSUFFICIENT_EVIDENCE_STATUS
        )

        return AnswerabilityDecisionV2(
            status=status,
            dense_top1_score=score,
            threshold=self.threshold,
            margin=(
                score
                - self.threshold
            ),
            policy_id=self.policy_id,
            policy_fingerprint=self.policy_fingerprint,
            is_supported=supported,
        )
