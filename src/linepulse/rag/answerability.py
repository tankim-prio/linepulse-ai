"""Calibration-frozen answerability gate for LinePulse RAG."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from linepulse.rag.chunks import CHUNK_SCHEMA_VERSION
from linepulse.rag.embeddings import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_REVISION,
)
from linepulse.rag.evidence import EvidenceBundle


ANSWERABILITY_POLICY_ID = "e5-top1-midpoint-v1"

ANSWERABILITY_THRESHOLD = 0.8190949857234955

ANSWERABILITY_CALIBRATION_NEGATIVE_MAX = 0.818340539932251
ANSWERABILITY_CALIBRATION_POSITIVE_MIN = 0.81984943151474

ANSWERABILITY_CALIBRATION_ROW_COUNT = 46
ANSWERABILITY_CALIBRATION_SUPPORTED_COUNT = 23
ANSWERABILITY_CALIBRATION_INSUFFICIENT_COUNT = 23

ANSWERABILITY_CALIBRATION_SHA256 = "7554f5361f955898f7d1cf467570d301a4d6026e0f1bf4e8f9207b1e0c15bca1"

ANSWERABILITY_MODEL_NAME = "intfloat/multilingual-e5-small"
ANSWERABILITY_MODEL_REVISION = (
    "614241f622f53c4eeff9890bdc4f31cfecc418b3"
)

ANSWERABILITY_CHUNK_SCHEMA_VERSION = "bilingual-section-v1"

SUPPORTED_STATUS = "supported"
INSUFFICIENT_EVIDENCE_STATUS = "insufficient_evidence"


class AnswerabilityError(RuntimeError):
    """Raised when evidence cannot be evaluated by the frozen policy."""


@dataclass(frozen=True)
class AnswerabilityDecision:
    """Transparent result from the frozen answerability policy."""

    query: str
    status: str
    answerable: bool
    top1_score: float
    threshold: float
    margin_to_threshold: float
    policy_id: str
    model_name: str
    model_revision: str
    chunk_schema_version: str


class AnswerabilityGate:
    """Apply the calibration-frozen top-1 similarity policy."""

    @property
    def threshold(self) -> float:
        return ANSWERABILITY_THRESHOLD

    @property
    def policy_id(self) -> str:
        return ANSWERABILITY_POLICY_ID

    def _validate_runtime_contract(self) -> None:
        if EMBEDDING_MODEL_NAME != ANSWERABILITY_MODEL_NAME:
            raise AnswerabilityError(
                "Embedding model name changed; recalibration is required."
            )

        if EMBEDDING_MODEL_REVISION != ANSWERABILITY_MODEL_REVISION:
            raise AnswerabilityError(
                "Embedding model revision changed; recalibration is required."
            )

        if CHUNK_SCHEMA_VERSION != ANSWERABILITY_CHUNK_SCHEMA_VERSION:
            raise AnswerabilityError(
                "Chunk schema changed; recalibration is required."
            )

    def decide(
        self,
        bundle: EvidenceBundle,
    ) -> AnswerabilityDecision:
        """Classify one evidence bundle without generating an answer."""

        self._validate_runtime_contract()

        if not isinstance(bundle, EvidenceBundle):
            raise AnswerabilityError(
                "Expected an EvidenceBundle."
            )

        query = bundle.query.strip()

        if not query:
            raise AnswerabilityError(
                "Evidence bundle query must not be blank."
            )

        if bundle.top_k < 1:
            raise AnswerabilityError(
                "Evidence bundle top_k must be at least 1."
            )

        if len(bundle.evidence) != bundle.top_k:
            raise AnswerabilityError(
                "Evidence count does not match top_k."
            )

        if not bundle.evidence:
            raise AnswerabilityError(
                "At least one evidence record is required."
            )

        top1 = bundle.evidence[0]

        if top1.rank != 1:
            raise AnswerabilityError(
                "First evidence record must have rank 1."
            )

        if (
            top1.chunk_schema_version
            != ANSWERABILITY_CHUNK_SCHEMA_VERSION
        ):
            raise AnswerabilityError(
                "Evidence chunk schema does not match "
                "the calibrated policy."
            )

        score = float(top1.score)

        if not isfinite(score):
            raise AnswerabilityError(
                "Top-1 score must be finite."
            )

        answerable = (
            score >= ANSWERABILITY_THRESHOLD
        )

        status = (
            SUPPORTED_STATUS
            if answerable
            else INSUFFICIENT_EVIDENCE_STATUS
        )

        return AnswerabilityDecision(
            query=query,
            status=status,
            answerable=answerable,
            top1_score=score,
            threshold=ANSWERABILITY_THRESHOLD,
            margin_to_threshold=(
                score
                - ANSWERABILITY_THRESHOLD
            ),
            policy_id=ANSWERABILITY_POLICY_ID,
            model_name=ANSWERABILITY_MODEL_NAME,
            model_revision=ANSWERABILITY_MODEL_REVISION,
            chunk_schema_version=(
                ANSWERABILITY_CHUNK_SCHEMA_VERSION
            ),
        )
