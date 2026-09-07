"""Calibration-frozen answerability gate for LinePulse RAG."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite

from linepulse.rag.chunks import CHUNK_SCHEMA_VERSION
from linepulse.rag.embeddings import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_REVISION,
)
from linepulse.rag.evidence import (
    EvidenceBundle,
    EvidenceRecord,
)


ANSWERABILITY_POLICY_ID = "e5-top1-midpoint-v1"

ANSWERABILITY_THRESHOLD = 0.8190949857234955

ANSWERABILITY_CALIBRATION_NEGATIVE_MAX = 0.818340539932251
ANSWERABILITY_CALIBRATION_POSITIVE_MIN = 0.81984943151474

ANSWERABILITY_CALIBRATION_ROW_COUNT = 46
ANSWERABILITY_CALIBRATION_SUPPORTED_COUNT = 23
ANSWERABILITY_CALIBRATION_INSUFFICIENT_COUNT = 23

ANSWERABILITY_CALIBRATION_SHA256 = (
    "7554f5361f955898f7d1cf467570d301"
    "a4d6026e0f1bf4e8f9207b1e0c15bca1"
)

ANSWERABILITY_MODEL_NAME = "intfloat/multilingual-e5-small"

ANSWERABILITY_MODEL_REVISION = (
    "614241f622f53c4eeff9890bdc4f31cfecc418b3"
)

ANSWERABILITY_CHUNK_SCHEMA_VERSION = "bilingual-section-v1"

ANSWERABILITY_CALIBRATION_GAP = (
    ANSWERABILITY_CALIBRATION_POSITIVE_MIN
    - ANSWERABILITY_CALIBRATION_NEGATIVE_MAX
)

ANSWERABILITY_POLICY_FINGERPRINT = (
    "8b3fb0ebde2632490790a44cc92cc44a"
    "1c1f0c86603cd653c38e0d3190034dea"
)


SUPPORTED_STATUS = "supported"
INSUFFICIENT_EVIDENCE_STATUS = "insufficient_evidence"

SUPPORTED_REASON_CODE = (
    "top1_at_or_above_calibrated_threshold"
)

INSUFFICIENT_REASON_CODE = (
    "top1_below_calibrated_threshold"
)

CALIBRATION_BOUNDARY_WARNING = (
    "score_inside_calibration_separation_gap"
)


class AnswerabilityError(RuntimeError):
    """Raised when evidence cannot be evaluated by the frozen policy."""


@dataclass(frozen=True)
class AnswerabilityDecision:
    """Transparent deterministic result from the frozen policy."""

    query: str

    status: str
    answerable: bool
    reason_code: str

    top1_score: float
    threshold: float
    margin_to_threshold: float

    top1_chunk_id: str
    evidence_count: int

    calibration_boundary_zone: bool
    warnings: tuple[str, ...]

    policy_id: str
    policy_fingerprint: str

    model_name: str
    model_revision: str
    chunk_schema_version: str


def _canonical_policy_payload() -> str:
    """Return the canonical frozen-policy fingerprint payload."""

    return "\n".join(
        [
            ANSWERABILITY_POLICY_ID,
            format(
                ANSWERABILITY_THRESHOLD,
                ".17g",
            ),
            format(
                ANSWERABILITY_CALIBRATION_NEGATIVE_MAX,
                ".17g",
            ),
            format(
                ANSWERABILITY_CALIBRATION_POSITIVE_MIN,
                ".17g",
            ),
            str(
                ANSWERABILITY_CALIBRATION_ROW_COUNT
            ),
            str(
                ANSWERABILITY_CALIBRATION_SUPPORTED_COUNT
            ),
            str(
                ANSWERABILITY_CALIBRATION_INSUFFICIENT_COUNT
            ),
            ANSWERABILITY_CALIBRATION_SHA256,
            ANSWERABILITY_MODEL_NAME,
            ANSWERABILITY_MODEL_REVISION,
            ANSWERABILITY_CHUNK_SCHEMA_VERSION,
        ]
    )


def _computed_policy_fingerprint() -> str:
    """Compute the fingerprint from current frozen metadata."""

    return sha256(
        _canonical_policy_payload().encode(
            "utf-8"
        )
    ).hexdigest()


class AnswerabilityGate:
    """Apply the calibration-frozen top-1 similarity policy."""

    @property
    def threshold(self) -> float:
        return ANSWERABILITY_THRESHOLD

    @property
    def policy_id(self) -> str:
        return ANSWERABILITY_POLICY_ID

    @property
    def policy_fingerprint(self) -> str:
        return ANSWERABILITY_POLICY_FINGERPRINT

    def _validate_runtime_contract(self) -> None:
        """Fail closed if frozen retrieval assumptions changed."""

        if (
            EMBEDDING_MODEL_NAME
            != ANSWERABILITY_MODEL_NAME
        ):
            raise AnswerabilityError(
                "Embedding model name changed; "
                "recalibration is required."
            )

        if (
            EMBEDDING_MODEL_REVISION
            != ANSWERABILITY_MODEL_REVISION
        ):
            raise AnswerabilityError(
                "Embedding model revision changed; "
                "recalibration is required."
            )

        if (
            CHUNK_SCHEMA_VERSION
            != ANSWERABILITY_CHUNK_SCHEMA_VERSION
        ):
            raise AnswerabilityError(
                "Chunk schema changed; "
                "recalibration is required."
            )

        current_fingerprint = (
            _computed_policy_fingerprint()
        )

        if (
            current_fingerprint
            != ANSWERABILITY_POLICY_FINGERPRINT
        ):
            raise AnswerabilityError(
                "Frozen answerability policy metadata changed; "
                "recalibration or a new policy version is required."
            )

    def _validate_bundle(
        self,
        bundle: EvidenceBundle,
    ) -> tuple[EvidenceRecord, ...]:
        """Validate the entire evidence package before classification."""

        if not isinstance(
            bundle,
            EvidenceBundle,
        ):
            raise AnswerabilityError(
                "Expected an EvidenceBundle."
            )

        if not isinstance(
            bundle.query,
            str,
        ):
            raise AnswerabilityError(
                "Evidence bundle query must be a string."
            )

        query = bundle.query.strip()

        if not query:
            raise AnswerabilityError(
                "Evidence bundle query must not be blank."
            )

        if (
            not isinstance(
                bundle.top_k,
                int,
            )
            or isinstance(
                bundle.top_k,
                bool,
            )
            or bundle.top_k < 1
        ):
            raise AnswerabilityError(
                "Evidence bundle top_k must be "
                "a positive integer."
            )

        evidence = tuple(
            bundle.evidence
        )

        if len(evidence) != bundle.top_k:
            raise AnswerabilityError(
                "Evidence count does not match top_k."
            )

        if not evidence:
            raise AnswerabilityError(
                "At least one evidence record is required."
            )

        expected_ranks = list(
            range(
                1,
                bundle.top_k + 1,
            )
        )

        actual_ranks = []

        seen_chunk_ids: set[str] = set()
        previous_score: float | None = None

        for index, record in enumerate(
            evidence
        ):
            if not isinstance(
                record,
                EvidenceRecord,
            ):
                raise AnswerabilityError(
                    "Every evidence item must be "
                    "an EvidenceRecord."
                )

            actual_ranks.append(
                record.rank
            )

            if (
                not isinstance(
                    record.rank,
                    int,
                )
                or isinstance(
                    record.rank,
                    bool,
                )
            ):
                raise AnswerabilityError(
                    "Evidence rank must be an integer."
                )

            if (
                record.chunk_schema_version
                != ANSWERABILITY_CHUNK_SCHEMA_VERSION
            ):
                raise AnswerabilityError(
                    "Evidence chunk schema does not match "
                    "the calibrated policy."
                )

            if (
                not isinstance(
                    record.chunk_id,
                    str,
                )
                or not record.chunk_id.strip()
            ):
                raise AnswerabilityError(
                    "Evidence chunk_id must not be blank."
                )

            if record.chunk_id in seen_chunk_ids:
                raise AnswerabilityError(
                    "Duplicate evidence chunk_id detected."
                )

            seen_chunk_ids.add(
                record.chunk_id
            )

            try:
                score = float(
                    record.score
                )
            except (
                TypeError,
                ValueError,
            ) as exc:
                raise AnswerabilityError(
                    "Evidence score must be numeric."
                ) from exc

            if not isfinite(
                score
            ):
                raise AnswerabilityError(
                    "Evidence score must be finite."
                )

            if not (
                -1.0
                <= score
                <= 1.0
            ):
                raise AnswerabilityError(
                    "Evidence cosine similarity score "
                    "must be within [-1, 1]."
                )

            if (
                previous_score is not None
                and score
                > previous_score + 1e-12
            ):
                raise AnswerabilityError(
                    "Evidence scores must be "
                    "non-increasing by rank."
                )

            previous_score = score

        if actual_ranks != expected_ranks:
            raise AnswerabilityError(
                "Evidence ranks must be contiguous "
                "and start at 1."
            )

        return evidence

    def decide(
        self,
        bundle: EvidenceBundle,
    ) -> AnswerabilityDecision:
        """Classify one validated evidence bundle."""

        self._validate_runtime_contract()

        evidence = self._validate_bundle(
            bundle
        )

        query = bundle.query.strip()

        top1 = evidence[0]

        score = float(
            top1.score
        )

        answerable = (
            score
            >= ANSWERABILITY_THRESHOLD
        )

        if answerable:
            status = SUPPORTED_STATUS
            reason_code = SUPPORTED_REASON_CODE

        else:
            status = INSUFFICIENT_EVIDENCE_STATUS
            reason_code = INSUFFICIENT_REASON_CODE

        calibration_boundary_zone = (
            ANSWERABILITY_CALIBRATION_NEGATIVE_MAX
            < score
            < ANSWERABILITY_CALIBRATION_POSITIVE_MIN
        )

        warnings: tuple[str, ...]

        if calibration_boundary_zone:
            warnings = (
                CALIBRATION_BOUNDARY_WARNING,
            )
        else:
            warnings = ()

        return AnswerabilityDecision(
            query=query,
            status=status,
            answerable=answerable,
            reason_code=reason_code,
            top1_score=score,
            threshold=ANSWERABILITY_THRESHOLD,
            margin_to_threshold=(
                score
                - ANSWERABILITY_THRESHOLD
            ),
            top1_chunk_id=top1.chunk_id,
            evidence_count=len(
                evidence
            ),
            calibration_boundary_zone=(
                calibration_boundary_zone
            ),
            warnings=warnings,
            policy_id=ANSWERABILITY_POLICY_ID,
            policy_fingerprint=(
                ANSWERABILITY_POLICY_FINGERPRINT
            ),
            model_name=ANSWERABILITY_MODEL_NAME,
            model_revision=ANSWERABILITY_MODEL_REVISION,
            chunk_schema_version=(
                ANSWERABILITY_CHUNK_SCHEMA_VERSION
            ),
        )
