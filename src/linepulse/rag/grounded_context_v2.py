"""Final Phase-4 grounded RAG integration boundary.

The service deliberately does not generate free-form language.

Responsibilities:
1. execute hybrid retrieval exactly once per query;
2. reuse dense scores already produced by hybrid retrieval;
3. apply the frozen RAG v2 answerability policy;
4. expose evidence only when the frozen policy permits generation;
5. bind every eligible evidence item to a chunk citation;
6. mark retrieved document text as untrusted input.

This keeps retrieval, answerability, and future generation separated.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .answerability_v2 import (
    DenseTop1AnswerabilityPolicyV2,
)
from .hybrid import (
    HybridRetriever,
)


UNTRUSTED_DOCUMENT_TEXT = (
    "untrusted_document_text"
)


class GroundedContextError(ValueError):
    """Invalid grounded-context request or retrieval state."""


@dataclass(
    frozen=True
)
class GroundedEvidenceItemV2:
    """One citation-bound piece of untrusted retrieved evidence."""

    rank: int
    chunk_id: str

    fusion_score: float
    dense_score: float
    lexical_score: float

    text: str

    trust_level: str = (
        UNTRUSTED_DOCUMENT_TEXT
    )


@dataclass(
    frozen=True
)
class GroundedContextV2:
    """Answerability-gated context for a downstream generator."""

    query: str

    status: str
    generation_allowed: bool

    dense_top1_score: float
    threshold: float
    margin: float

    policy_id: str
    policy_fingerprint: str

    evidence: tuple[
        GroundedEvidenceItemV2,
        ...
    ]

    citation_ids: tuple[
        str,
        ...
    ]


class GroundedContextServiceV2:
    """Hybrid retrieval + frozen v2 answerability integration.

    The HybridRetriever is called exactly once.

    To obtain the exact dense top-1 score without embedding the
    query a second time, the existing hybrid search is requested
    for the full current in-memory corpus. The service then:

    - uses max(result.dense_score) for answerability;
    - preserves hybrid RRF order for evidence;
    - slices only the requested top-k evidence after gating.

    This is appropriate for the current small exact corpus.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        *,
        policy: DenseTop1AnswerabilityPolicyV2 | None = None,
    ) -> None:

        self._retriever = retriever

        self._policy = (
            policy
            if policy is not None
            else DenseTop1AnswerabilityPolicyV2()
        )

        try:
            corpus_size = len(
                retriever.chunks
            )
        except Exception as exc:
            raise GroundedContextError(
                "retriever must expose a sized chunks collection"
            ) from exc

        if corpus_size <= 0:
            raise GroundedContextError(
                "retriever corpus must not be empty"
            )

        self._corpus_size = int(
            corpus_size
        )

    @property
    def corpus_size(
        self,
    ) -> int:
        """Number of chunks in the current exact in-memory corpus."""

        return self._corpus_size

    def build_context(
        self,
        query: str,
        *,
        top_k: int = 3,
    ) -> GroundedContextV2:
        """Build deterministic answerability-gated evidence context."""

        if not isinstance(
            query,
            str,
        ):
            raise GroundedContextError(
                "query must be a string"
            )

        normalized_query = (
            query.strip()
        )

        if not normalized_query:
            raise GroundedContextError(
                "query must not be empty"
            )

        if isinstance(
            top_k,
            bool,
        ) or not isinstance(
            top_k,
            int,
        ):
            raise GroundedContextError(
                "top_k must be an integer"
            )

        if top_k <= 0:
            raise GroundedContextError(
                "top_k must be greater than zero"
            )

        if top_k > self._corpus_size:
            raise GroundedContextError(
                "top_k cannot exceed corpus size"
            )

        # ----------------------------------------------------
        # Critical efficiency property:
        #
        # One HybridRetriever.search() call only.
        #
        # Hybrid retrieval performs the dense query embedding.
        # We reuse the dense scores returned from that search
        # instead of embedding the query again for the v2 gate.
        # ----------------------------------------------------

        results = self._retriever.search(
            normalized_query,
            top_k=self._corpus_size,
        )

        if len(
            results
        ) != self._corpus_size:
            raise GroundedContextError(
                "full-corpus hybrid retrieval returned "
                "an unexpected result count"
            )

        dense_scores = tuple(
            float(
                result.dense_score
            )
            for result in results
        )

        if not all(
            math.isfinite(
                score
            )
            for score in dense_scores
        ):
            raise GroundedContextError(
                "retrieval returned a non-finite dense score"
            )

        dense_top1_score = max(
            dense_scores
        )

        decision = self._policy.decide(
            dense_top1_score
        )

        # ----------------------------------------------------
        # Strict grounded-generation boundary.
        #
        # If evidence is judged insufficient, no document text
        # is released as generation context.
        # ----------------------------------------------------

        if not decision.is_supported:

            return GroundedContextV2(
                query=normalized_query,
                status=decision.status,
                generation_allowed=False,
                dense_top1_score=decision.dense_top1_score,
                threshold=decision.threshold,
                margin=decision.margin,
                policy_id=decision.policy_id,
                policy_fingerprint=decision.policy_fingerprint,
                evidence=(),
                citation_ids=(),
            )

        selected = results[
            :top_k
        ]

        evidence = tuple(
            GroundedEvidenceItemV2(
                rank=int(
                    result.rank
                ),
                chunk_id=str(
                    result.chunk.chunk_id
                ),
                fusion_score=float(
                    result.fusion_score
                ),
                dense_score=float(
                    result.dense_score
                ),
                lexical_score=float(
                    result.lexical_score
                ),
                text=str(
                    result.chunk.text
                ),
            )
            for result in selected
        )

        citation_ids = tuple(
            item.chunk_id
            for item in evidence
        )

        if len(
            set(
                citation_ids
            )
        ) != len(
            citation_ids
        ):
            raise GroundedContextError(
                "retrieval returned duplicate citation ids"
            )

        return GroundedContextV2(
            query=normalized_query,
            status=decision.status,
            generation_allowed=True,
            dense_top1_score=decision.dense_top1_score,
            threshold=decision.threshold,
            margin=decision.margin,
            policy_id=decision.policy_id,
            policy_fingerprint=decision.policy_fingerprint,
            evidence=evidence,
            citation_ids=citation_ids,
        )
