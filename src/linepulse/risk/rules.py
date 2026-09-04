"""Deterministic progress-gap risk baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


RULE_VERSION = "progress-gap-v1"


@dataclass(frozen=True)
class RuleRiskAssessment:
    rule_version: str
    risk_score: float
    factors: tuple[str, ...]


def _number(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(number):
        return default

    return number


def evaluate_rule_risk(
    snapshot: Mapping[str, object],
) -> RuleRiskAssessment:
    """Calculate transparent risk from progress against snapshot target."""

    if "progress_ratio" not in snapshot:
        raise ValueError("Snapshot is missing progress_ratio.")

    progress = _number(snapshot["progress_ratio"], default=-1.0)

    if progress < 0:
        raise ValueError("progress_ratio must be non-negative.")

    risk_score = round(
        max(0.0, min(1.0, 1.0 - progress)),
        4,
    )

    factors: list[str] = []

    if progress < 1.0:
        factors.append("progress_below_snapshot_target")

    planned = _number(snapshot.get("planned_workers"))
    present = _number(snapshot.get("present_workers"))

    if planned > 0 and present < planned:
        factors.append("attendance_shortfall")

    if _number(snapshot.get("downtime_to_snapshot")) > 0:
        factors.append("downtime_observed")

    if _number(snapshot.get("defects_to_snapshot")) > 0:
        factors.append("defects_observed")

    material = str(
        snapshot.get("material_status", "")
    ).strip().lower()

    if material and material != "available":
        factors.append("material_constraint")

    root_cause = str(
        snapshot.get("root_cause_signal", "none")
    ).strip().lower()

    if root_cause and root_cause != "none":
        factors.append(f"root_cause:{root_cause}")

    return RuleRiskAssessment(
        rule_version=RULE_VERSION,
        risk_score=risk_score,
        factors=tuple(factors),
    )
