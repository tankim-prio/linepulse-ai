"""Create and persist idempotent LinePulse risk events."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from linepulse.risk.rules import RuleRiskAssessment


@dataclass(frozen=True)
class RiskEvent:
    event_id: str
    factory_id: str
    line_id: str
    order_id: str
    snapshot_at: str
    rule_version: str
    risk_score: float
    factors: tuple[str, ...]


def _timestamp_text(value: object) -> str:
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)


def build_risk_event(
    snapshot: Mapping[str, object],
    assessment: RuleRiskAssessment,
) -> RiskEvent:

    required = (
        "factory_id",
        "line_id",
        "order_id",
        "snapshot_at",
    )

    missing = [
        key
        for key in required
        if key not in snapshot
    ]

    if missing:
        raise ValueError(
            f"Snapshot is missing event fields: {missing}"
        )

    snapshot_at = _timestamp_text(
        snapshot["snapshot_at"]
    )

    identity = "|".join(
        [
            str(snapshot["factory_id"]),
            str(snapshot["line_id"]),
            str(snapshot["order_id"]),
            snapshot_at,
            assessment.rule_version,
        ]
    )

    event_id = hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()[:24]

    return RiskEvent(
        event_id=event_id,
        factory_id=str(snapshot["factory_id"]),
        line_id=str(snapshot["line_id"]),
        order_id=str(snapshot["order_id"]),
        snapshot_at=snapshot_at,
        rule_version=assessment.rule_version,
        risk_score=assessment.risk_score,
        factors=assessment.factors,
    )


class RiskEventStore:
    """Small local JSONL store until PostgreSQL persistence is introduced."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def append(self, event: RiskEvent) -> bool:
        """Append once. Return False when event already exists."""

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if event.event_id in self._existing_ids():
            return False

        payload = json.dumps(
            asdict(event),
            ensure_ascii=False,
            sort_keys=True,
        )

        with self.path.open(
            "a",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            handle.write(payload + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        return True

    def _existing_ids(self) -> set[str]:
        if not self.path.exists():
            return set()

        ids: set[str] = set()

        with self.path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            for line in handle:
                line = line.strip()

                if not line:
                    continue

                record = json.loads(line)

                event_id = record.get("event_id")

                if event_id:
                    ids.add(str(event_id))

        return ids
