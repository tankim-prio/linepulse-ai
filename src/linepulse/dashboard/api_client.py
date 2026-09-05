"""HTTP client used by the Streamlit dashboard.

The dashboard communicates with LinePulse only through FastAPI.
It must not access PostgreSQL or database repositories directly.
"""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_API_URL = "http://127.0.0.1:8000"


class LinePulseApiError(RuntimeError):
    """Raised when the LinePulse FastAPI service cannot be read."""


class LinePulseApiClient:
    """Small read-only HTTP client for LinePulse FastAPI."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 5.0,
    ) -> None:
        resolved_url = (
            base_url
            or os.getenv(
                "LINEPULSE_API_URL"
            )
            or DEFAULT_API_URL
        )

        self.base_url = resolved_url.rstrip("/")
        self.timeout = timeout

    def health(
        self,
    ) -> dict[str, Any]:
        return self._get(
            "/health"
        )

    def database_health(
        self,
    ) -> dict[str, Any]:
        return self._get(
            "/health/database"
        )

    def factories(
        self,
    ) -> list[dict[str, Any]]:
        return self._get(
            "/api/factories"
        )

    def production_lines(
        self,
        *,
        factory_id: str | None = None,
        active: bool | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, str] = {}

        if factory_id is not None:
            params["factory_id"] = factory_id

        if active is not None:
            params["active"] = (
                "true"
                if active
                else "false"
            )

        return self._get(
            "/api/production-lines",
            params=params,
        )

    def analytics_overview(
        self,
    ) -> dict[str, Any]:
        return self._get(
            "/api/analytics/overview"
        )

    def analytics_lines(
        self,
        *,
        factory_id: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, str] = {}

        if factory_id is not None:
            params["factory_id"] = factory_id

        return self._get(
            "/api/analytics/lines",
            params=params,
        )

    def risk_trend(
        self,
        *,
        factory_id: str | None = None,
        line_id: str | None = None,
        rule_version: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, str] = {}

        if factory_id is not None:
            params["factory_id"] = factory_id

        if line_id is not None:
            params["line_id"] = line_id

        if rule_version is not None:
            params["rule_version"] = rule_version

        return self._get(
            "/api/analytics/risk-trend",
            params=params,
        )

    def factor_summaries(
        self,
        *,
        factory_id: str | None = None,
        line_id: str | None = None,
        rule_version: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, str] = {}

        if factory_id is not None:
            params["factory_id"] = factory_id

        if line_id is not None:
            params["line_id"] = line_id

        if rule_version is not None:
            params["rule_version"] = rule_version

        return self._get(
            "/api/analytics/factors",
            params=params,
        )

    def risk_events(
        self,
        *,
        limit: int = 10,
        factory_id: str | None = None,
        line_id: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, str] = {
            "limit": str(limit),
        }

        if factory_id is not None:
            params["factory_id"] = factory_id

        if line_id is not None:
            params["line_id"] = line_id

        return self._get(
            "/api/risk-events",
            params=params,
        )

    def _get(
        self,
        path: str,
        *,
        params: dict[str, str] | None = None,
    ) -> Any:
        url = self.base_url + path

        if params:
            url = (
                url
                + "?"
                + urlencode(params)
            )

        request = Request(
            url,
            method="GET",
            headers={
                "Accept": "application/json",
            },
        )

        try:
            with urlopen(
                request,
                timeout=self.timeout,
            ) as response:
                payload = response.read()

        except HTTPError as exc:
            raise LinePulseApiError(
                "LinePulse API returned "
                f"HTTP {exc.code} for {url}."
            ) from exc

        except URLError as exc:
            raise LinePulseApiError(
                "Could not connect to "
                f"LinePulse API at {url}: "
                f"{exc.reason}"
            ) from exc

        try:
            return json.loads(
                payload.decode("utf-8")
            )

        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise LinePulseApiError(
                "LinePulse API returned "
                f"invalid JSON for {url}."
            ) from exc
