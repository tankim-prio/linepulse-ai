from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from linepulse.dashboard.api_client import (
    LinePulseApiClient,
    LinePulseApiError,
)


class FakeResponse:
    def __init__(
        self,
        body,
    ):
        self.body = body

    def __enter__(
        self,
    ):
        return self

    def __exit__(
        self,
        exc_type,
        exc,
        traceback,
    ):
        return False

    def read(
        self,
    ):
        return json.dumps(
            self.body
        ).encode("utf-8")


class DashboardApiClientTests(
    unittest.TestCase
):
    def test_overview_uses_fastapi_url(
        self,
    ):
        client = LinePulseApiClient(
            "http://example.test:8000"
        )

        with patch(
            "linepulse.dashboard.api_client.urlopen",
            return_value=FakeResponse(
                {
                    "factory_count": 1,
                    "production_line_count": 8,
                    "risk_event_count": 1224,
                }
            ),
        ) as mocked:
            result = (
                client.analytics_overview()
            )

        self.assertEqual(
            result["risk_event_count"],
            1224,
        )

        request = (
            mocked.call_args.args[0]
        )

        self.assertEqual(
            request.full_url,
            (
                "http://example.test:8000"
                "/api/analytics/overview"
            ),
        )

    def test_line_query_encodes_factory(
        self,
    ):
        client = LinePulseApiClient(
            "http://example.test"
        )

        with patch(
            "linepulse.dashboard.api_client.urlopen",
            return_value=FakeResponse(
                []
            ),
        ) as mocked:
            client.analytics_lines(
                factory_id="FAC-001"
            )

        request = (
            mocked.call_args.args[0]
        )

        self.assertEqual(
            request.full_url,
            (
                "http://example.test"
                "/api/analytics/lines"
                "?factory_id=FAC-001"
            ),
        )

    def test_risk_event_filters_are_encoded(
        self,
    ):
        client = LinePulseApiClient(
            "http://example.test"
        )

        with patch(
            "linepulse.dashboard.api_client.urlopen",
            return_value=FakeResponse(
                []
            ),
        ) as mocked:
            client.risk_events(
                limit=5,
                factory_id="FAC-001",
                line_id="LINE-01",
            )

        request = (
            mocked.call_args.args[0]
        )

        self.assertIn(
            "limit=5",
            request.full_url,
        )

        self.assertIn(
            "factory_id=FAC-001",
            request.full_url,
        )

        self.assertIn(
            "line_id=LINE-01",
            request.full_url,
        )

    def test_risk_trend_filters_are_encoded(
        self,
    ):
        client = LinePulseApiClient(
            "http://example.test"
        )

        with patch(
            "linepulse.dashboard.api_client.urlopen",
            return_value=FakeResponse(
                []
            ),
        ) as mocked:
            client.risk_trend(
                factory_id="FAC-001",
                line_id="LINE-01",
                rule_version="progress-gap-v1",
            )

        request = (
            mocked.call_args.args[0]
        )

        self.assertTrue(
            request.full_url.startswith(
                "http://example.test"
                "/api/analytics/risk-trend?"
            )
        )

        self.assertIn(
            "factory_id=FAC-001",
            request.full_url,
        )

        self.assertIn(
            "line_id=LINE-01",
            request.full_url,
        )

        self.assertIn(
            "rule_version=progress-gap-v1",
            request.full_url,
        )

    def test_factor_summary_filters_are_encoded(
        self,
    ):
        client = LinePulseApiClient(
            "http://example.test"
        )

        with patch(
            "linepulse.dashboard.api_client.urlopen",
            return_value=FakeResponse(
                []
            ),
        ) as mocked:
            client.factor_summaries(
                factory_id="FAC-001",
                line_id="LINE-01",
                rule_version="progress-gap-v1",
            )

        request = (
            mocked.call_args.args[0]
        )

        self.assertTrue(
            request.full_url.startswith(
                "http://example.test"
                "/api/analytics/factors?"
            )
        )

        self.assertIn(
            "factory_id=FAC-001",
            request.full_url,
        )

        self.assertIn(
            "line_id=LINE-01",
            request.full_url,
        )

        self.assertIn(
            "rule_version=progress-gap-v1",
            request.full_url,
        )

    def test_invalid_json_raises_api_error(
        self,
    ):
        class InvalidResponse:
            def __enter__(self):
                return self

            def __exit__(
                self,
                exc_type,
                exc,
                traceback,
            ):
                return False

            def read(self):
                return b"not-json"

        client = LinePulseApiClient(
            "http://example.test"
        )

        with patch(
            "linepulse.dashboard.api_client.urlopen",
            return_value=InvalidResponse(),
        ):
            with self.assertRaises(
                LinePulseApiError
            ):
                client.health()

    def test_dashboard_has_no_database_imports(
        self,
    ):
        dashboard_dir = Path(
            "src/linepulse/dashboard"
        )

        forbidden_prefixes = (
            "linepulse.database",
            "sqlalchemy",
            "psycopg2",
        )

        for path in dashboard_dir.glob(
            "*.py"
        ):
            tree = ast.parse(
                path.read_text(
                    encoding="utf-8-sig"
                )
            )

            imports = []

            for node in ast.walk(
                tree
            ):
                if isinstance(
                    node,
                    ast.Import,
                ):
                    imports.extend(
                        alias.name
                        for alias
                        in node.names
                    )

                elif isinstance(
                    node,
                    ast.ImportFrom,
                ):
                    if node.module:
                        imports.append(
                            node.module
                        )

            for imported in imports:
                self.assertFalse(
                    imported.startswith(
                        forbidden_prefixes
                    ),
                    msg=(
                        f"{path} directly imports "
                        f"forbidden backend module "
                        f"{imported}"
                    ),
                )


if __name__ == "__main__":
    unittest.main()
