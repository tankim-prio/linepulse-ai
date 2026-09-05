from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from linepulse.database.reference_repository import (
    FactoryRecord,
    ProductionLineRecord,
)
from linepulse.main import (
    app,
    get_reference_data_repository,
)


class FakeReferenceRepository:
    def __init__(self):
        self.factory = FactoryRecord(
            factory_id="FAC-001",
            factory_name="Demo Apparel Works",
            country="Bangladesh",
            timezone="Asia/Dhaka",
            weekly_closure_day="Friday",
            dataset_provenance="synthetic",
        )

        self.lines = [
            ProductionLineRecord(
                line_id="LINE-01",
                factory_id="FAC-001",
                line_name="Knit Line 1",
                specialization="knit",
                standard_operator_capacity=48,
                planning_efficiency=0.71,
                active=True,
                dataset_provenance="synthetic",
            ),
            ProductionLineRecord(
                line_id="LINE-02",
                factory_id="FAC-001",
                line_name="Knit Line 2",
                specialization="knit",
                standard_operator_capacity=52,
                planning_efficiency=0.73,
                active=False,
                dataset_provenance="synthetic",
            ),
        ]

    def list_factories(self):
        return [
            self.factory
        ]

    def get_factory(
        self,
        factory_id,
    ):
        if factory_id == self.factory.factory_id:
            return self.factory

        return None

    def list_production_lines(
        self,
        *,
        factory_id=None,
        active=None,
    ):
        records = list(
            self.lines
        )

        if factory_id is not None:
            records = [
                record
                for record in records
                if record.factory_id
                == factory_id
            ]

        if active is not None:
            records = [
                record
                for record in records
                if record.active
                == active
            ]

        return records

    def get_production_line(
        self,
        line_id,
    ):
        for record in self.lines:
            if record.line_id == line_id:
                return record

        return None


class ReferenceApiTests(unittest.TestCase):
    def setUp(self):
        self.repository = FakeReferenceRepository()

        app.dependency_overrides[
            get_reference_data_repository
        ] = lambda: self.repository

        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()

    def test_list_factories_returns_200(self):
        response = self.client.get(
            "/api/factories"
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        body = response.json()

        self.assertEqual(
            len(body),
            1,
        )

        self.assertEqual(
            body[0]["factory_id"],
            "FAC-001",
        )

        self.assertEqual(
            body[0]["factory_name"],
            "Demo Apparel Works",
        )

    def test_get_factory_returns_200(self):
        response = self.client.get(
            "/api/factories/FAC-001"
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertEqual(
            response.json()["timezone"],
            "Asia/Dhaka",
        )

    def test_missing_factory_returns_404(self):
        response = self.client.get(
            "/api/factories/FAC-MISSING"
        )

        self.assertEqual(
            response.status_code,
            404,
        )

        self.assertEqual(
            response.json(),
            {
                "detail": "Factory not found."
            },
        )

    def test_list_production_lines_returns_200(self):
        response = self.client.get(
            "/api/production-lines",
            params={
                "factory_id": "FAC-001",
            },
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        body = response.json()

        self.assertEqual(
            len(body),
            2,
        )

        self.assertEqual(
            body[0]["line_id"],
            "LINE-01",
        )

    def test_production_line_active_filter(self):
        response = self.client.get(
            "/api/production-lines",
            params={
                "active": "true",
            },
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        body = response.json()

        self.assertEqual(
            len(body),
            1,
        )

        self.assertEqual(
            body[0]["line_id"],
            "LINE-01",
        )

        self.assertTrue(
            body[0]["active"]
        )

    def test_get_production_line_and_missing_404(
        self,
    ):
        response = self.client.get(
            "/api/production-lines/LINE-02"
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertEqual(
            response.json()["line_name"],
            "Knit Line 2",
        )

        missing = self.client.get(
            "/api/production-lines/LINE-MISSING"
        )

        self.assertEqual(
            missing.status_code,
            404,
        )

        self.assertEqual(
            missing.json(),
            {
                "detail":
                    "Production line not found."
            },
        )


if __name__ == "__main__":
    unittest.main()
