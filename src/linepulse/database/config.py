"""Database configuration for LinePulse AI."""

from __future__ import annotations

import os


DATABASE_URL = os.getenv(
    "LINEPULSE_DATABASE_URL",
    "postgresql+psycopg2://postgres@localhost:5432/linepulse",
)
