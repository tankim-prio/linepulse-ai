"""Project paths and environment-backed settings."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / os.getenv(
    "LINEPULSE_DATA_DIR",
    "data/linepulse",
)
DEFAULT_REPORT_PATH = PROJECT_ROOT / os.getenv(
    "LINEPULSE_REPORT_PATH",
    "reports/data_validation_report.json",
)

