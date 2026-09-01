"""Command-line entry point for LinePulse Phase 1 checks."""

from __future__ import annotations

import argparse
from pathlib import Path

from linepulse.data.validator import DataValidator
from linepulse.settings import DEFAULT_DATA_DIR, DEFAULT_REPORT_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the LinePulse synthetic dataset package.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Dataset root (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help=f"JSON report path (default: {DEFAULT_REPORT_PATH})",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = DataValidator(args.data_dir).run()
    report.write_json(args.report)

    for check in report.checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.name}: {check.detail}")

    print()
    print(
        "Summary: "
        f"datasets={report.dataset_count}, "
        f"rows={report.row_count}, "
        f"checks={len(report.checks)}, "
        f"failures={report.failed_count}"
    )
    print(f"Report: {args.report.resolve()}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

