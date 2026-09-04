"""End-to-end LinePulse workflow package."""

from __future__ import annotations

__all__ = [
    "VerticalSliceResult",
    "VerticalSliceWorkflow",
]


def __getattr__(name: str):
    if name in __all__:
        from linepulse.workflows.vertical_slice import (
            VerticalSliceResult,
            VerticalSliceWorkflow,
        )

        exports = {
            "VerticalSliceResult": VerticalSliceResult,
            "VerticalSliceWorkflow": VerticalSliceWorkflow,
        }
        return exports[name]

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
