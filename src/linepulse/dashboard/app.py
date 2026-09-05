"""LinePulse AI Streamlit dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from linepulse.dashboard.api_client import (
    DEFAULT_API_URL,
    LinePulseApiClient,
    LinePulseApiError,
)


def _risk_value(
    value: float | None,
) -> str:
    if value is None:
        return "N/A"

    return f"{value:.4f}"


def main() -> None:
    st.set_page_config(
        page_title="LinePulse AI",
        page_icon="📊",
        layout="wide",
    )

    st.title(
        "LinePulse AI"
    )

    st.caption(
        "Garment production risk intelligence "
        "dashboard · synthetic demonstration data"
    )

    with st.sidebar:
        st.header(
            "Connection"
        )

        api_url = st.text_input(
            "FastAPI URL",
            value=DEFAULT_API_URL,
        )

        st.caption(
            "The dashboard reads data through "
            "FastAPI only."
        )

    client = LinePulseApiClient(
        api_url
    )

    try:
        health = client.database_health()
        overview = client.analytics_overview()
        factories = client.factories()

    except LinePulseApiError as exc:
        st.error(
            "Could not load LinePulse API data."
        )

        st.code(
            str(exc)
        )

        st.info(
            "Start FastAPI with: "
            "python -m uvicorn "
            "linepulse.main:app --reload"
        )

        return

    if health.get("status") != "ok":
        st.warning(
            "FastAPI responded, but the "
            "database health status is not OK."
        )

    st.subheader(
        "Platform overview"
    )

    metric_columns = st.columns(
        5
    )

    metric_columns[0].metric(
        "Factories",
        overview[
            "factory_count"
        ],
    )

    metric_columns[1].metric(
        "Production lines",
        overview[
            "production_line_count"
        ],
    )

    metric_columns[2].metric(
        "Risk events",
        overview[
            "risk_event_count"
        ],
    )

    metric_columns[3].metric(
        "Average rule risk",
        _risk_value(
            overview[
                "average_risk_score"
            ]
        ),
    )

    metric_columns[4].metric(
        "Maximum rule risk",
        _risk_value(
            overview[
                "maximum_risk_score"
            ]
        ),
    )

    st.caption(
        "Latest persisted snapshot: "
        f"{overview['latest_snapshot_at']}"
    )

    factory_options = {
        item["factory_id"]:
            item["factory_name"]
        for item in factories
    }

    factory_ids = list(
        factory_options
    )

    selected_factory = (
        factory_ids[0]
        if factory_ids
        else None
    )

    if factory_ids:
        selected_factory = st.selectbox(
            "Factory",
            options=factory_ids,
            format_func=lambda factory_id: (
                f"{factory_id} — "
                f"{factory_options[factory_id]}"
            ),
        )

    try:
        line_summaries = (
            client.analytics_lines(
                factory_id=selected_factory
            )
        )

        production_lines = (
            client.production_lines(
                factory_id=selected_factory,
                active=True,
            )
        )

    except LinePulseApiError as exc:
        st.error(
            str(exc)
        )
        return

    st.subheader(
        "Production-line risk summary"
    )

    if line_summaries:
        line_frame = pd.DataFrame(
            line_summaries
        )

        display_columns = [
            "line_id",
            "line_name",
            "specialization",
            "event_count",
            "average_risk_score",
            "maximum_risk_score",
            "latest_snapshot_at",
        ]

        st.dataframe(
            line_frame[
                display_columns
            ],
            use_container_width=True,
            hide_index=True,
        )

        chart_frame = (
            line_frame[
                [
                    "line_id",
                    "average_risk_score",
                ]
            ]
            .set_index(
                "line_id"
            )
        )

        st.bar_chart(
            chart_frame
        )

    else:
        st.info(
            "No line analytics are available "
            "for this factory."
        )

    st.subheader(
        "Recent risk events"
    )

    line_ids = [
        item["line_id"]
        for item in production_lines
    ]

    line_filter_options = [
        "All active lines",
        *line_ids,
    ]

    selected_line = st.selectbox(
        "Production line",
        options=line_filter_options,
    )

    line_id = (
        None
        if selected_line == "All active lines"
        else selected_line
    )

    try:
        recent_events = client.risk_events(
            limit=10,
            factory_id=selected_factory,
            line_id=line_id,
        )

    except LinePulseApiError as exc:
        st.error(
            str(exc)
        )
        return

    if recent_events:
        event_frame = pd.DataFrame(
            recent_events
        )

        st.dataframe(
            event_frame[
                [
                    "snapshot_at",
                    "line_id",
                    "order_id",
                    "risk_score",
                    "factors",
                    "rule_version",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info(
            "No risk events matched "
            "the selected filters."
        )

    st.divider()

    st.caption(
        "Decision-support demonstration only. "
        "Current persisted data and rule-risk "
        "results are synthetic."
    )


if __name__ == "__main__":
    main()
