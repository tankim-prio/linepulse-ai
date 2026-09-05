"""LinePulse AI Streamlit dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from linepulse.dashboard.api_client import (
    DEFAULT_API_URL,
    LinePulseApiClient,
    LinePulseApiError,
)


RULE_VERSION = "progress-gap-v1"


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
            "The dashboard reads operational "
            "data through FastAPI only."
        )

        st.divider()

        st.caption(
            f"Rule version: {RULE_VERSION}"
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

    line_ids = [
        item["line_id"]
        for item in production_lines
    ]

    analytics_line_options = [
        "All active lines",
        *line_ids,
    ]

    selected_analytics_line = st.selectbox(
        "Analytics production line",
        options=analytics_line_options,
    )

    analytics_line_id = (
        None
        if selected_analytics_line
        == "All active lines"
        else selected_analytics_line
    )

    try:
        trend = client.risk_trend(
            factory_id=selected_factory,
            line_id=analytics_line_id,
            rule_version=RULE_VERSION,
        )

        factors = client.factor_summaries(
            factory_id=selected_factory,
            line_id=analytics_line_id,
            rule_version=RULE_VERSION,
        )

    except LinePulseApiError as exc:
        st.error(
            str(exc)
        )
        return

    st.subheader(
        "Rule-risk trend"
    )

    st.caption(
        "Average and maximum persisted "
        f"{RULE_VERSION} rule-risk scores "
        "by snapshot."
    )

    if trend:
        trend_frame = pd.DataFrame(
            trend
        )

        trend_frame[
            "snapshot_at"
        ] = pd.to_datetime(
            trend_frame[
                "snapshot_at"
            ]
        )

        trend_frame = (
            trend_frame
            .sort_values(
                "snapshot_at"
            )
            .set_index(
                "snapshot_at"
            )
        )

        st.line_chart(
            trend_frame[
                [
                    "average_risk_score",
                    "maximum_risk_score",
                ]
            ]
        )

        trend_display = (
            trend_frame
            .reset_index()
            [
                [
                    "snapshot_at",
                    "event_count",
                    "average_risk_score",
                    "maximum_risk_score",
                    "rule_version",
                ]
            ]
        )

        with st.expander(
            "View risk-trend data"
        ):
            st.dataframe(
                trend_display,
                use_container_width=True,
                hide_index=True,
            )

    else:
        st.info(
            "No trend data matched "
            "the selected filters."
        )

    st.subheader(
        "Observed risk factors"
    )

    st.caption(
        "Counts show how often each factor "
        "appears in persisted rule-risk events. "
        "They are occurrence counts, not probabilities."
    )

    if factors:
        factor_frame = pd.DataFrame(
            factors
        )

        factor_frame = (
            factor_frame
            .sort_values(
                [
                    "occurrence_count",
                    "factor",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
        )

        factor_chart = (
            factor_frame[
                [
                    "factor",
                    "occurrence_count",
                ]
            ]
            .set_index(
                "factor"
            )
        )

        st.bar_chart(
            factor_chart
        )

        st.dataframe(
            factor_frame[
                [
                    "factor",
                    "occurrence_count",
                    "rule_version",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Total observed factor occurrences: "
            f"{int(factor_frame['occurrence_count'].sum())}"
        )

    else:
        st.info(
            "No observed factors matched "
            "the selected filters."
        )

    st.subheader(
        "Recent risk events"
    )

    event_line_options = [
        "All active lines",
        *line_ids,
    ]

    selected_event_line = st.selectbox(
        "Recent-event production line",
        options=event_line_options,
    )

    event_line_id = (
        None
        if selected_event_line
        == "All active lines"
        else selected_event_line
    )

    try:
        recent_events = client.risk_events(
            limit=10,
            factory_id=selected_factory,
            line_id=event_line_id,
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
        "results are synthetic. "
        "No High/Medium/Low risk categories "
        "are inferred by this dashboard."
    )


if __name__ == "__main__":
    main()
