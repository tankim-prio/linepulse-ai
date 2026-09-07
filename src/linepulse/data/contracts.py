"""Machine-readable data and feature contracts for LinePulse AI."""

from __future__ import annotations


# Each item is: child columns, parent dataset, parent columns.
FOREIGN_KEYS: dict[str, tuple[tuple[tuple[str, ...], str, tuple[str, ...]], ...]] = {
    "production_lines": ((('factory_id',), "factories", ('factory_id',)),),
    "styles": ((('factory_id',), "factories", ('factory_id',)),),
    "production_orders": (
        (("factory_id",), "factories", ("factory_id",)),
        (("style_id",), "styles", ("style_id",)),
    ),
    "line_allocations": (
        (("factory_id",), "factories", ("factory_id",)),
        (("order_id",), "production_orders", ("order_id",)),
        (("line_id",), "production_lines", ("line_id",)),
    ),
    "hourly_production": (
        (("factory_id",), "factories", ("factory_id",)),
        (("allocation_id",), "line_allocations", ("allocation_id",)),
        (("line_id",), "production_lines", ("line_id",)),
        (("order_id",), "production_orders", ("order_id",)),
        (("style_id",), "styles", ("style_id",)),
    ),
    "attendance_summaries": (
        (("factory_id",), "factories", ("factory_id",)),
        (("line_id",), "production_lines", ("line_id",)),
    ),
    "machine_downtime": (
        (("factory_id",), "factories", ("factory_id",)),
        (("line_id",), "production_lines", ("line_id",)),
        (("order_id",), "production_orders", ("order_id",)),
    ),
    "quality_defects": (
        (("factory_id",), "factories", ("factory_id",)),
        (("line_id",), "production_lines", ("line_id",)),
        (("order_id",), "production_orders", ("order_id",)),
        (("style_id",), "styles", ("style_id",)),
    ),
    "material_status": (
        (("factory_id",), "factories", ("factory_id",)),
        (("order_id",), "production_orders", ("order_id",)),
    ),
    "supervisor_notes": (
        (("factory_id",), "factories", ("factory_id",)),
        (("line_id",), "production_lines", ("line_id",)),
        (("order_id",), "production_orders", ("order_id",)),
    ),
    "corrective_actions": (
        (("factory_id",), "factories", ("factory_id",)),
        (("line_id",), "production_lines", ("line_id",)),
        (("order_id",), "production_orders", ("order_id",)),
        (("note_id",), "supervisor_notes", ("note_id",)),
    ),
    "action_outcomes": (
        (("factory_id",), "factories", ("factory_id",)),
        (("line_id",), "production_lines", ("line_id",)),
        (("action_id",), "corrective_actions", ("action_id",)),
    ),
    "daily_line_ml": (
        (("factory_id",), "factories", ("factory_id",)),
        (("line_id",), "production_lines", ("line_id",)),
        (("order_id",), "production_orders", ("order_id",)),
        (("style_id",), "styles", ("style_id",)),
    ),
    "order_outcomes": (
        (("order_id",), "production_orders", ("order_id",)),
        (("factory_id",), "factories", ("factory_id",)),
        (("line_id",), "production_lines", ("line_id",)),
        (("style_id",), "styles", ("style_id",)),
    ),
    "document_sections": (
        (("document_id",), "documents", ("document_id",)),
    ),
    "rag_gold_questions": (
        (("relevant_document_id",), "documents", ("document_id",)),
        (("relevant_section_id",), "document_sections", ("section_id",)),
    ),
    "rag_answerability_calibration": (
        (("relevant_document_id",), "documents", ("document_id",)),
        (("relevant_section_id",), "document_sections", ("section_id",)),
    ),
    "agent_eval_cases": (
        (("snapshot_id",), "daily_line_ml", ("snapshot_id",)),
    ),
}


DAILY_RISK_FEATURES = (
    "day_in_order",
    "complexity_band",
    "sam_minutes",
    "planned_workers",
    "present_workers",
    "attendance_rate",
    "target_to_snapshot",
    "output_to_snapshot",
    "progress_ratio",
    "downtime_to_snapshot",
    "inspected_to_snapshot",
    "defects_to_snapshot",
    "defect_rate_to_snapshot",
    "material_status",
    "changeover_minutes",
    "root_cause_signal",
    "rolling_3d_efficiency",
    "rolling_3d_downtime",
    "previous_day_miss",
)

DAILY_RISK_TARGET = "label_daily_target_missed"

FORBIDDEN_DAILY_RISK_FEATURES = {
    "daily_target",
    "daily_actual_output",
    "label_daily_target_missed",
    "label_high_defect",
    "actual_completion_at",
}

ORDER_DELAY_FEATURES = (
    "order_quantity",
    "sam_minutes",
    "complexity_band",
    "planned_duration_days",
    "schedule_slack_days",
    "line_prior_14d_efficiency",
)

ORDER_DELAY_TARGET = "label_order_delayed"

FORBIDDEN_ORDER_DELAY_FEATURES = {
    "actual_completion_date",
    "actual_duration_days",
    "label_order_delayed",
}

ALLOWED_SPLITS = {"train", "validation", "test"}
REQUIRED_NOTE_LANGUAGES = {"bn", "en", "mixed"}
