# LinePulse Synthetic Dataset Package

## Authenticity notice

This package is **100% synthetic**. It contains no real factory, buyer, order,
employee, or production information. It is intended for portfolio development,
software testing, ML/RAG evaluation, and demonstration only.

## Coverage

- Period: 2025-01-04 to 2025-06-30
- Factory: 1 synthetic factory
- Production lines: 8
- Completed orders: 137
- Hourly production rows: 11893
- Point-in-time ML snapshots: 1224
- Multilingual supervisor notes: 1224
- RAG evaluation cases: 60
- Agent evaluation cases: 48
- Security evaluation cases: 40

## Directory structure

```text
relational/  Normalized factory operational tables
modeling/    Point-in-time ML tables, order outcomes, and time splits
knowledge/   Bilingual synthetic SOPs and document catalog
evaluation/  RAG, LangGraph, and security test cases
metadata/    Manifest, data dictionary, assumptions, and validation report
```

## Recommended starting files

1. `relational/hourly_production.csv`
2. `modeling/daily_line_ml.csv`
3. `metadata/data_dictionary.csv`
4. `metadata/validation_report.json`

## ML leakage warning

For daily-risk training, use feature columns available at `snapshot_at` and
predict `label_daily_target_missed`. Do not use `daily_actual_output`,
`daily_target`, or any label column as a model feature.

## RAG use

Index the Markdown files in `knowledge/`. Use
`evaluation/rag_gold_questions.csv` to measure retrieval and grounded answers.

## Production warning

Do not claim real business impact from this dataset. Replace it with properly
authorized, anonymized real data and approved factory documents before any
production pilot.
