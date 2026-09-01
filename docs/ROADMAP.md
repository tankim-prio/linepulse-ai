# LinePulse AI capability roadmap

Each capability must have a runnable deliverable and passing checks before
dependent work begins.

## Implemented

- **Repository and data validation** — datasets, contracts, validator, tests,
  and Windows commands.
- **Exploratory analysis** — distributions, missingness, class balance,
  line-level risk rates, and correlations.
- **Daily target-risk baseline** — dummy benchmark, logistic regression,
  chronological evaluation, threshold selection, and model card.

## Planned

1. **Local analytical data layer** — typed ingestion and views using DuckDB.
2. **Reusable feature pipeline** — deterministic point-in-time feature builds.
3. **Order-delay model** — order-start features and delivery-risk prediction.
4. **Multilingual note classification** — root-cause and urgency baselines.
5. **RAG ingestion** — bilingual document loading, chunking, version metadata,
   and a CPU-compatible vector index.
6. **RAG evaluation** — retrieval metrics, citations, answerability, and
   abstention tests.
7. **FastAPI service** — health, data, prediction, evidence, and RAG endpoints.
8. **LangGraph workflow** — evidence collection, risk routing, recommendation,
   and deterministic terminal states.
9. **Human approval** — approval records and write-blocking policy.
10. **MCP tools** — typed, permission-checked, idempotent read/write tools.
11. **Dashboard** — line status, risks, evidence, explanations, and approvals.
12. **Security tests** — prompt injection, authorization, replay, and tool abuse.
13. **Docker packaging** — CPU and memory-limited local services.
14. **Observability** — structured logs, latency, errors, model and RAG metrics.
15. **Portfolio demonstration** — reproducible demo and interview walkthrough.
16. **Optional real-data pilot** — only authorized, anonymized operational data
    and approved documents.

