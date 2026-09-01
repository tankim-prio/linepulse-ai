# LinePulse AI implementation roadmap

Each phase must have a runnable deliverable and passing checks before the next
phase begins.

1. **Repository and data validation** — project structure, datasets, contracts,
   validator, tests, and Windows scripts.
2. **Exploratory analysis** — distributions, missingness, class balance,
   relationships, and leakage review.
3. **Local data layer** — typed ingestion and analytical views using DuckDB.
4. **Feature pipeline** — deterministic point-in-time features for daily risk.
5. **Daily target-risk baseline** — dummy baseline and CPU scikit-learn model.
6. **Model evaluation** — PR-AUC, ROC-AUC, recall, precision, calibration, and
   threshold selection by business cost.
7. **Order-delay model** — order-start features and delivery-risk prediction.
8. **Multilingual note classification** — root-cause and urgency baselines.
9. **RAG ingestion** — bilingual document loading, chunking, version metadata,
   and a CPU-compatible vector index.
10. **RAG evaluation** — retrieval metrics, citations, answerability, and
    abstention tests.
11. **FastAPI service** — health, data, prediction, evidence, and RAG endpoints.
12. **LangGraph workflow** — evidence collection, risk routing, recommendation,
    and deterministic terminal states.
13. **Human approval** — approval records and write-blocking policy.
14. **MCP tools** — typed, permission-checked, idempotent read/write tools.
15. **Dashboard** — line status, risks, evidence, explanations, and approvals.
16. **Security tests** — prompt injection, authorization, replay, and tool abuse.
17. **Docker packaging** — CPU and memory-limited local services.
18. **Observability** — structured logs, latency, errors, model and RAG metrics.
19. **Documentation** — architecture, API reference, model cards, and decisions.
20. **Portfolio demonstration** — reproducible demo, screenshots, and interview
    walkthrough.
21. **Optional real-data pilot** — only authorized, anonymized operational data
    and approved documents.

