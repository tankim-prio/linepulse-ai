# Phase 4 — Multilingual NLP + RAG Exit Summary

## Status

Phase 4 completed after Step 11F-B validation. All required Phase-4 exit checks passed before commit.

The implementation keeps retrieval, answerability, evidence packaging,
and future generation as separate responsibilities.

## Frozen RAG v2 answerability policy

- Feature: `dense_top1_cosine_similarity`
- Operator: `>=`
- Threshold: `0.84151649475097656`
- Policy fingerprint:
  `59e9ab2bdf28abf157b3981a768ce2a646724dca294aff7525e1131d8e8f189f`
- Frozen pre-audit policy commit:
  `1dfefda81548f54ba018fa933971605112b0d997`
- Sealed audit record commit:
  `c628391e094c3d0f902fa2020359631864c5aabd`

The sealed audit was not used to refit the threshold, retrain a model,
select features, select candidates, or otherwise modify the v2 policy.

## Sealed synthetic audit

48 audit questions:

- 32 supported
- 16 insufficient-evidence
- 16 English
- 16 Bangla
- 16 mixed-language

Overall:

- TP/FN/TN/FP: 28/4/14/2
- Accuracy: 0.875000
- Balanced accuracy: 0.875000
- Precision: 0.933333
- Recall: 0.875000
- Specificity: 0.875000
- F1: 0.903226
- MCC: 0.730297
- Dense-top1 ROC AUC: 0.968750

Language-stratified balanced accuracy:

- English: 0.716667
- Bangla: 1.000000
- Mixed: 0.875000

The English subgroup result is retained as a limitation. It is not used
to retune v2. Any improvement based on these audit observations requires
a future policy version and new development data.

This audit is synthetic and development-isolated. It is not independent
real-factory validation.

## Final online RAG boundary

The Phase-4 integration performs:

1. one hybrid retrieval call per query;
2. one dense query embedding inside that retrieval path;
3. deterministic dense + lexical retrieval with RRF evidence ranking;
4. reuse of returned dense scores for frozen answerability;
5. no second query embedding for the answerability gate;
6. top-k evidence exposure only when the frozen gate returns supported;
7. chunk IDs as citation identifiers;
8. retrieved document text marked as untrusted;
9. no autonomous tool execution from retrieved document content;
10. no free-form LLM dependency in the core RAG path.

For insufficient evidence, generation context is empty.

## Complexity

For the current exact in-memory corpus:

- chunks `N = 16`
- dense dimension `d = 384`

Dense exact scoring is approximately `O(N*d)` per embedded query.
Lexical retrieval uses the existing sparse TF-IDF representation.
Deterministic ranking adds approximately `O(N log N)` work.

The integration intentionally reuses dense scores from the one hybrid
search call rather than embedding a query twice.

At the present corpus size, an ANN/vector index would add operational
complexity without a useful latency benefit. A database vector index
should be introduced only when corpus scale justifies it.

## Security and grounding boundary

Retrieved document text is untrusted data.

Documents cannot redefine system instructions, authorize tool calls,
or bypass approval controls. The Phase-4 service only creates
answerability-gated evidence context.

Write approval, exact tool/argument binding, idempotency, and audit
execution remain Phase-5 responsibilities.

## Phase 4 exit criteria

The phase can be marked complete only after Step 11F-B verifies:

- targeted grounded-context tests pass;
- real HybridRetriever integration smoke test passes;
- all RAG tests pass;
- full project tests pass;
- dependency integrity passes;
- frozen v2 policy module and manifest remain unchanged;
- sealed audit artifacts remain unchanged;
- exact intended files are committed;
- local and remote Git heads match;
- final working tree is clean.
