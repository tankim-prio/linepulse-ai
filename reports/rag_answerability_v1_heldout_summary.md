# RAG Answerability v1 ? Frozen Held-Out Evaluation

## Evaluation identity

- Frozen commit: `f1d3bbc4b2d4a6e9aa076fd0d91a83d5779d5ad4`
- Policy ID: `e5-top1-midpoint-v1`
- Policy fingerprint: `8b3fb0ebde2632490790a44cc92cc44a1c1f0c86603cd653c38e0d3190034dea`
- Threshold: `0.8190949857234955`
- Embedding model: `intfloat/multilingual-e5-small`
- Embedding revision: `614241f622f53c4eeff9890bdc4f31cfecc418b3`
- Gold dataset SHA256: `c0b2bbd261beeb99b1ec47b86a78b5496742f4b508f92a09fc7a54760b86afe9`
- Held-out cases: 37
- Supported cases: 25
- Insufficient-evidence cases: 12

## Answerability results

| Metric | Result |
| --- | ---: |
| TP | 23 |
| FN | 2 |
| TN | 10 |
| FP | 2 |
| Accuracy | 0.891892 |
| Balanced accuracy | 0.876667 |
| Precision | 0.920000 |
| Recall / sensitivity | 0.920000 |
| Specificity | 0.833333 |
| F1 | 0.920000 |
| MCC | 0.753333 |

## Supported retrieval

| Metric | Result |
| --- | ---: |
| R@1 | 0.880000 |
| R@3 | 1.000000 |
| MRR | 0.926667 |
| Top-3 misses | 0 |

## Misclassified cases

| query_id | language | expected_status | predicted_status | top1_score | margin_to_threshold | top1_chunk_id | expected_section_rank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RAG-007 | en | supported | insufficient_evidence | 0.813496172428 | -0.00559881329536 | CHK-eaff5443d7c82fd740f1aff7 | 1 |
| RAG-019 | en | supported | insufficient_evidence | 0.806248009205 | -0.0128469765186 | CHK-07357538803de3f2063c8ea2 | 1 |
| RAG-053 | bn | insufficient_evidence | supported | 0.821983158588 | 0.00288817286491 | CHK-eaff5443d7c82fd740f1aff7 |  |
| RAG-054 | mixed | insufficient_evidence | supported | 0.830691277981 | 0.0115962922573 | CHK-eaff5443d7c82fd740f1aff7 |  |

## Interpretation

The v1 top-1 cosine threshold is a useful baseline but is not a
complete answerability model. Held-out supported and insufficient-evidence
score distributions overlap. Therefore a single cosine threshold cannot
fully separate the two classes.

Retrieval remains strong: every supported held-out question retrieved its
expected section within the top three results.

The next development cycle must not tune this v1 policy using these held-out
labels or scores. A v2 policy requires fresh development/calibration cases
and a separate untouched audit set.

## Methodology disclosure

The threshold was selected from the calibration dataset and frozen in Git
before this formal held-out evaluation. These 37 held-out cases were not
used to optimize the v1 threshold.

Before the calibration protocol was finalized, held-out negative score
distributions had been inspected descriptively during earlier development.
Accordingly, this dataset should not be described as a perfectly pristine
never-inspected test set.

No v1 threshold modification was made after formal evaluation.
