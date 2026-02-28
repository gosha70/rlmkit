---
name: Evaluation Regression
about: Report a regression in retrieval or answer quality metrics
title: '[EVAL] '
labels: evaluation, regression
assignees: ''
---

## Summary

Brief description of the regression observed.

## Metrics Affected

- **Metric**: [e.g., recall@5, precision@10, answer quality score]
- **Previous value**: [e.g., 0.85]
- **Current value**: [e.g., 0.72]
- **Dataset**: [e.g., `eval/datasets/qa_v2.jsonl`]

## Suspected Cause

What change do you think caused the regression? Link to a commit or PR if known.

## Steps to Reproduce

```bash
# Commands to reproduce the eval run
```

## Evaluation Output

```
Paste eval harness output here
```

## Environment

- **Embeddings model**: [e.g., text-embedding-3-small]
- **LLM model**: [e.g., gpt-4o]
- **Chunking config**: [e.g., strategy=recursive, size=512, overlap=64]

## Additional Context

Any other context about the regression.
