---
layout: default
title: Examples
---

## Examples

**Who this is for:** Users who want runnable snippets before deeper tuning.

**When to use it:** To quickly verify install and behavior in binary and multiclass settings.

### Binary example expected output

- `predict(X)` returns a label list with one value per sample.
- `predict_proba(X)` returns rows like `[1.0, 0.0]` or `[0.0, 1.0]`.
- `score(X, y)` returns a float in `[0.0, 1.0]`.

### Multiclass example expected output

- `predict(X)` returns one of the known classes per row.
- `predict_proba(X)` returns per-class normalized probabilities summing to `1.0` per row.

For full runnable snippets, see [Getting Started]({{ '/getting-started.html' | relative_url }}) and [Multiclass Guide]({{ '/multiclass-guide.html' | relative_url }}).
