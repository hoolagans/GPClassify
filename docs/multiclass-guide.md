---
layout: default
title: Multiclass Guide
---

## Multiclass Guide

**Who this is for:** Users solving >2 class problems.

**When to use it:** When labels contain three or more classes and you want one-vs-rest training.

GPClassify trains multiclass problems with a one-vs-rest strategy. Use the same fit/predict API.

```python
from gpclassify import GPClassifier

X = [
    [9.0, 1.0, 1.0],
    [1.0, 9.0, 1.0],
    [1.0, 1.0, 9.0],
    [8.0, 2.0, 1.0],
    [1.0, 8.0, 2.0],
    [2.0, 1.0, 8.0],
]
y = [0, 1, 2, 0, 1, 2]

clf = GPClassifier(num_models=20, generations=20, random_state=7)
clf.fit(X, y)

pred = clf.predict(X)
proba = clf.predict_proba(X)
```

Interpretation tips:
- `predict` returns the class with the highest class score.
- `predict_proba` returns normalized per-class probabilities.
- Model inspection pages include class labels for one-vs-rest models.
