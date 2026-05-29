---
layout: default
title: Getting Started
---

## Getting Started (Binary Classification)

**Who this is for:** Users trying GPClassify for the first time.

**When to use it:** To train and evaluate a first binary classifier with minimal setup.

```python
from gpclassify import GPClassifier

X = [
    [4.0, 1.0],
    [5.0, 2.0],
    [1.0, 3.0],
    [2.0, 5.0],
]
y = [1 if row[0] > row[1] else 0 for row in X]

clf = GPClassifier(
    num_models=40,
    generations=40,
    max_depth=6,
    selection_method="pareto_tournament",
    fitness_method="pearson_r2",
    random_state=42,
)
clf.fit(X, y)

print(clf.predict(X))
print(clf.predict_proba(X))
print(clf.score(X, y))
```

Expected behavior on this toy dataset:
- Predictions should match labels exactly.
- Probabilities should return one row per sample.
- `score` should report a value near `1.0`.
