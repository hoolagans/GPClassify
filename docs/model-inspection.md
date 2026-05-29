---
layout: default
title: Model Inspection
---

## Model Inspection

**Who this is for:** Users who need interpretable model output.

**When to use it:** After fitting, to inspect evolved expressions and tree structures.

Use expression and tree views:

```python
expr = clf.view_model()            # one model as a readable expression
top3_expr = clf.view_model(3)      # top 3 models as expressions

tree = clf.view_model_tree()       # one model in tree-like format
top2_trees = clf.view_model_tree(2)
```

Guidance:
- Use `view_model` for compact expression summaries.
- Use `view_model_tree` for structural walkthroughs.
- In multiclass mode, outputs are labeled by class and model index.
