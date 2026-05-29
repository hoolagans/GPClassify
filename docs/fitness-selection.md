---
layout: default
title: Fitness & Selection
---

## Fitness & Selection

**Who this is for:** Users tuning model quality versus complexity.

**When to use it:** When selecting optimization targets and parent-selection behavior.

### Fitness methods

- `"f1_score"` (default): optimize F1 score with inversion symmetry.
- `"accuracy"`: optimize classification agreement with inversion symmetry.
- `"pearson_r2"`: optimize squared Pearson correlation.

### Selection methods

- `"tournament"`: fitness-driven tournament parent selection.
- `"pareto_tournament"`: uses a non-dominated front during tournament selection to:
  - maximize fitness
  - minimize tree complexity
