---
layout: default
title: API Reference
---

## API Reference

**Who this is for:** Users configuring training behavior and integrating with sklearn-like workflows.

**When to use it:** When you need parameter and method-level guidance.

### `GPClassifier` constructor

- `num_models` (default `30`): population size.
- `generations` (default `100`): evolution steps.
- `crossover_rate` (default `0.4`): crossover fraction.
- `mutation_rate` (default `0.3`): mutation fraction.
- `elitist_rate` (default `0.2`): elite carryover fraction.
- `max_depth` (default `6`): maximum tree depth.
- `tournament_size` (default `5`): tournament draw size.
- `selection_method` (default `"pareto_tournament"`): parent selection strategy.
- `fitness_method` (default `"f1_score"`): fitness objective.
- `random_state` (default `None`): reproducibility seed.
- `initial_population` (default `None`): optional starter models.
- `show_training_curve` (default `False`): print generation-by-generation best fitness.

### Main methods

- `fit(X, y)`: train model(s) on binary or multiclass data.
- `predict(X)`: return predicted labels.
- `predict_proba(X)`: return probability rows per sample.
- `score(X, y)`: return mean classification accuracy.
- `view_model(n_models=1)`: return expression model output.
- `view_model_tree(n_models=1)`: return tree-like model output.
