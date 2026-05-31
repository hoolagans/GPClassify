# Fitness Method Comparison Report

## Objective

Compare GPClassify fitness objectives:

- `accuracy`
- `f1_score`
- `pearson_r2` (correlation objective)

across real and synthetic datasets, with special emphasis on **imbalanced** and **high/complete overlap** class settings.  
In addition to accuracy and macro-F1, this report includes **per-class accuracy** (class recall) and mean per-class accuracy.

## Experimental Setup

- Classifier: `GPClassifier`
- Fitness method varied: `accuracy`, `f1_score`, `pearson_r2`
- Common settings per run:
  - `num_models=20`
  - `generations=12`
  - `max_depth=6`
  - `selection_method="pareto_tournament"`
- Seeds: 8 (`0..7`)
- Train/test split: 70/30 stratified by class

### Dataset groups

**Previous set (9 datasets):**

- Real: breast cancer (binary), iris (3-class), wine (3-class), digits (10-class)
- Synthetic: binary balanced, binary imbalanced, 3-class balanced, 3-class imbalanced, 5-class imbalanced

**New overlap-focused set (4 datasets):**

- `syn_bin_imbalanced_high_overlap`
- `syn_3c_imbalanced_high_overlap`
- `syn_bin_imbalanced_complete_overlap` (features generated independent of labels)
- `syn_4c_imbalanced_complete_overlap` (features generated independent of labels)

## Metrics

- **Accuracy**
- **Macro-F1**
- **Per-class accuracy** (class-wise recall from confusion matrix)
- **Mean per-class accuracy** (average of class recalls)

## Output Artifacts

- Run-level metrics: `run_level_metrics.csv`
- Per-class metrics: `per_class_metrics.csv`
- Dataset summary: `dataset_summary.csv`, `summary_by_dataset.json`
- Overlap-only per-class summary: `overlap_per_class_summary.csv`

## Figures (Box-and-Whisker)

1. Overall accuracy by method: `figures/boxplot_accuracy_by_method.png`
2. Overall macro-F1 by method: `figures/boxplot_macro_f1_by_method.png`
3. Overall mean per-class accuracy by method: `figures/boxplot_mean_per_class_accuracy_by_method.png`
4. Overlap-focused macro-F1 and mean per-class accuracy by method: `figures/boxplot_overlap_datasets.png`

## Results Summary

### 1) Combined overall means (all 13 datasets)

- `accuracy` fitness:
  - Accuracy: **0.7078**
  - Macro-F1: **0.5064**
  - Mean per-class accuracy: **0.5368**
- `f1_score` fitness:
  - Accuracy: **0.7033**
  - Macro-F1: **0.5536**
  - Mean per-class accuracy: **0.5804**
- `pearson_r2` fitness:
  - Accuracy: **0.6928**
  - Macro-F1: **0.5779**
  - Mean per-class accuracy: **0.6013**

Interpretation: on the combined benchmark, `pearson_r2` is strongest on class-balance-sensitive metrics (macro-F1 and mean per-class accuracy), while `accuracy` is narrowly highest on plain accuracy.

### 2) Winner counts by dataset (13 datasets total)

- Best **Accuracy**:
  - `f1_score`: 6
  - `pearson_r2`: 4
  - `accuracy`: 3
- Best **Macro-F1**:
  - `pearson_r2`: 7
  - `f1_score`: 4
  - `accuracy`: 2
- Best **Mean per-class accuracy**:
  - `pearson_r2`: 7
  - `f1_score`: 5
  - `accuracy`: 1

Interpretation: `pearson_r2` is the most consistent winner for balanced class treatment metrics.

### 3) Previous dataset set (9 datasets)

For the original benchmark set only, winners are:

- Accuracy: `pearson_r2` 4, `f1_score` 4, `accuracy` 1
- Macro-F1: `pearson_r2` 4, `f1_score` 4, `accuracy` 1
- Mean per-class accuracy: `pearson_r2` 4, `f1_score` 4, `accuracy` 1

Interpretation: the previous conclusion remains: `pearson_r2` and `f1_score` are both generally stronger than pure `accuracy`, with near parity between `pearson_r2` and `f1_score` on this subset.

### 4) New overlap-focused datasets (4 datasets)

#### High-overlap imbalanced 3-class (`syn_3c_imbalanced_high_overlap`)

- Accuracy is highest with `f1_score` (0.672), but macro-F1 and mean per-class accuracy are highest with `pearson_r2` (0.384 and 0.419).
- Minority class (label 2) per-class accuracy is very low for all methods:
  - `accuracy`: 0.059
  - `f1_score`: 0.015
  - `pearson_r2`: 0.041

#### High-overlap imbalanced binary (`syn_bin_imbalanced_high_overlap`)

- `accuracy` objective gives high overall accuracy (0.838) but collapses minority-class detection (minority per-class accuracy = 0.000).
- `f1_score` improves minority class substantially (0.517), with `pearson_r2` also improving over `accuracy` (0.335).
- Macro-F1 is best with `pearson_r2` (0.488).

#### Complete-overlap imbalanced binary (`syn_bin_imbalanced_complete_overlap`)

- `accuracy` objective again yields very high overall accuracy (0.903) by majority-class prediction, with minority-class per-class accuracy = 0.000.
- `f1_score` and `pearson_r2` recover minority signal:
  - minority per-class accuracy: 0.412 (`f1_score`) and 0.324 (`pearson_r2`)
- Best macro-F1 is `pearson_r2` (0.493).

#### Complete-overlap imbalanced 4-class (`syn_4c_imbalanced_complete_overlap`)

- `accuracy` and `f1_score` maximize overall accuracy (~0.71) by favoring majority classes.
- `pearson_r2` sacrifices overall accuracy (0.369) but gives better coverage of rare classes:
  - minority class (label 3) per-class accuracy: 0.179 (`pearson_r2`) vs 0.021 (`accuracy`) and 0.000 (`f1_score`)
- Mean per-class accuracy is best with `pearson_r2` (0.255, marginally above others).

## Discussion

1. **Accuracy objective is brittle under imbalance and overlap.**  
   It can report strong headline accuracy while ignoring minority classes, especially in overlap-heavy settings.

2. **`pearson_r2` is most robust for class-sensitive outcomes.**  
   It wins most often on macro-F1 and mean per-class accuracy and is especially useful when classes overlap heavily.

3. **`f1_score` remains a strong compromise.**  
   It often improves minority class behavior over `accuracy` while keeping competitive overall accuracy.

4. **Per-class accuracy changes conclusions materially.**  
   Without per-class analysis, several overlap scenarios would misleadingly favor the `accuracy` objective.

## Practical Recommendation

- For imbalanced and/or overlap-prone problems, prefer **`pearson_r2`** when minority-class coverage and balanced performance matter most.
- Use **`f1_score`** when you want a simpler metric with strong minority sensitivity and stable overall performance.
- Avoid relying on pure **`accuracy`** as the primary optimization objective for heavily imbalanced overlap cases.

## Reproducibility

Run:

```bash
PYTHONPATH=/tmp/workspace/hoolagans/GPClassify python /tmp/workspace/hoolagans/GPClassify/experiments/compare_fitness_methods.py
```

This regenerates all tables and figures in `reports/fitness_method_comparison/`.
