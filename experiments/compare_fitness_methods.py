from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
    make_classification,
)
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from gpclassify import GPClassifier


METHODS = ["accuracy", "f1_score", "pearson_r2"]
SEEDS = list(range(8))


def _random_overlap_dataset(
    seed: int,
    n_samples: int,
    n_features: int,
    weights: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    classes = len(weights)
    X = rng.normal(size=(n_samples, n_features))
    y = rng.choice(np.arange(classes), size=n_samples, p=np.array(weights))
    return X, y


def dataset_builders() -> Dict[str, Callable[[int], Tuple[np.ndarray, np.ndarray]]]:
    return {
        # Previous set (real + synthetic)
        "real_breast_cancer_bin": lambda seed: load_breast_cancer(return_X_y=True),
        "real_iris_3c": lambda seed: load_iris(return_X_y=True),
        "real_wine_3c": lambda seed: load_wine(return_X_y=True),
        "real_digits_10c": lambda seed: load_digits(return_X_y=True),
        "syn_bin_balanced": lambda seed: make_classification(
            n_samples=600,
            n_features=12,
            n_informative=8,
            n_redundant=2,
            n_classes=2,
            weights=[0.5, 0.5],
            flip_y=0.03,
            class_sep=1.0,
            random_state=seed,
        ),
        "syn_bin_imbalanced": lambda seed: make_classification(
            n_samples=600,
            n_features=12,
            n_informative=8,
            n_redundant=2,
            n_classes=2,
            weights=[0.9, 0.1],
            flip_y=0.03,
            class_sep=1.0,
            random_state=seed,
        ),
        "syn_3c_balanced": lambda seed: make_classification(
            n_samples=750,
            n_features=14,
            n_informative=10,
            n_redundant=2,
            n_classes=3,
            n_clusters_per_class=1,
            weights=[1 / 3, 1 / 3, 1 / 3],
            flip_y=0.03,
            class_sep=1.0,
            random_state=seed,
        ),
        "syn_3c_imbalanced": lambda seed: make_classification(
            n_samples=750,
            n_features=14,
            n_informative=10,
            n_redundant=2,
            n_classes=3,
            n_clusters_per_class=1,
            weights=[0.7, 0.2, 0.1],
            flip_y=0.03,
            class_sep=1.0,
            random_state=seed,
        ),
        "syn_5c_imbalanced": lambda seed: make_classification(
            n_samples=1000,
            n_features=16,
            n_informative=12,
            n_redundant=2,
            n_classes=5,
            n_clusters_per_class=1,
            weights=[0.55, 0.2, 0.15, 0.07, 0.03],
            flip_y=0.03,
            class_sep=1.1,
            random_state=seed,
        ),
        # New overlap-heavy imbalanced set
        "syn_bin_imbalanced_high_overlap": lambda seed: make_classification(
            n_samples=700,
            n_features=14,
            n_informative=6,
            n_redundant=4,
            n_classes=2,
            weights=[0.9, 0.1],
            flip_y=0.15,
            class_sep=0.2,
            random_state=seed,
        ),
        "syn_3c_imbalanced_high_overlap": lambda seed: make_classification(
            n_samples=800,
            n_features=16,
            n_informative=6,
            n_redundant=6,
            n_classes=3,
            n_clusters_per_class=1,
            weights=[0.75, 0.2, 0.05],
            flip_y=0.2,
            class_sep=0.2,
            random_state=seed,
        ),
        "syn_bin_imbalanced_complete_overlap": lambda seed: _random_overlap_dataset(
            seed=seed, n_samples=700, n_features=14, weights=[0.9, 0.1]
        ),
        "syn_4c_imbalanced_complete_overlap": lambda seed: _random_overlap_dataset(
            seed=seed, n_samples=900, n_features=14, weights=[0.7, 0.17, 0.1, 0.03]
        ),
    }


def _per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> Dict[int, float]:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = {}
    for i, cls in enumerate(labels):
        support = cm[i, :].sum()
        per_class[int(cls)] = float(cm[i, i] / support) if support > 0 else 0.0
    return per_class


def run_experiments(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    runs: List[Dict[str, object]] = []
    per_class_rows: List[Dict[str, object]] = []

    datasets = dataset_builders()
    for dataset_name, builder in datasets.items():
        for seed in SEEDS:
            X, y = builder(seed)
            labels = np.unique(y)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=seed, stratify=y
            )
            X_train_l = X_train.tolist()
            X_test_l = X_test.tolist()
            y_train_l = y_train.tolist()

            for method in METHODS:
                clf = GPClassifier(
                    num_models=20,
                    generations=12,
                    max_depth=6,
                    selection_method="pareto_tournament",
                    fitness_method=method,
                    random_state=seed,
                )
                clf.fit(X_train_l, y_train_l)
                y_pred = np.array(clf.predict(X_test_l))
                acc = float(accuracy_score(y_test, y_pred))
                macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
                per_class = _per_class_accuracy(y_test, y_pred, labels)
                mean_per_class_accuracy = float(np.mean(list(per_class.values())))

                runs.append(
                    {
                        "dataset": dataset_name,
                        "seed": seed,
                        "fitness_method": method,
                        "accuracy": acc,
                        "macro_f1": macro_f1,
                        "mean_per_class_accuracy": mean_per_class_accuracy,
                    }
                )
                for cls, cls_acc in per_class.items():
                    per_class_rows.append(
                        {
                            "dataset": dataset_name,
                            "seed": seed,
                            "fitness_method": method,
                            "class_label": cls,
                            "per_class_accuracy": cls_acc,
                        }
                    )

    _write_rows_csv(output_dir / "run_level_metrics.csv", runs)
    _write_rows_csv(output_dir / "per_class_metrics.csv", per_class_rows)

    summary = _build_summary(runs)
    with (output_dir / "summary_by_dataset.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    _make_metric_boxplot(runs, "accuracy", figures_dir / "boxplot_accuracy_by_method.png")
    _make_metric_boxplot(runs, "macro_f1", figures_dir / "boxplot_macro_f1_by_method.png")
    _make_metric_boxplot(
        runs,
        "mean_per_class_accuracy",
        figures_dir / "boxplot_mean_per_class_accuracy_by_method.png",
    )
    _make_overlap_dataset_boxplot(runs, figures_dir / "boxplot_overlap_datasets.png")


def _write_rows_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_summary(runs: List[Dict[str, object]]) -> Dict[str, object]:
    by_dataset_method: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(
        lambda: {"accuracy": [], "macro_f1": [], "mean_per_class_accuracy": []}
    )
    for row in runs:
        key = (str(row["dataset"]), str(row["fitness_method"]))
        by_dataset_method[key]["accuracy"].append(float(row["accuracy"]))
        by_dataset_method[key]["macro_f1"].append(float(row["macro_f1"]))
        by_dataset_method[key]["mean_per_class_accuracy"].append(float(row["mean_per_class_accuracy"]))

    dataset_summary: List[Dict[str, object]] = []
    winners = {"accuracy": defaultdict(int), "macro_f1": defaultdict(int), "mean_per_class_accuracy": defaultdict(int)}
    datasets = sorted({str(r["dataset"]) for r in runs})

    for dataset in datasets:
        rows = []
        for method in METHODS:
            vals = by_dataset_method[(dataset, method)]
            item = {
                "dataset": dataset,
                "fitness_method": method,
                "accuracy_mean": float(np.mean(vals["accuracy"])),
                "accuracy_std": float(np.std(vals["accuracy"])),
                "macro_f1_mean": float(np.mean(vals["macro_f1"])),
                "macro_f1_std": float(np.std(vals["macro_f1"])),
                "mean_per_class_accuracy_mean": float(np.mean(vals["mean_per_class_accuracy"])),
                "mean_per_class_accuracy_std": float(np.std(vals["mean_per_class_accuracy"])),
            }
            rows.append(item)
            dataset_summary.append(item)

        for metric_key, metric_field in [
            ("accuracy", "accuracy_mean"),
            ("macro_f1", "macro_f1_mean"),
            ("mean_per_class_accuracy", "mean_per_class_accuracy_mean"),
        ]:
            best = max(rows, key=lambda r: float(r[metric_field]))
            winners[metric_key][str(best["fitness_method"])] += 1

    overall = {}
    for method in METHODS:
        rows = [r for r in runs if str(r["fitness_method"]) == method]
        overall[method] = {
            "accuracy_mean": float(np.mean([float(r["accuracy"]) for r in rows])),
            "macro_f1_mean": float(np.mean([float(r["macro_f1"]) for r in rows])),
            "mean_per_class_accuracy_mean": float(
                np.mean([float(r["mean_per_class_accuracy"]) for r in rows])
            ),
        }

    return {
        "overall": overall,
        "winner_counts_by_dataset": {k: dict(v) for k, v in winners.items()},
        "dataset_summary": dataset_summary,
    }


def _make_metric_boxplot(runs: List[Dict[str, object]], metric: str, out_path: Path) -> None:
    values = []
    labels = []
    for method in METHODS:
        method_values = [float(r[metric]) for r in runs if str(r["fitness_method"]) == method]
        values.append(method_values)
        labels.append(method)

    plt.figure(figsize=(8, 5))
    plt.boxplot(values, labels=labels, showmeans=True)
    plt.title(f"Distribution of {metric} by fitness method")
    plt.ylabel(metric)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _make_overlap_dataset_boxplot(runs: List[Dict[str, object]], out_path: Path) -> None:
    overlap_datasets = {
        "syn_bin_imbalanced_high_overlap",
        "syn_3c_imbalanced_high_overlap",
        "syn_bin_imbalanced_complete_overlap",
        "syn_4c_imbalanced_complete_overlap",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=False)
    metrics = [("macro_f1", "Macro-F1"), ("mean_per_class_accuracy", "Mean Per-Class Accuracy")]
    for ax, (metric_key, title) in zip(axes, metrics):
        plot_values = []
        labels = []
        for method in METHODS:
            values = [
                float(r[metric_key])
                for r in runs
                if str(r["dataset"]) in overlap_datasets and str(r["fitness_method"]) == method
            ]
            plot_values.append(values)
            labels.append(method)
        ax.boxplot(plot_values, labels=labels, showmeans=True)
        ax.set_title(f"Overlap datasets: {title}")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    report_dir = Path("reports") / "fitness_method_comparison"
    run_experiments(report_dir)
