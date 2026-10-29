"""Multi-dataset Logistic Regression baseline for the VQC project.

Run from the project root:
    python -m src.classical_baseline

Outputs are saved under results/multi_dataset/ without overwriting VQC results.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
)

from src.data_preprocessing import prepare_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results" / "multi_dataset"
DATASETS = ("iris", "breast_cancer", "wine")
RANDOM_SEED = 42


def run_baseline_for_dataset(dataset_name: str, random_state: int = RANDOM_SEED) -> dict[str, Any]:
    """Fit and evaluate Logistic Regression on one dataset using shared preprocessing."""
    data = prepare_dataset(dataset_name, random_state=random_state)
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    # prepare_dataset already fits MinMaxScaler on training data only.
    # Do not scale these arrays a second time.
    model = LogisticRegression(random_state=random_state, max_iter=2000, solver="lbfgs")
    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    precision = precision_score(y_test, test_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, test_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, test_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_test, test_pred, labels=[0, 1])

    dataset_dir = RESULTS_ROOT / dataset_name
    tables_dir = dataset_dir / "tables"
    figures_dir = dataset_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    class_names_display = [str(label) for label in data["class_names"]]

    row = {
        "dataset": dataset_name,
        "model": "Logistic Regression",
        "samples": int(data["selected_sample_count"]),
        "train_samples": int(data["train_sample_count"]),
        "test_samples": int(data["test_sample_count"]),
        "random_state": int(random_state),
        "max_iter": 2000,
        "training_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "precision_class_1": float(precision),
        "recall_class_1": float(recall),
        "f1_class_1": float(f1),
        "training_time_seconds": float(training_time),
        "true_negative": int(cm[0, 0]),
        "false_positive": int(cm[0, 1]),
        "false_negative": int(cm[1, 0]),
        "true_positive": int(cm[1, 1]),
        "features": ", ".join(data["feature_names"]),
        "class_names": " vs ".join(class_names_display),
    }
    pd.DataFrame([row]).to_csv(tables_dir / "classical_baseline.csv", index=False)

    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names_display)
    display.plot(values_format="d", colorbar=False)
    plt.title(f"Logistic Regression — {data['display_name']}")
    plt.tight_layout()
    plt.savefig(figures_dir / "classical_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 68)
    print(f"LOGISTIC REGRESSION BASELINE — {data['display_name']}")
    print("=" * 68)
    print(f"Features          : {', '.join(data['feature_names'])}")
    print(f"Samples           : {data['selected_sample_count']}")
    print(f"Train/test split  : {data['train_sample_count']}/{data['test_sample_count']}")
    print(f"Training accuracy : {train_accuracy:.4f} ({train_accuracy:.2%})")
    print(f"Test accuracy     : {test_accuracy:.4f} ({test_accuracy:.2%})")
    print(f"Precision (class 1): {precision:.4f}")
    print(f"Recall (class 1)   : {recall:.4f}")
    print(f"F1 (class 1)       : {f1:.4f}")
    print(f"Training time     : {training_time:.3f} seconds")
    print("Confusion matrix (rows=true labels [0, 1], columns=predicted [0, 1]):")
    print(cm)
    print(f"Saved table       : {tables_dir / 'classical_baseline.csv'}")
    print(f"Saved plot        : {figures_dir / 'classical_confusion_matrix.png'}")
    return row


def main() -> None:
    """Run Logistic Regression on all three datasets and save combined results."""
    rows = [run_baseline_for_dataset(name, RANDOM_SEED) for name in DATASETS]
    combined = pd.DataFrame(rows)
    combined_dir = RESULTS_ROOT / "tables"
    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_path = combined_dir / "classical_baseline_all_datasets.csv"
    combined.to_csv(combined_path, index=False)
    columns = ["dataset", "training_accuracy", "test_accuracy", "precision_class_1",
               "recall_class_1", "f1_class_1", "training_time_seconds"]
    print("\n" + "=" * 68)
    print("CLASSICAL BASELINE SUMMARY")
    print("=" * 68)
    print(combined[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nCombined results saved to: {combined_path}")


if __name__ == "__main__":
    main()
