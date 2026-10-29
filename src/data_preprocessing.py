"""
data_preprocessing.py

Leakage-safe preprocessing for the multi-dataset Variational Quantum
Classifier (VQC) study.

Datasets and binary tasks
-------------------------
1. Iris: Setosa (0) vs Versicolor (1)
   Features: petal length, petal width
2. Breast Cancer Wisconsin Diagnostic: malignant (0) vs benign (1)
   Features: mean radius, mean concave points
3. Wine: class_0 (0) vs class_1 (1); class_2 is excluded
   Features: flavanoids, color intensity

The scaler is always fitted on the training partition only.

Backward compatibility
----------------------
The existing project expects this call to return five values:
    X_train, X_test, y_train, y_test, scaler = load_and_prepare_data()

That behavior is preserved. Pass dataset_name to select a dataset. For richer
metadata, use prepare_dataset() or set return_metadata=True.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_iris,
    load_wine,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


RANDOM_SEED = 42
TEST_SIZE = 0.20

# Dataset configuration is kept explicit so the experiment can be audited.
DATASET_CONFIG: Dict[str, Dict[str, Any]] = {
    "iris": {
        "display_name": "Iris",
        "feature_names": ["petal length (cm)", "petal width (cm)"],
        "feature_indices": [2, 3],
        "class_names": {0: "Setosa", 1: "Versicolor"},
        "description": "Iris Setosa versus Iris Versicolor",
    },
    "breast_cancer": {
        "display_name": "Breast Cancer Wisconsin Diagnostic",
        "feature_names": ["mean radius", "mean concave points"],
        "class_names": {0: "Malignant", 1: "Benign"},
        "description": "Malignant versus benign breast tumors",
    },
    "wine": {
        "display_name": "Wine",
        "feature_names": ["flavanoids", "color_intensity"],
        "class_names": {0: "Wine class 0", 1: "Wine class 1"},
        "description": "Wine class 0 versus Wine class 1; class 2 excluded",
    },
}


def _normalize_dataset_name(dataset_name: str) -> str:
    """Normalize common aliases and validate a dataset name."""
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError(
            "dataset_name must be one of: 'iris', 'breast_cancer', 'wine'."
        )

    name = dataset_name.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "breast": "breast_cancer",
        "cancer": "breast_cancer",
        "wisconsin": "breast_cancer",
        "breast_cancer_wisconsin": "breast_cancer",
        "breast_cancer_wisconsin_diagnostic": "breast_cancer",
    }
    name = aliases.get(name, name)

    if name not in DATASET_CONFIG:
        choices = ", ".join(DATASET_CONFIG.keys())
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Choose one of: {choices}."
        )
    return name


def _load_binary_dataset(dataset_name: str):
    """
    Load a dataset and return its two selected features and binary labels.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 2)
    y : np.ndarray, shape (n_samples,), integer labels 0 and 1
    feature_names : list[str]
    class_names : dict[int, str]
    original_sample_count : int
    """
    name = _normalize_dataset_name(dataset_name)
    config = DATASET_CONFIG[name]

    if name == "iris":
        dataset = load_iris()
        # Keep Setosa (0) and Versicolor (1); exclude Virginica (2).
        mask = np.isin(dataset.target, [0, 1])
        X = dataset.data[mask][:, config["feature_indices"]]
        y = dataset.target[mask].astype(int)
        feature_names = ["petal_length", "petal_width"]
        class_names = {0: "Setosa", 1: "Versicolor"}
        original_sample_count = int(dataset.data.shape[0])

    elif name == "breast_cancer":
        dataset = load_breast_cancer()
        # sklearn names the features exactly "mean radius" and
        # "mean concave points". Its target is 0=malignant, 1=benign.
        feature_names_all = list(dataset.feature_names)
        try:
            indices = [
                feature_names_all.index("mean radius"),
                feature_names_all.index("mean concave points"),
            ]
        except ValueError as exc:
            raise RuntimeError(
                "Expected Breast Cancer feature names were not found in "
                "the installed scikit-learn dataset."
            ) from exc

        X = dataset.data[:, indices]
        y = dataset.target.astype(int)
        feature_names = ["mean_radius", "mean_concave_points"]
        class_names = {0: "Malignant", 1: "Benign"}
        original_sample_count = int(dataset.data.shape[0])

    else:  # wine
        dataset = load_wine()
        feature_names_all = list(dataset.feature_names)
        try:
            indices = [
                feature_names_all.index("flavanoids"),
                feature_names_all.index("color_intensity"),
            ]
        except ValueError as exc:
            raise RuntimeError(
                "Expected Wine feature names were not found in the installed "
                "scikit-learn dataset."
            ) from exc

        # Keep Wine classes 0 and 1; exclude class 2. These labels are already
        # 0 and 1, but explicitly map them to guarantee a binary convention.
        mask = np.isin(dataset.target, [0, 1])
        X = dataset.data[mask][:, indices]
        original_y = dataset.target[mask]
        y = np.where(original_y == 0, 0, 1).astype(int)
        feature_names = ["flavanoids", "color_intensity"]
        class_names = {0: "Wine class 0", 1: "Wine class 1"}
        original_sample_count = int(dataset.data.shape[0])

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)

    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(
            f"{name}: expected a 2D feature matrix with exactly two columns; "
            f"received shape {X.shape}."
        )
    if len(X) != len(y):
        raise ValueError(f"{name}: feature and label sample counts differ.")
    if not np.isfinite(X).all():
        raise ValueError(f"{name}: selected features contain NaN or infinity.")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError(
            f"{name}: expected both binary labels 0 and 1, got "
            f"{sorted(np.unique(y).tolist())}."
        )

    return X, y, feature_names, class_names, original_sample_count


def prepare_dataset(
    dataset_name: str = "iris",
    random_state: int = RANDOM_SEED,
    test_size: float = TEST_SIZE,
) -> Dict[str, Any]:
    """
    Load, split, and scale one dataset without data leakage.

    Parameters
    ----------
    dataset_name:
        'iris', 'breast_cancer', or 'wine'. Common breast-cancer aliases
        such as 'breast-cancer' are also accepted.
    random_state:
        Seed used for the stratified train/test split.
    test_size:
        Fraction of samples assigned to the held-out test set.

    Returns
    -------
    dict
        Contains scaled arrays, labels, fitted scaler, feature/class names,
        raw selected data, class distributions, and split metadata.
    """
    name = _normalize_dataset_name(dataset_name)

    if not 0.0 < float(test_size) < 1.0:
        raise ValueError("test_size must be between 0 and 1, exclusive.")

    X, y, feature_names, class_names, original_sample_count = (
        _load_binary_dataset(name)
    )

    # Split raw selected features before fitting any preprocessing transform.
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Critical leakage prevention: fit only on the training partition.
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # Numerical tolerance is used because floating-point transformations
    # can produce tiny deviations around the mathematical range boundaries.
    tolerance = 1e-12
    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise ValueError(f"{name}: scaled features contain NaN or infinity.")
    if X_train.min() < -tolerance or X_train.max() > 1 + tolerance:
        raise ValueError(f"{name}: training values are outside [0, 1].")

    # Test values can fall outside [0, 1] if a test value lies beyond the
    # training minimum/maximum. Do not clip them: clipping would hide this
    # legitimate behavior of a train-fitted MinMaxScaler.
    train_distribution = {
        int(label): int(count)
        for label, count in zip(*np.unique(y_train, return_counts=True))
    }
    test_distribution = {
        int(label): int(count)
        for label, count in zip(*np.unique(y_test, return_counts=True))
    }
    full_distribution = {
        int(label): int(count)
        for label, count in zip(*np.unique(y, return_counts=True))
    }

    return {
        "dataset_name": name,
        "display_name": DATASET_CONFIG[name]["display_name"],
        "description": DATASET_CONFIG[name]["description"],
        "X_train": np.asarray(X_train, dtype=np.float64),
        "X_test": np.asarray(X_test, dtype=np.float64),
        "y_train": np.asarray(y_train, dtype=np.int64),
        "y_test": np.asarray(y_test, dtype=np.int64),
        "X_train_raw": np.asarray(X_train_raw, dtype=np.float64),
        "X_test_raw": np.asarray(X_test_raw, dtype=np.float64),
        "y_all": y.copy(),
        "X_all_raw": X.copy(),
        "scaler": scaler,
        "feature_names": feature_names,
        "class_names": class_names,
        "original_sample_count": original_sample_count,
        "selected_sample_count": int(len(y)),
        "train_sample_count": int(len(y_train)),
        "test_sample_count": int(len(y_test)),
        "random_state": int(random_state),
        "test_size": float(test_size),
        "full_class_distribution": full_distribution,
        "train_class_distribution": train_distribution,
        "test_class_distribution": test_distribution,
        "train_scaled_min": X_train.min(axis=0),
        "train_scaled_max": X_train.max(axis=0),
        "test_scaled_min": X_test.min(axis=0),
        "test_scaled_max": X_test.max(axis=0),
    }


def print_preprocessing_summary(data: Dict[str, Any]) -> None:
    """Print a readable validation summary for a prepared dataset."""
    print("\n" + "=" * 76)
    print(f"PREPROCESSING SUMMARY: {data['display_name']}")
    print("=" * 76)
    print(f"Task                 : {data['description']}")
    print(f"Original samples     : {data['original_sample_count']}")
    print(f"Selected samples     : {data['selected_sample_count']}")
    print(f"Selected features    : {data['feature_names']}")
    print(f"Feature matrix shape : {data['X_all_raw'].shape}")
    print(f"Training samples     : {data['train_sample_count']}")
    print(f"Testing samples      : {data['test_sample_count']}")
    print(f"Random seed          : {data['random_state']}")
    print(f"Test fraction        : {data['test_size']:.2f}")
    print(f"Class names          : {data['class_names']}")
    print(f"Full class counts    : {data['full_class_distribution']}")
    print(f"Training class counts: {data['train_class_distribution']}")
    print(f"Testing class counts : {data['test_class_distribution']}")
    print(f"X_train shape        : {data['X_train'].shape}")
    print(f"X_test shape         : {data['X_test'].shape}")
    print(f"Training scaled min  : {data['train_scaled_min']}")
    print(f"Training scaled max  : {data['train_scaled_max']}")
    print(f"Test scaled min      : {data['test_scaled_min']}")
    print(f"Test scaled max      : {data['test_scaled_max']}")
    print("Scaler fitted only on training data: YES")
    print("=" * 76)


def load_and_prepare_data(
    dataset_name: str = "iris",
    random_state: int = RANDOM_SEED,
    test_size: float = TEST_SIZE,
    return_metadata: bool = False,
):
    """
    Backward-compatible data-loading interface.

    By default, returns the original five-item tuple:
        X_train, X_test, y_train, y_test, scaler

    Examples
    --------
    # Existing Iris code continues to work:
    X_train, X_test, y_train, y_test, scaler = load_and_prepare_data()

    # Select another dataset:
    X_train, X_test, y_train, y_test, scaler = load_and_prepare_data(
        dataset_name="breast_cancer"
    )

    # Get all arrays plus metadata:
    data = load_and_prepare_data("wine", return_metadata=True)
    """
    data = prepare_dataset(
        dataset_name=dataset_name,
        random_state=random_state,
        test_size=test_size,
    )
    print_preprocessing_summary(data)

    if return_metadata:
        return data

    return (
        data["X_train"],
        data["X_test"],
        data["y_train"],
        data["y_test"],
        data["scaler"],
    )


def validate_all_datasets(random_state: int = RANDOM_SEED) -> bool:
    """
    Run Phase 2 checks for all datasets and raise an error if a check fails.

    This validates sample counts, feature dimensions, label values, split
    sizes, finite values, and training-range scaling. It does not train models.
    """
    expected = {
        "iris": {"samples": 100, "train": 80, "test": 20},
        "breast_cancer": {"samples": 569, "train": 455, "test": 114},
        "wine": {"samples": 130, "train": 104, "test": 26},
    }

    print("\nPHASE 2: MULTI-DATASET PREPROCESSING VALIDATION")
    for name, expected_values in expected.items():
        data = prepare_dataset(name, random_state=random_state)
        print_preprocessing_summary(data)

        assert data["selected_sample_count"] == expected_values["samples"], (
            f"{name}: expected {expected_values['samples']} selected samples, "
            f"got {data['selected_sample_count']}."
        )
        assert data["train_sample_count"] == expected_values["train"], (
            f"{name}: expected {expected_values['train']} training samples, "
            f"got {data['train_sample_count']}."
        )
        assert data["test_sample_count"] == expected_values["test"], (
            f"{name}: expected {expected_values['test']} test samples, "
            f"got {data['test_sample_count']}."
        )
        assert data["X_train"].shape[1] == 2
        assert data["X_test"].shape[1] == 2
        assert set(np.unique(data["y_train"])) == {0, 1}
        assert set(np.unique(data["y_test"])) == {0, 1}
        assert np.isfinite(data["X_train"]).all()
        assert np.isfinite(data["X_test"]).all()
        assert data["X_train"].min() >= -1e-12
        assert data["X_train"].max() <= 1 + 1e-12

    print("\nAll Phase 2 preprocessing checks passed.")
    return True


if __name__ == "__main__":
    validate_all_datasets()
