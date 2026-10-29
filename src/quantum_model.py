"""
quantum_model.py

Variational Quantum Classifier with selectable
feature encoding.

Supported encodings:
    E1 - Full range [-pi, pi]
    E2 - Compressed range [-pi/2, pi/2]
    E3 - Reordered features using [-pi, pi]

The VQC architecture, optimizer, learning rate,
number of layers, epochs, initialization and
dataset are kept identical for all encodings.

This allows a controlled ablation study.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np

from src.data_preprocessing import prepare_dataset
from src.encodings import VALID_ENCODINGS, apply_encoding


# ============================================================
# Configuration
# ============================================================

N_QUBITS = 2
N_LAYERS = 2

LEARNING_RATE = 0.1
EPOCHS = 30

SEED = 42

DEFAULT_DATASET = "iris"
DEFAULT_ENCODING = "E1"
RESULTS_ROOT = Path("results") / "multi_dataset"


# ============================================================
# Quantum device
# ============================================================

dev = qml.device(
    "default.qubit",
    wires=N_QUBITS
)


# ============================================================
# Feature encoding
# ============================================================

def encode_features(features, encoding_name):
    """Apply one of the five supported data encodings to the two qubits."""
    name = str(encoding_name).strip().upper()
    if name not in VALID_ENCODINGS:
        raise ValueError(
            f"Invalid encoding {encoding_name!r}. Choose from {', '.join(VALID_ENCODINGS)}."
        )
    apply_encoding(features, name, wires=(0, 1))


# ============================================================
# VQC Circuit
# ============================================================

@qml.qnode(dev)
def quantum_circuit(
    features,
    weights,
    encoding_name
):
    """Common two-qubit VQC circuit for all five feature encodings."""

    # --------------------------------------------------------
    # Feature encoding
    # --------------------------------------------------------

    encode_features(
        features,
        encoding_name
    )

    # --------------------------------------------------------
    # Trainable layers
    # --------------------------------------------------------

    for layer in range(N_LAYERS):

        # Qubit 0
        qml.RY(
            weights[layer, 0, 0],
            wires=0
        )

        qml.RZ(
            weights[layer, 0, 1],
            wires=0
        )

        # Qubit 1
        qml.RY(
            weights[layer, 1, 0],
            wires=1
        )

        qml.RZ(
            weights[layer, 1, 1],
            wires=1
        )

        # Entanglement
        qml.CNOT(
            wires=[0, 1]
        )

    # --------------------------------------------------------
    # Measurement
    # --------------------------------------------------------

    return qml.expval(
        qml.PauliZ(0)
    )


# ============================================================
# Prediction
# ============================================================

def predict_single(
    features,
    weights,
    encoding_name
):
    """
    Predict one sample.

    Expectation value >= 0 -> class 1
    Expectation value <  0 -> class 0
    """

    output = quantum_circuit(
        features,
        weights,
        encoding_name
    )

    if output >= 0:
        return 1

    return 0


def predict_dataset(
    X,
    weights,
    encoding_name
):
    """
    Predict all samples in X.
    """

    predictions = []

    for features in X:

        prediction = predict_single(
            features,
            weights,
            encoding_name
        )

        predictions.append(
            prediction
        )

    return np.array(
        predictions
    )


# ============================================================
# Loss function
# ============================================================

def loss_function(
    X,
    y,
    weights,
    encoding_name
):
    """
    Mean squared error loss.

    Class 0 -> -1
    Class 1 -> +1
    """

    total_loss = 0.0

    for features, label in zip(X, y):

        output = quantum_circuit(
            features,
            weights,
            encoding_name
        )

        target = 2 * label - 1

        total_loss += (
            output - target
        ) ** 2

    return total_loss / len(X)


# ============================================================
# Accuracy
# ============================================================

def calculate_accuracy(
    X,
    y,
    weights,
    encoding_name
):
    """
    Calculate classification accuracy.
    """

    predictions = predict_dataset(
        X,
        weights,
        encoding_name
    )

    correct = np.sum(
        predictions == y
    )

    return correct / len(y)


# ============================================================
# Train selected encoding
# ============================================================

def train_vqc(encoding_name=DEFAULT_ENCODING, dataset_name=DEFAULT_DATASET, *, epochs=EPOCHS, learning_rate=LEARNING_RATE, seed=SEED, save_results=True):
    """
    Train the VQC using the selected encoding.

    Parameters
    ----------
    encoding_name : str
        E1, E2 or E3.

    Returns
    -------
    dict
        Experiment results.
    """

    encoding_name = str(encoding_name).strip().upper()
    if encoding_name not in VALID_ENCODINGS:
        raise ValueError(
            f"Invalid encoding: {encoding_name}. Choose from {', '.join(VALID_ENCODINGS)}."
        )

    dataset_name = str(dataset_name).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "breast": "breast_cancer", "cancer": "breast_cancer", "wisconsin": "breast_cancer",
        "breast_cancer_wisconsin": "breast_cancer",
        "breast_cancer_wisconsin_diagnostic": "breast_cancer",
    }
    dataset_name = aliases.get(dataset_name, dataset_name)
    if dataset_name not in {"iris", "breast_cancer", "wine"}:
        raise ValueError("Invalid dataset. Choose iris, breast_cancer, or wine.")
    if int(epochs) < 1:
        raise ValueError("epochs must be at least 1.")
    if float(learning_rate) <= 0:
        raise ValueError("learning_rate must be positive.")

    print("\n" + "=" * 65)
    print(
        f"VQC EXPERIMENT - {dataset_name.upper()} - ENCODING {encoding_name}"
    )
    print("=" * 65)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    data = prepare_dataset(dataset_name=dataset_name, random_state=int(seed), test_size=0.20)
    X_train = np.asarray(data["X_train"], dtype=float)
    X_test = np.asarray(data["X_test"], dtype=float)
    y_train = np.asarray(data["y_train"], dtype=int)
    y_test = np.asarray(data["y_test"], dtype=int)

    # --------------------------------------------------------
    # Encoding information
    # --------------------------------------------------------

    encoding_ranges = {
        "ANGLE": "[0, pi] feature rotations",
        "ZZ": "RX feature rotations + data-dependent IsingZZ interaction",
        "E1": "[-pi, pi]",
        "E2": "[-pi/2, pi/2]",
        "E3": "[-pi, pi] + feature reorder",
    }

    print("\n" + "=" * 65)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 65)

    print(f"Dataset         : {data['display_name']} ({dataset_name})")
    print(f"Task            : {data['description']}")
    print(f"Features        : {', '.join(data['feature_names'])}")
    print(f"Train/test size : {len(X_train)}/{len(X_test)}")
    print(
        f"Encoding        : {encoding_name}"
    )

    print(
        f"Encoding scheme : "
        f"{encoding_ranges[encoding_name]}"
    )

    print(
        f"Qubits          : {N_QUBITS}"
    )

    print(
        f"VQC layers      : {N_LAYERS}"
    )

    print(
        "Optimizer       : Adam"
    )

    print(
        f"Learning rate   : {learning_rate}"
    )

    print(
        f"Epochs          : {epochs}"
    )

    print(
        f"Random seed     : {seed}"
    )
    if encoding_name == "ZZ":
        print("ZZ feature map  : RX rotations + IsingZZ interaction")

    # --------------------------------------------------------
    # Initialize weights
    # --------------------------------------------------------

    np.random.seed(int(seed))

    weights = np.array(
        0.01 * np.random.randn(
            N_LAYERS,
            N_QUBITS,
            2
        ),
        requires_grad=True
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = qml.AdamOptimizer(
        stepsize=float(learning_rate)
    )

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    loss_history = []
    accuracy_history = []

    # --------------------------------------------------------
    # Start timer
    # --------------------------------------------------------

    start_time = time.time()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print(
        f"TRAINING {encoding_name}"
    )
    print("=" * 65)

    for epoch in range(int(epochs)):

        weights, current_loss = (
            optimizer.step_and_cost(
                lambda w: loss_function(
                    X_train,
                    y_train,
                    w,
                    encoding_name
                ),
                weights
            )
        )

        train_accuracy = calculate_accuracy(
            X_train,
            y_train,
            weights,
            encoding_name
        )

        loss_history.append(
            float(current_loss)
        )

        accuracy_history.append(
            float(train_accuracy)
        )

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Loss: {current_loss:.6f} | "
            f"Train Accuracy: "
            f"{train_accuracy * 100:.2f}%"
        )

    # --------------------------------------------------------
    # Training time
    # --------------------------------------------------------

    training_time = (
        time.time() - start_time
    )

    # --------------------------------------------------------
    # Test accuracy
    # --------------------------------------------------------

    test_accuracy = calculate_accuracy(
        X_test,
        y_test,
        weights,
        encoding_name
    )

    # --------------------------------------------------------
    # Best loss and convergence
    # --------------------------------------------------------

    best_loss = min(
        loss_history
    )

    best_loss_epoch = (
        loss_history.index(best_loss) + 1
    )

    # --------------------------------------------------------
    # Final results
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print(
        f"{encoding_name} FINAL RESULTS"
    )
    print("=" * 65)

    print(
        f"Final training loss : "
        f"{loss_history[-1]:.6f}"
    )

    print(
        f"Best training loss  : "
        f"{best_loss:.6f}"
    )

    print(
        f"Best loss epoch     : "
        f"{best_loss_epoch}"
    )

    print(
        f"Training accuracy   : "
        f"{accuracy_history[-1] * 100:.2f}%"
    )

    print(
        f"Test accuracy       : "
        f"{test_accuracy * 100:.2f}%"
    )

    print(
        f"Training time       : "
        f"{training_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # Save results in a new dataset-specific directory so legacy
    # results/figures and results/tables are never overwritten.
    # --------------------------------------------------------

    if save_results:
        base_dir = RESULTS_ROOT / dataset_name
        figures_dir = base_dir / "figures"
        tables_dir = base_dir / "tables"
        figures_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)
        stem = f"vqc_{dataset_name}_{encoding_name.lower()}"

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, int(epochs) + 1), loss_history, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss (MSE)")
        plt.title(f"{data['display_name']} VQC Training Loss - {encoding_name}")
        plt.grid(True)
        plt.tight_layout()
        loss_plot_path = figures_dir / f"{stem}_loss_curve.png"
        plt.savefig(loss_plot_path, dpi=300)
        plt.close()

        history_path = tables_dir / f"{stem}_training_history.csv"
        with history_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "loss", "train_accuracy"])
            for i, (loss, accuracy) in enumerate(zip(loss_history, accuracy_history), start=1):
                writer.writerow([i, float(loss), float(accuracy)])

        summary_path = tables_dir / "experiment_summary.csv"
        summary_fields = [
            "dataset", "encoding", "samples", "train_samples", "test_samples",
            "epochs", "learning_rate", "final_loss", "best_loss", "best_loss_epoch",
            "training_accuracy", "test_accuracy", "training_time_seconds", "seed",
        ]
        result_row = {
            "dataset": dataset_name,
            "encoding": encoding_name,
            "samples": int(data["selected_sample_count"]),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "final_loss": float(loss_history[-1]),
            "best_loss": float(best_loss),
            "best_loss_epoch": int(best_loss_epoch),
            "training_accuracy": float(accuracy_history[-1]),
            "test_accuracy": float(test_accuracy),
            "training_time_seconds": float(training_time),
            "seed": int(seed),
        }
        file_exists = summary_path.exists() and summary_path.stat().st_size > 0
        with summary_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=summary_fields)
            if not file_exists:
                writer.writeheader()
            writer.writerow(result_row)

        print(f"\nLoss curve saved   : {loss_plot_path}")
        print(f"Training history   : {history_path}")
        print(f"Experiment summary : {summary_path}")

    return {
        "dataset": dataset_name,
        "encoding": encoding_name,
        "samples": int(data["selected_sample_count"]),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "final_loss": float(loss_history[-1]),
        "best_loss": float(best_loss),
        "best_loss_epoch": int(best_loss_epoch),
        "training_accuracy": float(accuracy_history[-1]),
        "test_accuracy": float(test_accuracy),
        "training_time": float(training_time),
        "training_time_seconds": float(training_time),
        "weights": weights,
        "loss_history": [float(loss) for loss in loss_history],
        "accuracy_history": [float(acc) for acc in accuracy_history],
        "y_test": y_test,
        "predictions": predict_dataset(X_test, weights, encoding_name),
        "feature_names": list(data["feature_names"]),
        "class_names": dict(data["class_names"]),
    }


# ============================================================
# Multi-dataset experiment runner and command-line interface
# ============================================================

def run_all_experiments(datasets=None, encodings=None, *, epochs=EPOCHS,
                        learning_rate=LEARNING_RATE, seed=SEED):
    """Run each requested dataset/encoding pair sequentially."""
    datasets = list(datasets or ["iris", "breast_cancer", "wine"])
    encodings = [str(item).upper() for item in (encodings or VALID_ENCODINGS)]
    results = []
    for dataset_name in datasets:
        for encoding_name in encodings:
            results.append(train_vqc(
                encoding_name, dataset_name, epochs=epochs,
                learning_rate=learning_rate, seed=seed,
            ))
    return results


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Train the VQC on Iris, Breast Cancer, or Wine with five encodings."
    )
    parser.add_argument(
        "dataset_or_encoding", nargs="?", default=DEFAULT_DATASET,
        help="Dataset (iris/breast_cancer/wine), or a legacy encoding name such as E2.",
    )
    parser.add_argument(
        "encoding", nargs="?", default=None,
        help="Encoding: ANGLE, ZZ, E1, E2, or E3 (default E1).",
    )
    parser.add_argument("--all", action="store_true", help="Run all 15 dataset/encoding combinations.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"Training epochs (default {EPOCHS}).")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help=f"Adam learning rate (default {LEARNING_RATE}).")
    parser.add_argument("--seed", type=int, default=SEED, help=f"Random seed (default {SEED}).")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.all:
        run_all_experiments(epochs=args.epochs, learning_rate=args.learning_rate, seed=args.seed)
    else:
        first = args.dataset_or_encoding
        if args.encoding is None and str(first).strip().upper() in VALID_ENCODINGS:
            # Preserve the original CLI: python -m src.quantum_model E2
            dataset_name, encoding_name = DEFAULT_DATASET, str(first).strip().upper()
        else:
            dataset_name = first
            encoding_name = args.encoding or DEFAULT_ENCODING
        train_vqc(
            encoding_name,
            dataset_name,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
