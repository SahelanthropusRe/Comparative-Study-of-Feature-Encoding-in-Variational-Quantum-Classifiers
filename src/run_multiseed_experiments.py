"""Run reproducibility experiments for the multi-dataset VQC study.

Runs five VQC encodings and a Logistic Regression baseline on each dataset
using the same stratified train/test split for each seed. Results are written
under results/multi_seed/ so existing single-seed outputs are not overwritten.

Run from the project root:
    python -m src.run_multiseed_experiments

Optional examples:
    python -m src.run_multiseed_experiments --seeds 7 21
    python -m src.run_multiseed_experiments --datasets iris wine
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.data_preprocessing import prepare_dataset
from src.encodings import VALID_ENCODINGS
from src.quantum_model import train_vqc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "multi_seed"
DEFAULT_DATASETS = ["iris", "breast_cancer", "wine"]
DEFAULT_SEEDS = [42, 7, 21, 100, 123]


def score_predictions(y_true, y_pred):
    return {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_class_1": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_class_1": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_class_1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def run_one_seed(dataset_name: str, seed: int, encodings: list[str], epochs: int) -> list[dict]:
    """Run the classical baseline and all requested VQC encodings for a seed."""
    data = prepare_dataset(dataset_name=dataset_name, random_state=seed, test_size=0.20)
    rows: list[dict] = []

    # Classical baseline uses exactly the same seed-specific split and scaled data.
    classical = LogisticRegression(random_state=seed, max_iter=2000, solver="lbfgs")
    start = time.perf_counter()
    classical.fit(data["X_train"], data["y_train"])
    classical_time = time.perf_counter() - start
    classical_train_pred = classical.predict(data["X_train"])
    classical_test_pred = classical.predict(data["X_test"])
    classical_scores = score_predictions(data["y_test"], classical_test_pred)
    rows.append({
        "dataset": dataset_name,
        "model": "Logistic Regression",
        "encoding": "Classical",
        "seed": seed,
        "epochs": None,
        "train_samples": data["train_sample_count"],
        "test_samples": data["test_sample_count"],
        "training_accuracy": float(accuracy_score(data["y_train"], classical_train_pred)),
        "final_loss": None,
        "best_loss": None,
        "training_time_seconds": float(classical_time),
        **classical_scores,
    })

    for encoding in encodings:
        print(f"\n[{dataset_name} | seed={seed} | encoding={encoding}]", flush=True)
        result = train_vqc(
            encoding,
            dataset_name,
            epochs=epochs,
            seed=seed,
            save_results=False,
        )
        y_true = result.get("y_test", data["y_test"])
        y_pred = result.get("predictions")
        scores = score_predictions(y_true, y_pred)
        rows.append({
            "dataset": dataset_name,
            "model": "VQC",
            "encoding": encoding,
            "seed": seed,
            "epochs": epochs,
            "train_samples": result["train_samples"],
            "test_samples": result["test_samples"],
            "training_accuracy": result["training_accuracy"],
            "final_loss": result["final_loss"],
            "best_loss": result["best_loss"],
            "training_time_seconds": result["training_time_seconds"],
            **scores,
        })

    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-seed VQC and Logistic Regression evaluation.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                        choices=DEFAULT_DATASETS, help="Datasets to evaluate.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help="Random seeds (default: 42 7 21 100 123).")
    parser.add_argument("--epochs", type=int, default=30, help="VQC training epochs (default: 30).")
    parser.add_argument("--encodings", nargs="+", default=list(VALID_ENCODINGS),
                        choices=list(VALID_ENCODINGS), help="VQC encodings to evaluate.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1:
        raise SystemExit("--epochs must be at least 1.")
    if len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("Please provide each seed only once.")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    total_vqc = len(args.datasets) * len(args.encodings) * len(args.seeds)
    print("=" * 76)
    print("MULTI-SEED VQC REPRODUCIBILITY EXPERIMENTS")
    print(f"Datasets : {args.datasets}")
    print(f"Encodings: {args.encodings}")
    print(f"Seeds    : {args.seeds}")
    print(f"VQC runs : {total_vqc}")
    print(f"Epochs   : {args.epochs}")
    print(f"Output   : {OUTPUT_ROOT}")
    print("Existing results will not be modified.")
    print("=" * 76)

    for dataset in args.datasets:
        for seed in args.seeds:
            all_rows.extend(run_one_seed(dataset, seed, args.encodings, args.epochs))
            # Save progress after each dataset/seed pair in case a long run is interrupted.
            pd.DataFrame(all_rows).to_csv(OUTPUT_ROOT / "multiseed_results.csv", index=False)

    detailed = pd.DataFrame(all_rows)
    detailed_path = OUTPUT_ROOT / "multiseed_results.csv"
    detailed.to_csv(detailed_path, index=False)

    metric_columns = [
        "training_accuracy", "test_accuracy", "precision_class_1", "recall_class_1",
        "f1_class_1", "final_loss", "best_loss", "training_time_seconds",
    ]
    summary = detailed.groupby(["dataset", "model", "encoding"], dropna=False)[metric_columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary_path = OUTPUT_ROOT / "multiseed_summary_mean_std.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 76)
    print("MULTI-SEED SUMMARY (TEST ACCURACY MEAN AND STANDARD DEVIATION)")
    print("=" * 76)
    display_columns = ["dataset", "model", "encoding", "test_accuracy_mean", "test_accuracy_std"]
    print(summary[display_columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nDetailed results: {detailed_path}")
    print(f"Mean/std summary: {summary_path}")


if __name__ == "__main__":
    main()
