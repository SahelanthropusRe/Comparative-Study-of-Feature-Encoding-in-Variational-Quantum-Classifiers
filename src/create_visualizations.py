"""Create a small, teacher-friendly set of VQC experiment visualizations.

Input:  results/multi_seed/multiseed_results.csv
Output: results/multi_seed/essential_visualizations/

Run from project root: python -m src.create_visualizations
(Place this file at src/create_visualizations.py first.)

Creates five main figures:
1. Three dataset comparison dashboards (accuracy, class-1 F1, runtime)
2. One cross-dataset encoding heatmap
3. One seed-stability plot
Plus summary CSVs and a simple HTML report.
"""
from pathlib import Path
import html
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = Path("results/multi_seed/multiseed_results.csv")
OUTPUT_DIR = Path("results/multi_seed/essential_visualizations")
ENCODINGS = ["ANGLE", "ZZ", "E1", "E2", "E3"]
DATASETS = {
    "iris": "Iris: Setosa vs Versicolor",
    "breast_cancer": "Breast Cancer Wisconsin",
    "wine": "Wine: Class 0 vs Class 1",
}


def load_results():
    """Read the raw run-level CSV and validate the required columns."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_FILE}. Run the multi-seed experiment first, "
            "or update INPUT_FILE to your CSV location."
        )
    data = pd.read_csv(INPUT_FILE)
    required = {
        "dataset", "model", "encoding", "seed", "test_accuracy",
        "training_accuracy", "f1_class_1", "training_time_seconds",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    if data.empty:
        raise ValueError("The results CSV is empty.")
    return data


def mean_and_std(values):
    """Return mean and sample standard deviation, ignoring missing values."""
    values = values.dropna()
    if values.empty:
        return float("nan"), 0.0
    return values.mean(), values.std(ddof=1) if len(values) > 1 else 0.0


def plot_dataset_dashboard(data, dataset_key, dataset_title):
    """Compare all encodings and Logistic Regression on three key metrics."""
    subset = data[data["dataset"] == dataset_key]
    labels = [f"VQC {name}" for name in ENCODINGS] + ["Logistic\nRegression"]
    metric_specs = [
        ("test_accuracy", "Mean test accuracy (%)", 100, False),
        ("f1_class_1", "Mean class-1 F1-score (%)", 100, False),
        ("training_time_seconds", "Training time (seconds)", 1, True),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    for axis, (metric, label, multiplier, log_scale) in zip(axes, metric_specs):
        means, deviations = [], []
        for encoding in ENCODINGS:
            values = subset.loc[
                (subset["model"] == "VQC") & (subset["encoding"] == encoding), metric
            ]
            mean, deviation = mean_and_std(values)
            means.append(mean * multiplier)
            deviations.append(deviation * multiplier)

        baseline = subset.loc[subset["model"] == "Logistic Regression", metric]
        mean, deviation = mean_and_std(baseline)
        means.append(mean * multiplier)
        deviations.append(deviation * multiplier)

        axis.bar(range(len(labels)), means, yerr=deviations, capsize=3)
        axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        axis.set_title(label)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
        if multiplier == 100:
            axis.set_ylim(0, 105)
        if log_scale:
            axis.set_yscale("log")
            axis.set_ylabel("Training time (seconds; logarithmic scale)")

    fig.suptitle(f"{dataset_title}: mean ± standard deviation across seeds", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{dataset_key}_model_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_encoding_heatmap(data):
    """Show the mean accuracy of each encoding across all three datasets."""
    quantum_rows = data[data["model"] == "VQC"]
    matrix = (
        quantum_rows.groupby(["dataset", "encoding"])["test_accuracy"]
        .mean().unstack().reindex(index=list(DATASETS), columns=ENCODINGS) * 100
    )
    fig, axis = plt.subplots(figsize=(8.5, 4.5))
    image = axis.imshow(matrix.values, aspect="auto", vmin=0, vmax=100)
    axis.set_xticks(range(len(ENCODINGS)), ENCODINGS)
    axis.set_yticks(range(len(DATASETS)), [DATASETS[key] for key in matrix.index])
    axis.set_title("Mean VQC test accuracy across datasets")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(col, row, f"{matrix.iloc[row, col]:.1f}%", ha="center", va="center")
    fig.colorbar(image, ax=axis, label="Mean test accuracy (%)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cross_dataset_encoding_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_seed_stability(data):
    """Show whether model accuracy changes with the random seed."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3))
    for axis, (dataset_key, title) in zip(axes, DATASETS.items()):
        subset = data[data["dataset"] == dataset_key]
        for encoding in ENCODINGS:
            rows = subset[
                (subset["model"] == "VQC") & (subset["encoding"] == encoding)
            ].sort_values("seed")
            axis.plot(rows["seed"], rows["test_accuracy"] * 100,
                      marker="o", label=encoding)
        baseline = subset[subset["model"] == "Logistic Regression"].sort_values("seed")
        axis.plot(baseline["seed"], baseline["test_accuracy"] * 100,
                  marker="s", linestyle="--", label="Logistic Regression")
        axis.set_title(title)
        axis.set_xlabel("Random seed")
        axis.set_ylabel("Test accuracy (%)")
        axis.set_ylim(0, 105)
        axis.grid(alpha=0.25)
    axes[-1].legend(fontsize=8, loc="lower left")
    fig.suptitle("Test accuracy stability across five random seeds")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "accuracy_stability_across_seeds.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary_tables(data):
    """Write aggregate metrics and a compact best-VQC-versus-baseline table."""
    metrics = [
        "test_accuracy", "training_accuracy", "precision_class_1", "recall_class_1",
        "f1_class_1", "training_time_seconds", "final_loss", "best_loss",
    ]
    metrics = [metric for metric in metrics if metric in data.columns]
    summary = data.groupby(["dataset", "model", "encoding"])[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(str(piece) for piece in column if str(piece) not in ("", "None"))
        if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    summary.to_csv(OUTPUT_DIR / "summary_statistics.csv", index=False)

    comparison_rows = []
    for key, title in DATASETS.items():
        subset = data[data["dataset"] == key]
        quantum = subset[subset["model"] == "VQC"].groupby("encoding")["test_accuracy"].mean()
        best_encoding = quantum.idxmax()
        best_runs = subset[(subset["model"] == "VQC") & (subset["encoding"] == best_encoding)]["test_accuracy"]
        baseline = subset[subset["model"] == "Logistic Regression"]["test_accuracy"]
        comparison_rows.append({
            "dataset": title,
            "best_vqc_encoding": best_encoding,
            "best_vqc_mean_accuracy_pct": best_runs.mean() * 100,
            "best_vqc_std_percentage_points": best_runs.std(ddof=1) * 100,
            "logistic_regression_mean_accuracy_pct": baseline.mean() * 100,
            "logistic_regression_std_percentage_points": baseline.std(ddof=1) * 100,
            "difference_percentage_points": (best_runs.mean() - baseline.mean()) * 100,
        })
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUTPUT_DIR / "best_vqc_vs_logistic_regression.csv", index=False)
    return comparison


def write_explanation(data, comparison):
    """Write concise talking points for explaining the results to a teacher."""
    lines = [
        "VQC PROJECT — SIMPLE RESULTS EXPLANATION",
        "=" * 42,
        "",
        "What was compared:",
        "- Five VQC encodings: ANGLE, ZZ, E1, E2, and E3.",
        "- Logistic Regression as the classical baseline.",
        "- Three binary datasets and five random seeds per configuration.",
        "- Error bars represent standard deviation across seeds.",
        "",
        "MAIN ACCURACY RESULTS",
    ]
    for _, row in comparison.iterrows():
        lines.extend([
            "",
            f"{row['dataset']}:",
            f"  Best VQC encoding: {row['best_vqc_encoding']} — "
            f"{row['best_vqc_mean_accuracy_pct']:.2f}% ± "
            f"{row['best_vqc_std_percentage_points']:.2f} percentage points.",
            f"  Logistic Regression: {row['logistic_regression_mean_accuracy_pct']:.2f}% ± "
            f"{row['logistic_regression_std_percentage_points']:.2f} percentage points.",
            f"  Difference (VQC minus baseline): {row['difference_percentage_points']:+.2f} percentage points.",
        ])
    lines.extend([
        "",
        "HOW TO EXPLAIN THE FIGURES",
        "1. Each dataset dashboard compares test accuracy, class-1 F1, and training time.",
        "2. The heatmap makes it easy to see that no encoding wins on every dataset.",
        "3. The seed plot shows how sensitive the results are to the random seed.",
        "4. Higher accuracy/F1 is generally better; lower runtime is faster.",
        "",
        "LIMITATIONS TO MENTION",
        "- Five seeds give an initial comparison, not proof of statistical significance.",
        "- Seeds change both the train/test split and model initialization.",
        "- The Wine test set is small, so a few predictions can change accuracy noticeably.",
        "- For Breast Cancer, class 1 is benign under scikit-learn's target convention.",
        "- These simulator results do not demonstrate quantum advantage.",
        "- Runtime depends on the implementation and computer used.",
    ])
    (OUTPUT_DIR / "results_explanation.txt").write_text("\n".join(lines), encoding="utf-8")
    return "\n".join(lines)


def write_html_report(comparison, explanation):
    """Create a simple HTML page that displays the five main figures."""
    figure_files = [
        (f"{key}_model_comparison.png", f"{title}: model comparison")
        for key, title in DATASETS.items()
    ]
    figure_files += [
        ("cross_dataset_encoding_heatmap.png", "Encoding comparison across datasets"),
        ("accuracy_stability_across_seeds.png", "Accuracy stability across seeds"),
    ]
    pieces = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>VQC Essential Results</title>",
        "<style>body{font-family:Arial,sans-serif;max-width:1150px;margin:28px auto;padding:0 18px;line-height:1.5}img{width:100%;height:auto;border:1px solid #ddd;margin:8px 0 26px}pre{white-space:pre-wrap;background:#f5f5f5;padding:16px;border-radius:8px}</style>",
        "</head><body><h1>VQC Project — Essential Results</h1>",
        "<p>Five key figures: one model comparison per dataset, one encoding heatmap, and one seed-stability plot.</p>",
    ]
    for filename, title in figure_files:
        pieces.append(f"<h2>{html.escape(title)}</h2><img src='{html.escape(filename)}' alt='{html.escape(title)}'>")
    pieces.append("<h2>Simple results explanation</h2><pre>" + html.escape(explanation) + "</pre>")
    pieces.append("</body></html>")
    (OUTPUT_DIR / "index.html").write_text("\n".join(pieces), encoding="utf-8")


def main():
    """Validate the input and generate the five main figures and supporting files."""
    data = load_results()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checks = [
        ("Expected 90 rows", len(data) == 90),
        ("All three datasets present", set(data["dataset"]) == set(DATASETS)),
        ("Expected five seeds present", set(data["seed"].astype(int)) == {7, 21, 42, 100, 123}),
        ("No duplicate experiment rows", not data.duplicated(["dataset", "model", "encoding", "seed"]).any()),
    ]
    for key in DATASETS:
        subset = data[data["dataset"] == key]
        checks.append((f"{key}: all five VQC encodings have five runs", all(
            len(subset[(subset["model"] == "VQC") & (subset["encoding"] == encoding)]) == 5
            for encoding in ENCODINGS
        )))
        checks.append((f"{key}: five Logistic Regression runs", len(subset[subset["model"] == "Logistic Regression"]) == 5))
    pd.DataFrame(checks, columns=["check", "passed"]).to_csv(OUTPUT_DIR / "validation_checks.csv", index=False)
    for key, title in DATASETS.items():
        plot_dataset_dashboard(data, key, title)
    plot_encoding_heatmap(data)
    plot_seed_stability(data)
    comparison = write_summary_tables(data)
    explanation = write_explanation(data, comparison)
    write_html_report(comparison, explanation)
    failed = [name for name, passed in checks if not passed]
    print(f"Created 5 figures and supporting report files in: {OUTPUT_DIR.resolve()}")
    print(f"Validation checks passed: {len(checks) - len(failed)}/{len(checks)}")
    if failed:
        print("Checks needing attention:", "; ".join(failed))

if __name__ == "__main__":
    main()
