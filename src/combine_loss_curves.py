from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Save this script in your project root and run:
# python combine_loss_curves.py
# It searches the project recursively for the 15 training-history CSV files.

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_ROOT = PROJECT_ROOT / "results" / "multi_dataset"
DATASETS = ["iris", "breast_cancer", "wine"]
ENCODINGS = ["angle", "zz", "e1", "e2", "e3"]


def find_history_csv(dataset: str, encoding: str) -> Path | None:
    expected_name = f"vqc_{dataset}_{encoding}_training_history.csv"
    # Search recursively, but avoid searching virtual environments and caches.
    excluded = {".git", "venv", ".venv", "env", "__pycache__", "node_modules"}
    for path in PROJECT_ROOT.rglob(expected_name):
        if not any(part in excluded for part in path.parts):
            return path
    return None


def main() -> None:
    fig, ax = plt.subplots(figsize=(15, 8.5))
    cmap = plt.get_cmap("tab20")
    missing = []
    curve_index = 0

    for dataset in DATASETS:
        for encoding in ENCODINGS:
            csv_path = find_history_csv(dataset, encoding)
            if csv_path is None:
                missing.append(f"vqc_{dataset}_{encoding}_training_history.csv")
                continue

            try:
                df = pd.read_csv(csv_path)
            except Exception as exc:
                print(f"Could not read {csv_path}: {exc}")
                missing.append(str(csv_path))
                continue

            # Expected columns in the uploaded files: epoch, loss, train_accuracy.
            if not {"epoch", "loss"}.issubset(df.columns):
                print(f"Skipping {csv_path}: expected columns 'epoch' and 'loss'.")
                missing.append(str(csv_path))
                continue

            color = cmap(curve_index % cmap.N)
            label = f"{dataset.replace('_', ' ').title()} - {encoding.upper()}"
            ax.plot(
                df["epoch"], df["loss"],
                color=color,
                linewidth=2,
                marker="o",
                markersize=2.5,
                label=label
            )
            curve_index += 1

    ax.set_title("VQC Training Loss: 15 Curves Across Three Datasets", fontsize=16, weight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Training loss (MSE)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xticks(range(1, 31, 2))
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=True)

    fig.tight_layout()
    output_dir = PROJECT_ROOT / "results" / "multi_dataset" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "all_15_loss_curves.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"\nPlotted {curve_index} loss curves.")
    print(f"Graph saved to: {output_path}")
    if missing:
        print("\nCould not find/read these history files:")
        for name in missing:
            print(f"  - {name}")
        print("Make sure all 15 CSV files are somewhere inside the project folder.")
    elif curve_index == 15:
        print("Success: all 15 curves are on one graph, each with a different colour.")


if __name__ == "__main__":
    main()
