# Variational Quantum Classifier with Self-Designed Feature Encoding

## A Comparative Study Across Multiple Datasets

This project compares five data-encoding strategies in a two-qubit
Variational Quantum Classifier (VQC) across three binary-classification
datasets. Logistic Regression is used as a classical baseline, and
experiments are repeated across five random seeds.

> **Important:** This is an ideal state-vector simulator study. It does
> not establish quantum advantage. The Breast Cancer experiment is not a
> clinically validated diagnostic system.

## Objectives

-   Prepare three datasets using two numerical features per dataset.
-   Use stratified 80:20 train/test splits and fit scaling on training
    data only.
-   Implement a two-qubit VQC with trainable RY/RZ rotations, CNOT
    entanglement, and Pauli-Z expectation measurement.
-   Compare ANGLE, ZZ, E1, E2, and E3 encodings.
-   Compare with Logistic Regression.
-   Repeat experiments over five seeds and summarize mean and standard
    deviation.

## Datasets

| Dataset | Binary Task | Features | Samples |
|---|---|---|---:|
| Iris | Setosa vs. Versicolor | Petal length, petal width | 100 |
| Breast Cancer Wisconsin Diagnostic | Malignant vs. benign | Mean radius, mean concave points | 569 |
| Wine | Class 0 vs. class 1 (class 2 excluded) | Flavanoids, color intensity | 130 |

Seeds used in the multi-seed experiment: `42`, `7`, `21`, `100`, `123`.

The scaler is fitted only on the training split and then applied to the test split. Test values are not clipped and may fall outside `[0, 1]`.

## Encoding Strategies

Let `x₁` and `x₂` be the two scaled features.

| Encoding | Definition | Purpose |
|---|---|---|
| ANGLE | `θᵢ = πxᵢ` | Reference angle encoding |
| ZZ | Feature rotations + `φ = π(2x₁ − 1)(2x₂ − 1)` using an IsingZZ interaction | Data-dependent entangling encoding |
| E1 | `θᵢ = 2πxᵢ − π` | Project-designed mapping to `[-π, π]` |
| E2 | `θᵢ = πxᵢ − π/2` | Project-designed mapping to `[-π/2, π/2]` |
| E3 | E1 mapping with feature assignment swapped | Feature-order ablation |

E1, E2, and E3 are project-designed experimental variants motivated by data-encoding literature. Their exact names and formulas are not claimed to be copied from a publication. Check `src/encodings.py` for the exact gate axes and ordering used in your implementation.

## VQC Architecture and Training

-   Qubits: **2**; variational layers: **2**
-   Trainable gates: **RY and RZ rotations**; entangling gate: **CNOT**
-   Measurement: expectation value of Pauli-Z on qubit 0
-   Optimizer: **Adam**, learning rate **0.1**, **30 epochs**
-   Loss: **Mean Squared Error (MSE)**, with labels mapped from `{0, 1}`
    to `{-1, +1}`

Prediction: class 0 if the expectation value is below zero; otherwise
class 1. The implementation uses an ideal state-vector simulator, not a
physical quantum processor.

## Technologies

Python, PennyLane, NumPy, pandas, scikit-learn, Matplotlib, and
optionally Pillow for combining existing plot images.

## Project Structure

``` text
project-root/
├── src/
│   ├── data_preprocessing.py
│   ├── encodings.py
│   ├── quantum_model.py
│   ├── classical_baseline.py
│   ├── run_multiseed_experiments.py
│   ├── create_visualizations.py
│   └── combine_loss_curves.py
├── results/
│   ├── multi_dataset/{iris,breast_cancer,wine}/{figures,tables}/
│   └── multi_seed/
│       ├── multiseed_results.csv
│       ├── multiseed_summary_mean_std.csv
│       └── essential_visualizations/
├── README.md
└── requirements.txt  # if maintained
```

This is the expected main structure; use the filenames actually present
in your checkout.

## Setup (Windows)

Replace the repository placeholders with your actual GitHub URL and
folder name:

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
python -m venv venv
venv\Scripts\activate
```

If a tested `requirements.txt` exists:

``` bash
python -m pip install -r requirements.txt
```

Otherwise, install the main packages:

``` bash
python -m pip install pennylane numpy pandas scikit-learn matplotlib pillow
```

Record the Python and package versions used for your final run.

## Run the Project

Run commands from the **project root**, not from inside `src`.

### Validate preprocessing

``` bash
python -m src.data_preprocessing
```

### Run the classical baseline

``` bash
python -m src.classical_baseline
```

### Run one VQC experiment

``` bash
python -m src.quantum_model iris E1
```

Other dataset names are `breast_cancer` and `wine`; encoding names are
`ANGLE`, `ZZ`, `E1`, `E2`, and `E3`. Check the current module's CLI if
the interface differs.

### Run all encodings for one seed

``` bash
python -m src.quantum_model --all
```

This is an all-dataset/all-encoding run for one seed, not the five-seed
experiment.

### Run the multi-seed experiments

``` bash
python -m src.run_multiseed_experiments
```

The full run is expected to contain 3 datasets × 5 encodings × 5 seeds =
**75 VQC runs**, plus 3 datasets × 5 seeds = **15 Logistic Regression
evaluations**. It may take substantially longer than a single run. Check
available options with:

``` bash
python -m src.run_multiseed_experiments --help
```

### Generate visualizations

``` bash
python -m src.create_visualizations
```

Run after the required multi-seed CSV files have been generated.


## Experiment Configuration

| Setting | Value |
|---|---|
| Datasets | Iris, Breast Cancer Wisconsin Diagnostic, Wine |
| Features per dataset | 2 |
| Train/test split | Stratified 80:20 |
| Random seeds | 42, 7, 21, 100, 123 |
| Encodings | ANGLE, ZZ, E1, E2, E3 |
| Qubits | 2 |
| Variational layers | 2 |
| Optimizer | Adam |
| Learning rate | 0.1 |
| Epochs | 30 |
| Loss function | Mean Squared Error (MSE) |
| Classical baseline | Logistic Regression |
| Backend | Ideal state-vector simulator |



## Results

The currently available five-seed summary reports the following mean test accuracies. Verify these values against the final CSV files before submission. Standard deviations are reported in percentage points (pp).

| Dataset | Best VQC Encoding | Mean VQC Test Accuracy | Logistic Regression Mean Accuracy |
|---|---|---:|---:|
| Iris | E2 | 98.00% ± 2.74 pp | 100.00% ± 0.00 pp |
| Breast Cancer | ANGLE | 89.82% ± 2.29 pp | 90.35% ± 2.63 pp |
| Wine | ANGLE and ZZ (tie) | 89.23% | 84.62% ± 3.85 pp |

### Interpretation

- The best VQC encoding varies by dataset.
- Logistic Regression performs better on Iris and slightly better on Breast Cancer in this summary.
- ANGLE and ZZ exceed the Logistic Regression mean on Wine in this experiment. However, the Wine test set contains only 26 samples per split.
- Five seeds provide an initial stability check but do not establish statistical significance.
- These results do not demonstrate quantum advantage.

### Output Files

Expected outputs may include:

- `results/multi_seed/multiseed_results.csv`
- `results/multi_seed/multiseed_summary_mean_std.csv`
- `results/multi_dataset/tables/classical_baseline_all_datasets.csv`
- Per-dataset summaries, training histories, confusion matrices, and plots.

Use the files generated by your final experiment run as the source of truth.


## Recommended Visualizations

1.  Experimental workflow.
2.  Two-qubit VQC circuit architecture.
3.  Comparison of all five encodings.
4.  Per-dataset model comparison.
5.  Cross-dataset encoding heatmap.
6.  Accuracy stability across seeds.
7.  Combined loss plot with all 15 encoding--dataset curves, when
    numerical training histories are available.

Clearly distinguish single-seed loss curves from multi-seed summary
statistics.

## Reproducibility Notes

-   Run commands from the project root.
-   Fit the scaler on training data only.
-   Keep seeds and model settings consistent across comparisons.
-   Record the code version and package versions used for final results.
-   Ensure VQC and baseline use comparable splits.
-   Do not call loss curves multi-seed averages unless the values were
    explicitly aggregated.

## Limitations

-   Only two features are used per dataset.
-   The VQC uses two qubits and two variational layers.
-   Only an ideal state-vector simulator is used; hardware noise,
    finite-shot sampling, and error mitigation are not evaluated.
-   Only five seeds and one classical baseline are used.
-   Wine has a small test set.
-   The Breast Cancer model is not clinically validated.
-   No quantum speedup or quantum advantage is demonstrated.

## References

1.  M. Schuld, "Variational classifier," PennyLane Demos.
    https://pennylane.ai/demos/tutorial_variational_classifier/
2.  M. Schuld, R. Sweke, and J. J. Meyer, "Effect of data encoding on
    the expressive power of variational quantum-machine-learning
    models," *Physical Review A*, vol. 103, 032430, 2021.
    https://doi.org/10.1103/PhysRevA.103.032430
3.  V. Havlíček et al., "Supervised learning with quantum-enhanced
    feature spaces," *Nature*, vol. 567, pp. 209--212, 2019.
    https://doi.org/10.1038/s41586-019-0980-2
4.  D. Ranga et al., "Quantum Machine Learning: Exploring the Role of
    Data Encoding Techniques, Challenges, and Future Directions,"
    *Mathematics*, vol. 12, no. 21, 3318, 2024.
    https://doi.org/10.3390/math12213318

