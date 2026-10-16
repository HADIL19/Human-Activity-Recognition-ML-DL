# Human Activity Recognition on Shimmer Accelerometer Data

Classical machine learning (KNN, SVM, Decision Tree) for recognising seven human activities from Shimmer accelerometer recordings. The project includes a Streamlit dashboard, repeated-split evaluation, statistical testing, leakage-aware segment-grouped evaluation, and **saved-model inference without retraining**.

Developed during the CDTA internship *ML Development & Transfer Learning* at the Centre de Développement des Technologies Avancées, October 2026.

## Key Results

| Protocol | Best Model | Accuracy | Balanced Accuracy | MCC |
|---|---|---:|---:|---:|
| A. Random window split (10 runs) | KNN | 0.962 | 0.930 | 0.955 |
| B. Segment-grouped split (10 runs) | SVM | 0.779 | 0.780 | 0.742 |

- All models performed above chance under the random-window evaluation protocol (permutation test, 100 permutations, p = 0.0099).
- Random window splitting can overestimate generalisation performance because windows from the same recording segment may appear in both training and test sets.
- Holding out entire segments reduces balanced accuracy by approximately 0.11–0.19 and changes the model ranking.
- On unseen segments, Sit and Stand are the more challenging classes, with Sit recall around 0.33 and Stand recall around 0.6–0.7.

These results illustrate why evaluation should account for the relationship between windows originating from the same recording.

The full analysis is documented in the project report.

## Features

- **Activity recognition:** classify seven activities using KNN, SVM and Decision Tree.
- **Statistical feature extraction:** extract 12 features from each 128-sample accelerometer window.
- **Repeated evaluation:** compare models across multiple random or segment-grouped train/test splits.
- **Hyperparameter tuning:** optimise models using cross-validation on training data.
- **Statistical analysis:** evaluate performance differences and compare models using statistical tests.
- **Interactive dashboard:** explore data, run experiments, visualise results and download outputs.
- **Model persistence:** export fitted models and associated metadata into a downloadable ZIP bundle.
- **Inference without retraining:** load a previously saved model bundle and predict activities from compatible feature CSV files or supported MATLAB recordings.
- **Prediction export:** download activity predictions as a CSV file.

## Dataset

The dataset consists of seven MATLAB files, one for each activity:

`Fall.mat`, `Jump.mat`, `Lay.mat`, `Run.mat`, `Sit.mat`, `Stand.mat` and `Walk.mat`.

Each file contains a numeric matrix named `data` with 11 columns: a timestamp, accelerometer X/Y/Z, and seven additional channels that are not used by the baseline feature extraction.

| Activity | Samples | Windows (128 samples) | Segments |
|---|---:|---:|---:|
| Fall | 6,016 | 47 | 13 |
| Jump | 18,304 | 143 | 12 |
| Lay | 42,240 | 330 | 12 |
| Run | 36,992 | 289 | 13 |
| Sit | 44,032 | 344 | 10 |
| Stand | 52,480 | 410 | 16 |
| Walk | 45,696 | 357 | 13 |
| **Total** | **245,760** | **1,920** | **89** |

**Dataset note:** The raw MATLAB files are not included in the repository. Place them in the local `data/` directory before running experiments.

## Methodology

### 1. Windowing

Each recording is divided into non-overlapping windows of 128 consecutive samples. All samples are used, with no leftover samples in the current dataset.

### 2. Feature Extraction

Twelve statistical features are extracted from each window:

- X-axis: maximum, minimum, mean and standard deviation.
- Y-axis: maximum, minimum, mean and standard deviation.
- Z-axis: maximum, minimum, mean and standard deviation.

The sample standard deviation uses `ddof=1`.

The timestamp and additional sensor channels are not included in the baseline model's feature vector.

### 3. Models

Three classical machine-learning algorithms are evaluated:

- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM), using RBF or linear kernels
- Decision Tree

Where required, feature standardisation is performed within a pipeline fitted exclusively on training data.

### 4. Hyperparameter Tuning

Grid search with five-fold cross-validation is used to tune model hyperparameters. Balanced accuracy is the optimisation metric, and the test split is reserved for final evaluation.

For the segment-grouped protocol, the cross-validation folds are also group-aware.

### 5. Evaluation Protocols

Each protocol is evaluated over 10 runs, using seeds starting at 42.

**Protocol A — Random Window Split**
- Stratified 75/25 train/test split at the window level.
- Requested by the internship supervisor.
- Windows from the same recording segment may occur in both sets.

**Protocol B — Segment-Grouped Split**
- Entire recording segments are assigned to either training or testing.
- No segment is shared between the two sets.
- Cross-validation also respects segment grouping.

The grouped protocol provides a more demanding assessment of generalisation to unseen recording segments.

### 6. Evaluation Metrics

The project reports:

- Accuracy
- Balanced accuracy
- Weighted and macro F1-score
- Matthews Correlation Coefficient (MCC)
- Macro one-vs-rest ROC-AUC

### 7. Statistical Testing

Implemented in `stat_tests.py`:

- Permutation test against chance-level performance under the random protocol.
- Paired Wilcoxon signed-rank tests.
- Nadeau–Bengio corrected t-tests.
- Holm adjustment for multiple pairwise comparisons.
- Friedman test for comparisons across the three models.

Repeated runs reuse the same underlying dataset. Consequently, the results are not fully independent, and statistical p-values should be interpreted cautiously.

## Saved Models and Inference

The project now supports **saving trained models and reusing them for predictions without retraining**.

### Exporting Trained Models

1. Start the Streamlit dashboard.
2. Load the dataset and select an evaluation protocol.
3. Open the **ML Experiments** tab.
4. Click **Run model experiments**.
5. Open the **Downloads** tab.
6. Download the trained-model ZIP bundle.

The bundle contains the fitted model artefacts and metadata needed for compatible inference, including:

- Fitted model pipelines.
- Label encoder.
- Feature order.
- Window size.
- Evaluation protocol metadata.

Keep the downloaded ZIP file somewhere safe so you can reuse it later.

**Important:** The existing experiment pipeline continues to fit models for each experiment run. Saving a model preserves a fitted model for later use; it does not automatically make the model learn from new data each time you run the application.

### Predicting with a Saved Model

To use a previously saved model:

1. Expand **Predict activities with a saved model** near the top of the dashboard.
2. Upload the saved model ZIP bundle.
3. Upload either:
   - A CSV containing the 12 expected baseline statistical features, or
   - A compatible MATLAB `.mat` recording.
4. Select the model you want to use.
5. Click **Predict activities**.
6. Download the predictions as a CSV file.

This workflow allows inference without repeating the original training and hyperparameter-tuning process.

### MATLAB Inference Assumptions

MATLAB inference assumes that:

- The variable is named `data`.
- `data` is a numeric matrix.
- The first column contains timestamps.
- The next three columns contain accelerometer X, Y and Z.
- The input represents one continuous recording.
- Feature extraction uses non-overlapping windows of 128 samples and sample standard deviation with `ddof=1`.

Do not apply this workflow across unverified recording boundaries or unknown channel mappings. Confirm that inference uses the same feature definitions and ordering as training.

### Model Improvement

Model improvement is an experimental process, not cumulative training.

The saved-model feature preserves fitted models for reuse, but it does not incrementally train them whenever new data is uploaded. Improving a model requires a deliberate training procedure with suitable training data and evaluation.

Avoid selecting models by repeatedly checking held-out test performance. Use training and validation data for model selection, keeping the test data separate for final assessment.

### Security

Model bundles use Joblib serialisation. **Only load model bundles from trusted sources**, because loading an untrusted serialised model can execute malicious code.

## Project Structure

```text
.
├── app.py                  # Streamlit dashboard and inference interface
├── model_io.py             # Model bundle export/import and prediction helpers
├── data_utils.py           # Load and clean MATLAB and CSV data
├── feature_extraction.py   # Windowing and statistical feature extraction
├── ml_experiments.py       # Training, tuning and repeated evaluation
├── stat_tests.py           # Statistical tests and model comparisons
├── eda_visualizations.py   # Exploratory data visualisations
├── model_visualizations.py # Model comparison visualisations
├── requirements.txt        # Python dependencies
├── data/                   # Local MATLAB recordings (git-ignored)
└── results/                # Experiment results and CSV outputs
```

The `data/` directory contains the local raw recordings and should not be committed to the repository. Generated results may be kept locally or selectively committed when they are useful for documenting the project.

## Installation

### 1. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

If Joblib or SciPy is not already included in your dependencies, install them with:

```bash
python -m pip install joblib scipy
```

## Usage

### Launch the Dashboard

```bash
python -m streamlit run app.py
```

Set **Dataset folder** in the sidebar to the directory containing your MATLAB files, or upload the required files through the interface.

### Run Model Experiments

1. Select the random or segment-grouped evaluation protocol.
2. Open the **ML Experiments** tab.
3. Run the experiments.
4. Inspect model performance, evaluation metrics and statistical results.
5. Download the available outputs from the **Model Comparison** and **Downloads** tabs.

### Save and Reuse a Model

After running experiments, open **Downloads** and download the trained-model ZIP bundle if the dashboard offers it.

Later, upload the ZIP under **Predict activities with a saved model** to perform inference without rerunning training.

### Run Experiments from Python

```python
import pandas as pd
from ml_experiments import repeated_random_split_experiments

features = pd.read_csv("results/features_128.csv")

result = repeated_random_split_experiments(
    features,
    n_runs=10,
    split_mode="grouped",  # "random" or "grouped"
    n_permutations=100,    # Used for the random protocol
)

print(result["summary"])
print(result["pairwise_tests"])
print(result["chance_test"])
```

The example assumes that `results/features_128.csv` has already been generated and contains the expected feature columns and metadata.

### Rebuild Features from Raw MATLAB Files

```python
import glob
from data_utils import load_uploaded_files, clean_raw_data
from feature_extraction import (
    extract_window_features,
    validate_feature_table,
)

raw, _ = load_uploaded_files(sorted(glob.glob("data/*.mat")))
clean, report = clean_raw_data(raw)

features = validate_feature_table(
    extract_window_features(clean)
)
```

This rebuilds the feature table from the local MATLAB recordings. The generated feature CSV is not required to be committed to GitHub.

## Dependencies

The project uses Python, pandas, NumPy, SciPy, scikit-learn, Streamlit, Matplotlib, Joblib and the appropriate MATLAB-file loading dependencies.

See `requirements.txt` for the project's actual dependency list.

## Reproducibility and Data Availability

- Raw MATLAB recordings are stored locally and excluded from version control.
- Experiments use recorded random seeds for repeatability.
- Saved-model bundles allow fitted models to be reused for inference.
- Evaluation results depend on the split protocol, training data, feature extraction and model configuration.
- The random-window and segment-grouped protocols measure different generalisation scenarios and should not be treated as interchangeable.

## Acknowledgements

Developed as part of the *ML Development & Transfer Learning* internship at the Centre de Développement des Technologies Avancées (CDTA), October 2026.