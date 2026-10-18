# Human Activity Recognition on Shimmer Accelerometer Data

Classical machine learning for recognizing seven human activities from Shimmer accelerometer recordings. This project compares K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and Decision Trees using repeated evaluations, statistical tests, and a segment-grouped control for potential data leakage. A Streamlit dashboard supports exploration, training, saved-model management, and inference without retraining.

## Key Results

Mean scores over 10 runs:

| Evaluation protocol | Highest-accuracy model | Accuracy | Balanced accuracy | MCC |
|---|---|---:|---:|---:|
| A — Random window split | KNN | 0.962 | 0.930 | 0.955 |
| B — Segment-grouped split | SVM | 0.779 | 0.780 | 0.742 |

- All three algorithms scored above chance under protocol A (100-permutation test; p = 0.0099).
- Random window splits can overestimate performance when neighboring windows from the same recording segment occur in both training and test sets.
- Grouped evaluation lowers balanced accuracy and changes the model ranking. Pairwise differences do not survive the corrected t-test under protocol B.
- The grouped protocol is a more conservative estimate for unseen recording segments, **not** a verified subject-independent evaluation.

## Dataset

Seven MATLAB files, one per activity: `Fall.mat`, `Jump.mat`, `Lay.mat`, `Run.mat`, `Sit.mat`, `Stand.mat`, `Walk.mat`.

Each contains an 11-column numeric matrix named `data`: timestamp, accelerometer X/Y/Z, and seven additional sensor channels not used by the baseline model.

| Activity | Samples | 128-sample windows | Segments |
|---|---:|---:|---:|
| Fall | 6,016 | 47 | 13 |
| Jump | 18,304 | 143 | 12 |
| Lay | 42,240 | 330 | 12 |
| Run | 36,992 | 289 | 13 |
| Sit | 44,032 | 344 | 10 |
| Stand | 52,480 | 410 | 16 |
| Walk | 45,696 | 357 | 13 |
| **Total** | **245,760** | **1,920** | **89** |

Recording segments are inferred from timestamp resets; subject/session identities are unavailable. Timestamp units and sensor mapping have not been independently confirmed. The median timestamp step suggests approximately 51.2 Hz **if timestamps are in milliseconds**, giving about 2.50 seconds per 128-sample window.

Raw recordings are **not included** in this repository. Place authorized copies under `data/` locally.

## Methodology

### Windowing and features

The baseline extracts non-overlapping 128-sample windows and computes 12 features per window:

- X: maximum, minimum, mean, sample standard deviation
- Y: maximum, minimum, mean, sample standard deviation
- Z: maximum, minimum, mean, sample standard deviation

Standard deviation uses `ddof=1`. Timestamps and other sensor channels are excluded from the feature vector. Each file's sample count is divisible by 128; one window in the dataset crosses an inferred timestamp reset, a limitation of the original windowing procedure.

### Models and tuning

- **KNN:** neighbors, distance weighting, and Manhattan/Euclidean distance.
- **SVM:** RBF and linear kernels, with `C` and (for RBF) `gamma` tuned.
- **Decision Tree:** split criterion, maximum depth, and minimum split/leaf sizes.

Grid search uses five-fold cross-validation on training data, optimizing balanced accuracy. Imputation and scaling, where applicable, are fitted within training pipelines to avoid test-data leakage.

### Evaluation protocols

**A — Random window split:** 10 stratified 75/25 train/test splits, with seeds starting at 42. Windows from the same inferred segment may appear in both sets.

**B — Segment-grouped split:** 10 splits holding out complete inferred recording segments. Group-aware folds are also used for hyperparameter tuning. Because whole segments are held out, the test fraction is roughly 30–35% of windows.

Reported metrics include accuracy, balanced accuracy, weighted precision/recall/F1, macro F1, MCC, and macro one-vs-rest ROC-AUC. Statistical analysis includes a permutation test, Wilcoxon signed-rank test, Nadeau–Bengio corrected resampled t-test, Holm adjustment, and Friedman test. Repeated splits share data, so statistical p-values are approximate.

## Saved Models and Inference

The project saves **one final fitted model per algorithm** after the evaluation experiments. The repeated runs estimate performance; they are **not** cumulative training. The final model is refitted on all 1,920 feature windows using selected hyperparameters.

Saved Joblib artifacts include the fitted preprocessing/classifier pipeline and metadata such as class labels, feature names/order, hyperparameters, evaluation scores, and library versions. The persistence workflow maintains dated model files, a `latest` copy, and a `models_registry.csv` record of saved versions.

### Save models

1. Launch the Streamlit dashboard and load the data.
2. Run the ML experiments.
3. Open **Downloads → Trained models** to save final models to the local models folder or download individual `.joblib` files.
4. Download the model ZIP bundle for later inference, if offered in the dashboard.

Keep generated model artifacts local unless you deliberately want to publish them. You can exclude `models/` from Git with `.gitignore`.

### Predict without retraining

1. Open **Predict activities with a saved model** at the top of the dashboard, or use `predict_app.py` / `predict.py` where available.
2. Upload the saved model ZIP bundle.
3. Upload compatible raw Shimmer `.mat` or `.csv` recordings, or a CSV containing the 12 expected statistical features.
4. Select a model and run prediction.
5. View predictions per 128-sample window, a majority-vote activity for each recording, and download predictions as CSV.

When a true activity label is available, the inference interface can also show evaluation metrics and a confusion table. Inference must preserve the original feature definitions, order, and sensor mapping. Treat an uploaded raw recording as one continuous recording; do not silently window across unknown boundaries.

**Important:** Saving a model preserves its fitted parameters; it does not make KNN, SVM, or Decision Tree learn incrementally. Accuracy measured on the same data used to refit a final model is not a valid estimate of generalization. Use the held-out experiment results instead.

**Security:** Joblib files can execute code when loaded. Only load model files from trusted sources.

## Project Structure

```text
.
├── app.py                   # Streamlit dashboard
├── predict_app.py           # Standalone inference interface (if present)
├── predict.py               # Command-line inference (if present)
├── model_io.py              # Model persistence and inference utilities
├── data_utils.py            # Data loading and cleaning
├── feature_extraction.py    # Windowing and feature extraction
├── ml_experiments.py        # Training, tuning, evaluation
├── stat_tests.py            # Statistical analysis
├── eda_visualizations.py    # Exploratory plots
├── model_visualizations.py  # Model comparison plots
├── requirements.txt
├── data/                    # Local raw data; not committed
├── models/                  # Local saved models; not committed
└── results/                 # Generated evaluation results
```

## Installation and Usage

Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the dashboard:

```powershell
python -m streamlit run app.py
```

Use the dashboard to load the recordings, explore signals, compare both evaluation protocols, inspect metrics and statistical tests, and save or load trained models.

If needed, install model persistence dependencies:

```powershell
python -m pip install joblib scipy
```

## Reproducibility and Limitations

- Experiment runs use reproducible random seeds (starting at 42).
- Inferred segments are based on timestamp resets, not verified participant identifiers.
- Class frequencies are imbalanced; Fall has only 47 windows.
- Timestamp units, sampling rate, and column mapping require independent confirmation.
- The permutation test is for protocol A, and repeated-run statistical tests should be interpreted cautiously.
- Saved models should be evaluated on genuinely unseen recordings before deployment in real-world activity monitoring.

## Future Work

- Compare transfer learning with pre-trained VGG or ResNet models using image representations of accelerometer signals, such as CWT scalograms or Gramian angular fields.
- Investigate additional time- and frequency-domain features (RMS, median, skewness, kurtosis, magnitude, and spectral energy).
- Evaluate on verified participant-independent splits when participant IDs become available.
