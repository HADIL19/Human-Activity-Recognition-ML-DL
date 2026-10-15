# Human Activity Recognition on Shimmer Accelerometer Data

Classical machine learning (KNN, SVM, Decision Tree) for recognising seven activities
from Shimmer accelerometer recordings, with a Streamlit dashboard, repeated-split
evaluation, statistical tests and a leakage-aware (segment-grouped) control.

Developed during the CDTA internship *ML Development & Transfer Learning*
(Centre de Développement des Technologies Avancées, October 2026).


## Key results

| Protocol | Best model | Accuracy | Balanced accuracy | MCC |
|---|---|---|---|---|
| A. Random window split (10 runs) | KNN | 0.962 | 0.930 | 0.955 |
| B. Segment-grouped split (10 runs) | SVM | 0.779 | 0.780 | 0.742 |

- All models are far above chance (permutation test, 100 permutations, p = 0.0099, the minimum possible).
- A random split of windows **overestimates** performance: neighbouring windows from the same
  recording segment end up in both train and test. Holding out whole segments lowers balanced
  accuracy by 0.11 to 0.19 and changes the model ranking.
- On unseen segments, **Sit and Stand are the weak classes** (recall about 0.33 and 0.6 to 0.7).


The full analysis is in the project report.

## Dataset

Seven MATLAB files, one per activity (`Fall`, `Jump`, `Lay`, `Run`, `Sit`, `Stand`, `Walk`), each with a
variable `data` of 11 columns: timestamp, accelerometer X/Y/Z, and seven further channels (not used).

| Activity | Samples | Windows (128) | Segments |
|---|---|---|---|
| Fall | 6,016 | 47 | 13 |
| Jump | 18,304 | 143 | 12 |
| Lay | 42,240 | 330 | 12 |
| Run | 36,992 | 289 | 13 |
| Sit | 44,032 | 344 | 10 |
| Stand | 52,480 | 410 | 16 |
| Walk | 45,696 | 357 | 13 |


## Method

1. **Windows**: non-overlapping windows of 128 samples (no leftover samples).
2. **Features** (12 per window): max, min, mean and standard deviation (`ddof=1`) of X, Y and Z.
3. **Models**: KNN, SVM (RBF or linear) and Decision Tree, each in a pipeline with standardisation
   fitted on the training data only.
4. **Tuning**: grid search with 5-fold stratified cross-validation on the training split only,
   scored by balanced accuracy. The test split is used once.
5. **Protocols** (10 runs each, seed 42 + run number):
   - **A. Random**: stratified random 75/25 split of windows (requested by the supervisor).
   - **B. Segment-grouped**: whole segments are assigned to train or test, class by class, so no
     segment appears on both sides. Tuning folds are grouped too.
6. **Metrics**: accuracy, balanced accuracy, weighted/macro F1, MCC and macro one-vs-rest ROC-AUC.
7. **Statistical tests** (`stat_tests.py`):
   - permutation test against chance level (random protocol);
   - paired Wilcoxon signed-rank and Nadeau-Bengio corrected t-test between models, Holm-adjusted;
   - Friedman test over the three models.

   The runs reuse the same windows, so they are not independent and p-values are approximate.

## Project structure

```
.
├── app.py                  # Streamlit dashboard
├── data_utils.py           # load / clean MAT and CSV files
├── feature_extraction.py   # 128-sample windows, 12 features, segment group_id
├── ml_experiments.py       # models, nested tuning, repeated evaluation (random / grouped)
├── stat_tests.py           # permutation test, paired tests, Friedman
├── eda_visualizations.py   # exploratory plots used by the dashboard
├── model_visualizations.py # model comparison plots used by the dashboard
├── requirements.txt
├── data/                   # put Fall.mat ... Walk.mat here (git-ignored)
├── results/                # CSV outputs of the experiments reported above
```





## Installation

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux / macOS:      source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Usage

### Dashboard

```bash
python -m streamlit run app.py
```

Set **Dataset folder** in the sidebar to the folder holding the `.mat` files (or upload them),
then pick an **Evaluation protocol** (random or segment-grouped) and run the experiments from the
*ML Experiments* tab. Results, p-values and CSV downloads are in the *Model Comparison* and
*Downloads* tabs. The default dataset path in `app.py` is a Windows path: change `DEFAULT_PATH`
for your machine.

### From Python

```python
import pandas as pd
from ml_experiments import repeated_random_split_experiments

features = pd.read_csv("results/features_128.csv")   # 1,920 windows, 12 features + metadata

result = repeated_random_split_experiments(
    features,
    n_runs=10,                 # at least 10; 100 is possible for a research run
    split_mode="grouped",      # "random" (supervisor protocol) or "grouped"
    n_permutations=100,        # permutation test, random protocol only
)

print(result["summary"])           # mean and std of every metric per model
print(result["pairwise_tests"])    # Wilcoxon + corrected t-test (Holm)
print(result["chance_test"])       # None in grouped mode
```

To rebuild the features from the raw files:

```python
import glob
from data_utils import load_uploaded_files, clean_raw_data
from feature_extraction import extract_window_features, validate_feature_table

raw, _ = load_uploaded_files(sorted(glob.glob("data/*.mat")))
clean, report = clean_raw_data(raw)
features = validate_feature_table(extract_window_features(clean))
```
