# Human Activity Recognition Using Shimmer Sensor Data

### Machine Learning | Wearable Sensors | Signal Processing | Streamlit

An end-to-end machine learning pipeline for **Human Activity Recognition (HAR)** using accelerometer data collected from Shimmer wearable sensors.

This project explores how classical machine learning algorithms can classify human activities from time-series sensor measurements using statistical feature extraction.

Developed as part of a machine learning research internship at the **Centre de Développement des Technologies Avancées (CDTA), Algeria**.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Activities](#activities)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Data Preprocessing](#data-preprocessing)
- [Feature Extraction](#feature-extraction)
- [Machine Learning Models](#machine-learning-models)
- [Experimental Evaluation](#experimental-evaluation)
- [Results](#results)
- [Interactive Streamlit Dashboard](#interactive-streamlit-dashboard)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Future Work](#future-work)

## Project Overview

Human Activity Recognition (HAR) involves identifying physical activities from sensor measurements, such as acceleration and angular velocity.

This project implements a complete workflow covering:

- Sensor data loading and preprocessing
- Exploratory data analysis and signal visualization
- Time-series segmentation using non-overlapping, fixed-length windows
- Statistical feature extraction
- Machine learning model training and hyperparameter optimization
- Repeated experimental evaluation
- Interactive visualization through a Streamlit dashboard

The objective is to investigate the effectiveness of classical machine learning techniques for recognizing different human activities from wearable sensor data.

## Activities

The dataset contains seven activity classes:

| Activity | Description |
|---|---|
| Fall | Falling activity |
| Jump | Jumping |
| Lay | Lying down |
| Run | Running |
| Sit | Sitting |
| Stand | Standing |
| Walk | Walking |

The signals differ clearly between activities. Fall appears as sharp isolated spikes separated by quiet stretches, Jump as frequent high-amplitude bursts, and Sit stays close to zero. These differences in spread and extreme values are what the maximum, minimum and standard-deviation features capture.

## Machine Learning Pipeline

```text
Shimmer Sensor Data (.mat / .csv)
              |
              v
       Data Loading
              |
              v
    Data Quality Checks
              |
              v
      Data Preprocessing
              |
              v
    Signal Visualization
              |
              v
   Window Segmentation
       (128 samples)
              |
              v
  Statistical Feature Extraction
       (12 features)
              |
              v
      Train / Test Split
              |
              v
   Hyperparameter Optimization
              |
              v
     Model Training
     /     |      \
   KNN    SVM   Decision Tree
     \     |      /
              v
       Model Evaluation
              |
              v
     Results Visualization
       (Streamlit App)
```

## Data Preprocessing

The preprocessing pipeline includes data loading, numerical validation, missing-value inspection, and preparation of accelerometer signals.

The three accelerometer axes are used:

- **X-axis:** `accel_x`
- **Y-axis:** `accel_y`
- **Z-axis:** `accel_z`

### Window Segmentation

Continuous sensor signals are divided into non-overlapping windows of **128 samples**.

Windows are built only within continuous runs: missing sample indices, non-increasing timestamps, and timestamp steps over 1.5 times the recording's median positive step start a new run. This prevents windows from crossing timestamp resets or acquisition gaps; incomplete run tails are dropped.

The supplied MAT recordings contain a timestamp followed by accelerometer X, Y, and Z in columns 2-4; later sensor channels are not used for baseline features. The observed median timestamp step is about **19.53 timestamp units**. If those units are milliseconds, that corresponds to approximately 51.2 Hz and a 128-sample window is about 2.50 seconds. The physical timestamp unit should be confirmed before reporting duration in seconds.

Each window is transformed into a feature vector for classification.

## Feature Extraction

For every window, four statistical features are calculated independently for each accelerometer axis.

| Feature | X-axis | Y-axis | Z-axis |
|---|---|---|---|
| Maximum | x_max | y_max | z_max |
| Minimum | x_min | y_min | z_min |
| Mean | x_mean | y_mean | z_mean |
| Standard deviation | x_std | y_std | z_std |

This produces:

**3 axes × 4 statistical features = 12 features per window.**

The resulting feature vectors are used as inputs to the classification algorithms.

## Machine Learning Models

Three supervised machine learning algorithms are implemented.

### 1. K-Nearest Neighbors (KNN)

A distance-based classification algorithm that predicts activity labels using neighboring training samples.

The pipeline includes missing-value imputation, feature standardization, and hyperparameter tuning.

### 2. Support Vector Machine (SVM)

A supervised learning algorithm that identifies decision boundaries between activity classes.

The implementation explores linear and radial basis function (RBF) kernels.

### 3. Decision Tree

A tree-based classification algorithm that learns decision rules from statistical features.

Hyperparameter optimization is performed to control tree complexity.

### Hyperparameter Optimization

Models are optimized using **GridSearchCV** with stratified cross-validation.

The primary optimization metric is balanced accuracy.

Preprocessing steps are included within the training pipelines to avoid fitting them directly on test data.

## Experimental Evaluation

To assess model performance across different train/test partitions, the experiments use repeated stratified random splits.

The number of repetitions is configurable, with a minimum of **10 runs**.

For each run:

1. Generate a stratified train/test split.
2. Perform hyperparameter optimization on the training data.
3. Train the selected model.
4. Evaluate predictions on the test partition.
5. Record classification metrics and execution time.

### Evaluation Metrics

The following metrics are calculated:

| Metric | Purpose |
|---|---|
| Accuracy | Overall classification accuracy |
| Balanced Accuracy | Average recall across classes |
| Weighted Precision | Precision accounting for class support |
| Weighted Recall | Recall accounting for class support |
| Weighted F1-score | F1-score accounting for class support |
| Macro F1-score | Average F1-score across classes |
| Macro AUC-ROC (one-vs-rest) | Mean area under the ROC curve across activity classes |
| Matthews Correlation Coefficient | Correlation-based classification metric |
| Confusion Matrix | Visualization of classification errors |
| Runtime | Model training and evaluation time |

Results are summarized using the mean and standard deviation across repeated experiments. The dashboard also reports corrected resampled t-tests for balanced accuracy: each model versus the 1/number-of-classes chance level, plus pairwise model comparisons. Holm adjustment is applied across the six tests.

These p-values are approximate because repeated random splits reuse observations, and windows from one recording may appear in both training and test partitions. Treat them as exploratory evidence. A recording-aware or participant-aware evaluation is needed before making strong generalization claims.

**Evaluation limitation:** Repeated random splits can contain overlapping test observations. Therefore, aggregated confusion matrices represent predictions across repetitions rather than independent observations. Window-level random splitting may also overestimate generalization when windows originate from the same recording.

## Results

The results below use the seven supplied MAT recordings, 128-sample windows split at timestamp discontinuities, 10 stratified random splits, a 25% test fraction, and random seed 42. Hyperparameters were tuned inside each training split using five-fold cross-validation.

| Model | Accuracy | Balanced accuracy | Weighted F1 | Macro F1 | Macro AUC-ROC | MCC | Mean tuning + fit time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| KNN | 0.9651 ± 0.0088 | 0.9261 ± 0.0195 | 0.9643 ± 0.0090 | 0.9407 ± 0.0162 | 0.9815 ± 0.0096 | 0.9580 ± 0.0106 | 1.38 |
| Decision Tree | 0.9237 ± 0.0166 | 0.9042 ± 0.0221 | 0.9237 ± 0.0166 | 0.9059 ± 0.0198 | 0.9547 ± 0.0136 | 0.9083 ± 0.0201 | 4.94 |
| SVM | 0.8828 ± 0.0117 | 0.8910 ± 0.0205 | 0.8809 ± 0.0122 | 0.8929 ± 0.0165 | 0.9854 ± 0.0033 | 0.8595 ± 0.0142 | 6.13 |

Corrected resampled t-tests against the balanced-accuracy chance level of 1/7 were significant for all three models after Holm adjustment (adjusted p-values: KNN 1.27 × 10⁻¹², SVM 2.56 × 10⁻¹², Decision Tree 3.40 × 10⁻¹²). No pairwise model difference remained significant after Holm adjustment; KNN vs SVM was closest (adjusted p = 0.0559). These tests are exploratory because the 10 random splits reuse windows from the same recordings.

The corrected pipeline produces **1,856 complete windows**: Fall 44, Jump 138, Lay 318, Run 279, Sit 334, Stand 397, and Walk 346. Fall is the smallest class, so inspect per-class recall and MCC alongside accuracy.

## Interactive Streamlit Dashboard

The project includes an interactive application built with Streamlit.

### Available Sections

**Overview**
- Dataset summary
- Activity distribution
- Number of extracted windows

**Data Quality**
- Missing-value inspection
- Duplicate inspection
- Timestamp analysis
- Preprocessing reports

**Raw Signals**
- Visualization of accelerometer signals
- Activity selection
- Grid of the first 1,000 samples for every activity across X, Y, and Z
- Elapsed timestamp-based horizontal axis; timestamp resets and large gaps are compressed for plotting
- Export of signal plots

**Model Experiments**
- KNN, SVM, and Decision Tree comparison
- Configurable experimental settings
- Hyperparameter optimization
- Performance metrics
- Corrected resampled significance tests and Holm-adjusted p-values
- Confusion matrices
- Per-class evaluation

**Downloads**
- Cleaned data
- Extracted features
- Experimental results
- Model comparison summaries
- Confusion matrices

## Project Structure

```text
shimmer-activity-recognition-ml/
│
├── app.py
│   └── Streamlit application
│
├── data_utils.py
│   └── Data loading and preprocessing
│
├── feature_extraction.py
│   └── Window segmentation and feature extraction
│
├── ml_experiments.py
│   └── Model training, tuning, and evaluation
│
├── requirements.txt
│   └── Python dependencies
│
├── .gitignore
│
└── README.md
```

## Technologies

**Programming Language**
- Python

**Machine Learning**
- Scikit-learn
- NumPy
- Pandas

**Scientific Computing**
- SciPy

**Visualization**
- Matplotlib
- Streamlit

**Development Tools**
- Visual Studio Code
- Git
- GitHub

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/HADIL19/shimmer-activity-recognition-ml.git
cd shimmer-activity-recognition-ml
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the application

```bash
python -m streamlit run app.py
```

The application will be accessible locally through the URL displayed in the terminal.

### Dataset Availability

The original Shimmer sensor dataset is not included in this repository.

Users must provide compatible `.mat` or `.csv` files through the application or configure an accessible local dataset directory.

## Future Work

Potential extensions include:

- Transfer learning for sensor-based activity recognition
- Signal transformation into image-based representations
- Accelerometer magnitude computation
- Deep learning model experimentation
- Improved recording-aware evaluation
- Comparative analysis between classical ML and transfer learning approaches

These extensions are planned research directions and are not part of the current implementation.

## Research Context

**Institution:** Centre de Développement des Technologies Avancées (CDTA), Algeria

**Internship Area:** Machine Learning and Transfer Learning

**Research Supervisor:** Dr. Oussama Kerdjidj

**Academic Institution:** University of Science and Technology Houari Boumediene (USTHB)

**Department:** Artificial Intelligence and Data Science

## Author

**Hadil Khelif**

Master's Student in Bioinformatics  
Department of Artificial Intelligence and Data Science  
University of Science and Technology Houari Boumediene (USTHB)

**Interests:** Machine Learning Engineering, Artificial Intelligence, Software Engineering, and Biomedical AI.

---

*This repository documents an academic research internship project. Experimental results should be interpreted in the context of the dataset, preprocessing assumptions, and evaluation methodology.*
