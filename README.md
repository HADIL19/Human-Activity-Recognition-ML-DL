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

At a nominal sampling frequency of 50 Hz, each window represents approximately **2.56 seconds** of activity. The actual sampling frequency should be confirmed from the recording timestamps.

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
| Matthews Correlation Coefficient | Correlation-based classification metric |
| Confusion Matrix | Visualization of classification errors |
| Runtime | Model training and evaluation time |

Results are summarized using the mean and standard deviation across repeated experiments.

**Evaluation limitation:** Repeated random splits can contain overlapping test observations. Therefore, aggregated confusion matrices represent predictions across repetitions rather than independent observations. Window-level random splitting may also overestimate generalization when windows originate from the same recording.

## Results

Mean ± standard deviation over 10 repeated stratified train/test splits:

| Model | Accuracy | Balanced accuracy | Weighted F1 | Macro F1 | MCC | Mean tuning + fit time (s) |
|---|---|---|---|---|---|---|
| **KNN** | **0.9621** ± 0.0089 | **0.9297** ± 0.0240 | **0.9613** ± 0.0094 | **0.9434** | **0.9545** ± 0.0107 | 1.58 |
| Decision Tree | 0.9300 ± 0.0139 | 0.9165 ± 0.0189 | 0.9299 ± 0.0142 | 0.9145 | 0.9159 ± 0.0165 | 2.52 |
| SVM | 0.8842 ± 0.0086 | 0.8938 ± 0.0168 | 0.8825 ± 0.0092 | 0.8955 | 0.8613 ± 0.0103 | **0.78** |

**Main observations**

- KNN scores highest on every quality metric, and it has the highest accuracy in all ten splits. SVM is the lowest in all ten, but it is the fastest to tune and fit.
- For KNN and the Decision Tree, balanced accuracy and macro F1 are lower than overall accuracy, which points to weaker recognition of some minority classes (Fall is the smallest class). SVM shows the opposite pattern.
- Run-to-run variation is small compared with the gaps between models, so the ranking KNN > Decision Tree > SVM is stable across splits.

**Interpretation note:** these scores come from random window-level splits (see the evaluation limitation above), so they are likely optimistic compared with performance on unseen recordings or participants. A recording-aware or participant-aware split is the planned next step.

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
- Visualization of the first 1,000 samples
- Export of signal plots

**Model Experiments**
- KNN, SVM, and Decision Tree comparison
- Configurable experimental settings
- Hyperparameter optimization
- Performance metrics
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
