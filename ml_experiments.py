"""
ml_experiments.py

Classical machine-learning experiments for the CDTA Shimmer dataset.

Models:
    1. K-Nearest Neighbors (KNN)
    2. Support Vector Machine (SVM)
    3. Decision Tree

Features:
    12 baseline statistical accelerometer features.

Evaluation:
    Repeated stratified holdout experiments.
    Hyperparameter tuning inside each training split.
    Accuracy, balanced accuracy, precision, recall, F1-score, MCC,
    confusion matrices, per-class metrics, and execution times.

Important:
    Repeated random splits are useful for assessing variability,
    but their results are not statistically independent.
"""

import time
import warnings
from typing import Any, cast

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    StratifiedShuffleSplit,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# Must match feature_extraction.py.
FEATURE_COLUMNS = [
    "x_max",
    "x_min",
    "x_mean",
    "x_std",
    "y_max",
    "y_min",
    "y_mean",
    "y_std",
    "z_max",
    "z_min",
    "z_mean",
    "z_std",
]

METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "f1_macro",
    "mcc",
]


def prepare_dataset(feature_df):
    """Validate features and encode activity labels."""

    if not isinstance(feature_df, pd.DataFrame):
        raise TypeError("feature_df must be a pandas DataFrame.")

    required = {"activity", *FEATURE_COLUMNS}
    missing = required - set(feature_df.columns)

    if missing:
        raise ValueError(
            "Feature table is missing required columns: "
            f"{sorted(missing)}"
        )

    df = feature_df.copy()

    df["activity"] = df["activity"].astype("string").str.strip()
    df = df.dropna(subset=["activity"])
    df = df[df["activity"] != ""]

    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Drop rows only if every feature is missing.
    # Remaining missing values are handled by the pipeline imputer.
    df = df.dropna(subset=FEATURE_COLUMNS, how="all")

    if df.empty:
        raise ValueError("No valid feature rows remain.")

    class_counts = df["activity"].value_counts()

    if len(class_counts) < 2:
        raise ValueError("At least two activity classes are required.")

    if class_counts.min() < 3:
        raise ValueError(
            "Every activity must have at least three valid windows "
            "for repeated holdout evaluation and cross-validation. "
            f"Class counts: {class_counts.to_dict()}"
        )

    X = df[FEATURE_COLUMNS].reset_index(drop=True)
    y_text = df["activity"].astype(str).to_numpy()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)

    return X, y, label_encoder


def build_models(random_state=42):
    """Create the three classification pipelines."""

    return {
        "KNN": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier()),
            ]
        ),
        "SVM": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def build_parameter_grids():
    """Return valid hyperparameter grids for GridSearchCV."""

    return {
        "KNN": {
            "classifier__n_neighbors": [3, 5, 7, 9, 11],
            "classifier__weights": ["uniform", "distance"],
            "classifier__p": [1, 2],
        },
        "SVM": [
            {
                "classifier__kernel": ["rbf"],
                "classifier__C": [0.1, 1, 10],
                "classifier__gamma": ["scale", 0.01, 0.1],
            },
            {
                "classifier__kernel": ["linear"],
                "classifier__C": [0.1, 1, 10],
            },
        ],
        "Decision Tree": {
            "classifier__criterion": ["gini", "entropy"],
            "classifier__max_depth": [None, 5, 10, 20],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
        },
    }


def calculate_metrics(y_true, y_pred):
    """Calculate classification metrics."""

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "recall_weighted": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "f1_weighted": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def _make_cv(y, random_state, max_splits=5):
    """Create feasible stratified cross-validation folds."""

    _, counts = np.unique(y, return_counts=True)

    if len(counts) < 2:
        raise ValueError("At least two classes are required.")

    n_splits = min(int(max_splits), int(counts.min()))

    if n_splits < 2:
        raise ValueError(
            "Cross-validation requires at least two samples "
            "per class in the training data."
        )

    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )


def _valid_parameter_candidates(
    model_name,
    parameter_grid,
    y_train,
    cv_splits,
):
    """
    Filter KNN neighbor counts and return a valid parameter grid.

    Each parameter value remains a list, as required by GridSearchCV.
    """

    if model_name != "KNN":
        return parameter_grid

    _, counts = np.unique(y_train, return_counts=True)
    actual_splits = min(int(cv_splits), int(counts.min()))

    if actual_splits < 2:
        raise ValueError(
            "KNN tuning requires at least two samples per class."
        )

    smallest_inner_training_size = (
        len(y_train)
        - int(np.ceil(len(y_train) / actual_splits))
    )

    valid_neighbors = [
        n
        for n in parameter_grid["classifier__n_neighbors"]
        if n <= smallest_inner_training_size
    ]

    if not valid_neighbors:
        raise ValueError(
            "No valid KNN neighbor counts remain. "
            "Increase the training set size or reduce the CV folds."
        )

    return {
        "classifier__n_neighbors": valid_neighbors,
        "classifier__weights": parameter_grid["classifier__weights"],
        "classifier__p": parameter_grid["classifier__p"],
    }


def _tune_and_fit(
    model_name,
    model,
    parameter_grid,
    X_train,
    y_train,
    random_state,
    inner_cv_splits=5,
):
    """Tune and fit a model using training data only."""

    cv = _make_cv(
        y_train,
        random_state=random_state,
        max_splits=inner_cv_splits,
    )

    safe_grid = _valid_parameter_candidates(
        model_name=model_name,
        parameter_grid=parameter_grid,
        y_train=y_train,
        cv_splits=cv.get_n_splits(),
    )

    search = GridSearchCV(
        estimator=model,
        param_grid=safe_grid,
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )

    start_time = time.perf_counter()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X_train, y_train)

    fit_seconds = time.perf_counter() - start_time

    return (
        search.best_estimator_,
        search.best_params_,
        float(fit_seconds),
        float(search.best_score_),
    )


def _extract_per_class_metrics(y_true, y_pred, label_encoder):
    """Build robust per-class metric records."""

    class_names = label_encoder.classes_
    class_indices = np.arange(len(class_names))

    report = cast(
        dict[str, Any],
        classification_report(
            y_true,
            y_pred,
            labels=class_indices,
            target_names=class_names.tolist(),
            output_dict=True,
            zero_division=0,
        ),
    )

    records = []

    for class_index, class_name in enumerate(class_names):
        class_report = report[str(class_name)]

        records.append(
            {
                "class": str(class_name),
                "precision": float(class_report["precision"]),
                "recall": float(class_report["recall"]),
                "f1_score": float(class_report["f1-score"]),
                "support": int(class_report["support"]),
            }
        )

    return records


def _fit_final_models(
    X,
    y,
    models,
    parameter_grids,
    random_state,
    inner_cv_splits,
):
    """Tune and fit deployable models using all available windows."""

    final_models = {}
    final_model_parameters = []

    for model_name, model in models.items():
        full_cv = _make_cv(
            y,
            random_state=random_state,
            max_splits=inner_cv_splits,
        )

        safe_grid = _valid_parameter_candidates(
            model_name=model_name,
            parameter_grid=parameter_grids[model_name],
            y_train=y,
            cv_splits=full_cv.get_n_splits(),
        )

        search = GridSearchCV(
            estimator=clone(model),
            param_grid=safe_grid,
            scoring="balanced_accuracy",
            cv=full_cv,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X, y)

        final_models[model_name] = search.best_estimator_

        final_model_parameters.append(
            {
                "model": model_name,
                "best_parameters": repr(search.best_params_),
                "best_cv_balanced_accuracy": float(search.best_score_),
            }
        )

    return final_models, pd.DataFrame(final_model_parameters)


def repeated_random_split_experiments(
    feature_df,
    n_runs=10,
    test_size=0.25,
    random_state=42,
    knn_neighbors=5,
    inner_cv_splits=5,
):
    """
    Compare KNN, SVM, and Decision Tree across repeated stratified splits.

    Hyperparameter tuning happens inside each training partition.
    The test partition is reserved for evaluation.

    Returns
    -------
    dict
        Results, summary statistics, confusion matrices, per-class
        reports, best parameters, label encoder, and final models.
    """

    if not isinstance(n_runs, (int, np.integer)) or n_runs < 10:
        raise ValueError("n_runs must be an integer of at least 10.")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    if inner_cv_splits < 2:
        raise ValueError("inner_cv_splits must be at least 2.")

    X, y, label_encoder = prepare_dataset(feature_df)

    n_classes = len(label_encoder.classes_)

    n_samples = len(y)
    n_test = int(np.ceil(n_samples * test_size))
    n_train = n_samples - n_test

    if n_test < n_classes or n_train < n_classes:
        raise ValueError(
            "The dataset is too small for the selected test size "
            "and number of classes."
        )

    models = build_models(random_state=random_state)
    parameter_grids = build_parameter_grids()

    outer_cv = StratifiedShuffleSplit(
        n_splits=int(n_runs),
        test_size=test_size,
        random_state=random_state,
    )

    results = []
    per_class_reports = []
    best_parameters = []

    confusion_matrices = {
        name: np.zeros((n_classes, n_classes), dtype=np.int64)
        for name in models
    }

    for run_number, (train_indices, test_indices) in enumerate(
        outer_cv.split(X, y),
        start=1,
    ):
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]

        y_train = y[train_indices]
        y_test = y[test_indices]

        for model_name, model in models.items():
            try:
                (
                    fitted_model,
                    best_params,
                    fit_seconds,
                    best_cv_score,
                ) = _tune_and_fit(
                    model_name=model_name,
                    model=model,
                    parameter_grid=parameter_grids[model_name],
                    X_train=X_train,
                    y_train=y_train,
                    random_state=random_state + run_number,
                    inner_cv_splits=inner_cv_splits,
                )

                prediction_start = time.perf_counter()
                y_pred = fitted_model.predict(X_test)
                prediction_seconds = (
                    time.perf_counter() - prediction_start
                )

                metrics = calculate_metrics(y_test, y_pred)

                results.append(
                    {
                        "run": run_number,
                        "model": model_name,
                        **metrics,
                        "best_cv_balanced_accuracy": best_cv_score,
                        "tuning_and_fit_seconds": fit_seconds,
                        "prediction_seconds": float(prediction_seconds),
                        "best_parameters": repr(best_params),
                    }
                )

                confusion_matrices[model_name] += confusion_matrix(
                    y_test,
                    y_pred,
                    labels=np.arange(n_classes),
                )

                class_records = _extract_per_class_metrics(
                    y_test,
                    y_pred,
                    label_encoder,
                )

                for record in class_records:
                    per_class_reports.append(
                        {
                            "run": run_number,
                            "model": model_name,
                            **record,
                        }
                    )

                best_parameters.append(
                    {
                        "run": run_number,
                        "model": model_name,
                        "best_parameters": repr(best_params),
                        "best_cv_balanced_accuracy": best_cv_score,
                    }
                )

            except Exception as exc:
                raise RuntimeError(
                    f"Experiment failed during run {run_number} "
                    f"for {model_name}: {exc}"
                ) from exc

    results_df = pd.DataFrame(results)

    summary_rows = []

    for model_name, group in results_df.groupby("model"):
        row = {"model": model_name}

        for metric in METRIC_COLUMNS:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)

        row["mean_tuning_and_fit_seconds"] = (
            group["tuning_and_fit_seconds"].mean()
        )
        row["mean_prediction_seconds"] = (
            group["prediction_seconds"].mean()
        )

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        "balanced_accuracy_mean",
        ascending=False,
    ).reset_index(drop=True)

    per_class_df = pd.DataFrame(per_class_reports)
    best_parameters_df = pd.DataFrame(best_parameters)

    try:
        final_models, final_model_parameters_df = _fit_final_models(
            X=X,
            y=y,
            models=models,
            parameter_grids=parameter_grids,
            random_state=random_state,
            inner_cv_splits=inner_cv_splits,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Final model fitting failed: {exc}"
        ) from exc

    return {
        "results": results_df,
        "summary": summary_df,
        "confusion_matrices": confusion_matrices,
        "per_class_metrics": per_class_df,
        "best_parameters": best_parameters_df,
        "final_model_parameters": final_model_parameters_df,
        "label_encoder": label_encoder,
        "final_models": final_models,
        "feature_columns": FEATURE_COLUMNS,
        "n_runs": int(n_runs),
        "test_size": test_size,
        "random_state": random_state,
        "confusion_matrix_labels": list(label_encoder.classes_),
        "confusion_matrix_note": (
            "Counts accumulated across repeated test splits. "
            "A window may appear in multiple test splits."
        ),
        "evaluation_note": (
            "Repeated stratified holdout results are useful for "
            "comparing models, but runs overlap and are not "
            "statistically independent. If independent recording or "
            "participant identifiers become available, use a "
            "group-aware evaluation design."
        ),
    }
