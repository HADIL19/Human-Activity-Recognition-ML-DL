"""
CLASSICAL MACHINE LEARNING
==========================

Models:
    1. K-Nearest Neighbors
    2. Support Vector Machine
       - Linear
       - RBF
       - Polynomial
    3. Decision Tree

Evaluation:
    - Repeated stratified train/test splits
    - GridSearchCV
    - Accuracy
    - Macro precision
    - Macro recall
    - Macro F1
    - Matthews Correlation Coefficient (MCC)
"""

import time

import numpy as np
import pandas as pd

from sklearn.base import clone

from sklearn.model_selection import (
    StratifiedShuffleSplit,
    StratifiedKFold,
    GridSearchCV
)

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# ---------------------------------------------------------
# 1. DEFINE THE MODELS AND HYPERPARAMETER GRIDS
# ---------------------------------------------------------

def build_model_grids(random_state=42):
    """
    Create model pipelines and their hyperparameter search spaces.

    Preprocessing is included inside each pipeline so that the
    imputer and scaler are fitted only on the training data.
    """

    models = {

        # -------------------------------------------------
        # KNN
        # -------------------------------------------------

        "KNN": {

            "estimator": Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                ),
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "classifier",
                    KNeighborsClassifier()
                )
            ]),

            "parameters": {

                "classifier__n_neighbors": [
                    3, 5, 7, 9, 11, 15
                ],

                "classifier__weights": [
                    "uniform",
                    "distance"
                ],

                # p=1: Manhattan distance
                # p=2: Euclidean distance
                "classifier__p": [1, 2]
            }
        },


        # -------------------------------------------------
        # SVM
        # -------------------------------------------------

        "SVM": {

            "estimator": Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                ),
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "classifier",
                    SVC()
                )
            ]),

            # Each dictionary defines a separate search space.
            # Polynomial-specific parameters are only searched
            # when the polynomial kernel is selected.
            "parameters": [

                # Linear kernel
                {
                    "classifier__kernel": ["linear"],

                    "classifier__C": [
                        0.1, 1, 10, 100
                    ]
                },

                # Radial Basis Function kernel
                {
                    "classifier__kernel": ["rbf"],

                    "classifier__C": [
                        0.1, 1, 10, 100
                    ],

                    "classifier__gamma": [
                        "scale",
                        "auto",
                        0.01,
                        0.1
                    ]
                },

                # Polynomial kernel
                {
                    "classifier__kernel": ["poly"],

                    "classifier__C": [
                        0.1, 1, 10, 100
                    ],

                    "classifier__gamma": [
                        "scale",
                        "auto"
                    ],

                    "classifier__degree": [
                        2, 3, 4
                    ]
                }
            ]
        },


        # -------------------------------------------------
        # DECISION TREE
        # -------------------------------------------------

        "Decision Tree": {

            "estimator": Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                ),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        random_state=random_state
                    )
                )
            ]),

            "parameters": {

                "classifier__criterion": [
                    "gini",
                    "entropy",
                    "log_loss"
                ],

                "classifier__max_depth": [
                    None, 5, 10, 15, 20
                ],

                "classifier__min_samples_split": [
                    2, 5, 10
                ],

                "classifier__min_samples_leaf": [
                    1, 2, 4
                ],

                "classifier__class_weight": [
                    None,
                    "balanced"
                ]
            }
        }
    }

    return models


# ---------------------------------------------------------
# 2. CALCULATE EVALUATION METRICS
# ---------------------------------------------------------

def calculate_metrics(y_true, y_pred):
    """
    Calculate the metrics used to compare models.

    Macro averaging gives every activity class equal weight.
    MCC is our primary comparison metric.
    """

    return {

        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "precision_macro": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "recall_macro": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "f1_macro": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "mcc": matthews_corrcoef(
            y_true,
            y_pred
        )
    }


# ---------------------------------------------------------
# 3. RUN REPEATED EXPERIMENTS
# ---------------------------------------------------------

def run_repeated_experiments(
    feature_df,
    feature_columns,
    n_splits=10,
    test_size=0.2,
    tune_hyperparameters=True,
    cv_folds=5,
    random_state=42,
    selected_models=None,
    progress_callback=None
):
    """
    Run multiple stratified train/test experiments.

    All selected models use the same train/test split for each run.

    Hyperparameter tuning, when enabled, takes place only inside
    the training split. The held-out test set is not used to choose
    the best hyperparameters.

    Returns:
        results_df: individual metrics for each model and run
        summary_df: aggregate performance across runs
        details: final-run predictions, reports and confusion matrices

    LIMITATION:
    Repeated random window-level splits are not independent tests.
    If windows from the same original recording appear in both
    training and testing, the scores may be optimistic.
    """

    if n_splits < 2:

        raise ValueError(
            "Use at least two repeated splits."
        )

    if not 0 < test_size < 0.5:

        raise ValueError(
            "Test proportion must be between 0 and 0.5."
        )

    df = feature_df.copy().reset_index(drop=True)

    feature_columns = [
        column
        for column in feature_columns
        if column in df.columns
    ]

    if not feature_columns:

        raise ValueError(
            "No valid model feature columns were selected."
        )

    # X contains the predictors.
    X = df[feature_columns].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # y contains the class labels.
    y = df["activity"].astype(str)

    # Stratified splitting needs at least two examples per class.
    class_counts = y.value_counts()

    small_classes = class_counts[
        class_counts < 2
    ]

    if not small_classes.empty:

        raise ValueError(
            "Every class needs at least two examples. "
            f"Insufficient classes: {small_classes.to_dict()}"
        )

    model_grids = build_model_grids(
        random_state=random_state
    )

    if selected_models:

        model_grids = {
            name: config
            for name, config in model_grids.items()
            if name in selected_models
        }

    if not model_grids:

        raise ValueError(
            "Select at least one model."
        )

    # Generate all splits before training any models.
    splitter = StratifiedShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state
    )

    splits = list(
        splitter.split(X, y)
    )

    result_rows = []
    details = {}

    total_steps = len(splits) * len(model_grids)
    completed = 0

    # -----------------------------------------------------
    # LOOP OVER THE REPEATED SPLITS
    # -----------------------------------------------------

    for run_number, (train_idx, test_idx) in enumerate(
        splits,
        start=1
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Reduce the number of CV folds if a training class
        # has fewer examples than the requested fold count.
        smallest_class = int(
            y_train.value_counts().min()
        )

        actual_cv_folds = min(
            cv_folds,
            smallest_class
        )

        if actual_cv_folds < 2:

            raise ValueError(
                "Not enough examples per class in a training "
                "split for cross-validation."
            )

        inner_cv = StratifiedKFold(
            n_splits=actual_cv_folds,
            shuffle=True,
            random_state=random_state + run_number
        )

        # -------------------------------------------------
        # TRAIN EACH SELECTED MODEL
        # -------------------------------------------------

        for model_name, config in model_grids.items():

            start_time = time.perf_counter()

            estimator = clone(
                config["estimator"]
            )

            best_params = {}
            cv_best_mcc = np.nan

            if tune_hyperparameters:

                # MCC is used to choose hyperparameters because
                # your dataset has imbalanced activity classes.
                search = GridSearchCV(
                    estimator=estimator,
                    param_grid=config["parameters"],
                    scoring="matthews_corrcoef",
                    cv=inner_cv,
                    n_jobs=-1,
                    refit=True,
                    error_score="raise"
                )

                search.fit(
                    X_train,
                    y_train
                )

                fitted_model = search.best_estimator_

                best_params = search.best_params_

                cv_best_mcc = float(
                    search.best_score_
                )

            else:

                fitted_model = estimator.fit(
                    X_train,
                    y_train
                )

            elapsed_seconds = (
                time.perf_counter() - start_time
            )

            # Predict only on the held-out test split.
            y_pred = fitted_model.predict(
                X_test
            )

            metrics = calculate_metrics(
                y_test,
                y_pred
            )

            result_rows.append({

                "run": run_number,

                "model": model_name,

                **metrics,

                "cv_best_mcc": cv_best_mcc,

                "fit_seconds": elapsed_seconds,

                "best_params": repr(best_params),

                "train_rows": len(train_idx),

                "test_rows": len(test_idx)
            })

            # Save the last run's detailed outputs for the app.
            if run_number == len(splits):

                labels = sorted(
                    y.unique()
                )

                details[model_name] = {

                    "model": fitted_model,

                    "y_test": y_test.to_numpy(),

                    "y_pred": np.asarray(y_pred),

                    "labels": labels,

                    "confusion_matrix": confusion_matrix(
                        y_test,
                        y_pred,
                        labels=labels
                    ),

                    "classification_report": classification_report(
                        y_test,
                        y_pred,
                        labels=labels,
                        output_dict=True,
                        zero_division=0
                    ),

                    "best_params": best_params
                }

            completed += 1

            if progress_callback:

                progress_callback(
                    completed / total_steps,
                    f"Run {run_number}/{len(splits)} — {model_name}"
                )

    # -----------------------------------------------------
    # 4. SUMMARIZE THE REPEATED EXPERIMENTS
    # -----------------------------------------------------

    results_df = pd.DataFrame(
        result_rows
    )

    summary_df = (
        results_df
        .groupby("model")
        .agg(

            runs=("run", "count"),

            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),

            precision_macro_mean=(
                "precision_macro", "mean"
            ),

            recall_macro_mean=(
                "recall_macro", "mean"
            ),

            f1_macro_mean=(
                "f1_macro", "mean"
            ),

            f1_macro_std=(
                "f1_macro", "std"
            ),

            mcc_mean=("mcc", "mean"),
            mcc_std=("mcc", "std"),

            mcc_min=("mcc", "min"),
            mcc_max=("mcc", "max"),

            fit_seconds_mean=(
                "fit_seconds", "mean"
            )
        )
        .reset_index()
        .sort_values(
            "mcc_mean",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return results_df, summary_df, details