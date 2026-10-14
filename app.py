"""
app.py

CDTA Shimmer Activity Recognition — Classical Machine Learning

Run:
    python -m streamlit run app.py
"""

from pathlib import Path
import io
import traceback
import time
from typing import cast

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from data_utils import load_uploaded_files, clean_raw_data
from feature_extraction import (
    extract_window_features,
    validate_feature_table,
)
from ml_experiments import repeated_random_split_experiments


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CDTA Shimmer ML",
    page_icon="📊",
    layout="wide",
)

st.title("CDTA Shimmer Activity Recognition")
st.caption(
    "Data preparation, raw-signal exploration, "
    "and classical machine-learning experiments"
)

DEFAULT_DATASET_PATH = (
    r"C:\Users\HP\Downloads\shimmer dataset_cdta"
    r"\shimmer dataset_cdta\mat"
)


# =========================================================
# 2. UTILITY FUNCTIONS
# IMPORTANT: Define these BEFORE the application calls them.
# =========================================================

def clear_experiment_results():
    """Clear results when the dataset or settings change."""
    st.session_state["experiment_output"] = None
    st.session_state["experiment_signature"] = None


def make_activity_counts(df):
    """Count samples per activity."""
    if df is None or df.empty or "activity" not in df.columns:
        return pd.DataFrame(columns=["Activity", "Samples"])

    return (
        df["activity"]
        .astype(str)
        .value_counts()
        .sort_index()
        .rename_axis("Activity")
        .reset_index(name="Samples")
    )


def make_quality_report(raw_df, cleaned_df):
    """Compare basic quality indicators before and after cleaning."""
    rows = []

    for name, df in [
        ("Raw data", raw_df),
        ("Cleaned data", cleaned_df),
    ]:
        rows.append(
            {
                "Dataset": name,
                "Rows": len(df),
                "Columns": len(df.columns),
                "Missing cells": int(df.isna().sum().sum()),
                "Duplicate rows": int(df.duplicated().sum()),
            }
        )

    return pd.DataFrame(rows)


def make_timestamp_report(df):
    """Summarize timestamp differences without assuming their units."""
    columns = [
        "Source file",
        "Rows",
        "Missing timestamps",
        "Non-increasing differences",
        "Median positive difference",
        "Large positive differences",
    ]

    if df is None or df.empty or "timestamp" not in df.columns:
        return pd.DataFrame(columns=columns)

    rows = []

    for source_file, group in df.groupby(
        "source_file",
        sort=False,
        dropna=False,
    ):
        timestamp = pd.to_numeric(
            group["timestamp"],
            errors="coerce",
        )

        values = timestamp.to_numpy(dtype=float)
        valid = values[np.isfinite(values)]

        if len(valid) >= 2:
            differences = np.diff(valid)
            positive = differences[differences > 0]

            median_positive = (
                float(np.median(positive))
                if len(positive)
                else np.nan
            )

            non_increasing = int(np.sum(differences <= 0))

            large_positive = (
                int(np.sum(differences > 5 * median_positive))
                if np.isfinite(median_positive)
                and median_positive > 0
                else 0
            )
        else:
            median_positive = np.nan
            non_increasing = 0
            large_positive = 0

        rows.append(
            {
                "Source file": str(source_file),
                "Rows": len(group),
                "Missing timestamps": int(timestamp.isna().sum()),
                "Non-increasing differences": non_increasing,
                "Median positive difference": median_positive,
                "Large positive differences": large_positive,
            }
        )

    return pd.DataFrame(rows, columns=columns)


def make_confusion_matrix_figure(
    matrix,
    labels,
    title,
    normalize=False,
):
    """Create a confusion-matrix visualization."""
    matrix = np.asarray(matrix, dtype=float)

    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(
            matrix,
            row_sums,
            out=np.zeros_like(matrix),
            where=row_sums != 0,
        )

    fig, ax = plt.subplots(figsize=(9, 7))

    image = ax.imshow(matrix, interpolation="nearest")
    fig.colorbar(image, ax=ax)

    ax.set(
        title=title,
        xlabel="Predicted activity",
        ylabel="True activity",
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    threshold = matrix.max() / 2 if matrix.size else 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            cell_text = (
                f"{matrix[i, j]:.2f}"
                if normalize
                else str(int(matrix[i, j]))
            )

            ax.text(
                j,
                i,
                cell_text,
                ha="center",
                va="center",
                color=(
                    "white"
                    if matrix[i, j] > threshold
                    else "black"
                ),
                fontsize=8,
            )

    fig.tight_layout()
    return fig


def dataframe_to_csv(df):
    """Convert a DataFrame into downloadable CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")


def figure_to_png_bytes(signal_data):
    """
    Convert the selected activity's accelerometer signals to PNG bytes.

    This function is defined BEFORE the Raw Signals tab uses it,
    preventing the previous NameError.
    """
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 8),
        sharex=True,
    )

    columns = [
        ("accel_x", "Accelerometer X"),
        ("accel_y", "Accelerometer Y"),
        ("accel_z", "Accelerometer Z"),
    ]

    x_values = signal_data["sample_index"].to_numpy()

    for ax, (column, title) in zip(axes, columns):
        if column not in signal_data.columns:
            ax.set_title(f"{title}: column unavailable")
            continue

        y_values = pd.to_numeric(
            signal_data[column],
            errors="coerce",
        )

        ax.plot(x_values, y_values, linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Original sample index")
    fig.tight_layout()

    buffer = io.BytesIO()

    try:
        fig.savefig(
            buffer,
            format="png",
            dpi=150,
            bbox_inches="tight",
        )
    finally:
        plt.close(fig)

    buffer.seek(0)
    return buffer.getvalue()


def show_exception(error):
    """Display an error and its traceback."""
    st.error(f"{type(error).__name__}: {error}")

    with st.expander("Technical details"):
        st.code(traceback.format_exc())


# =========================================================
# 3. SESSION STATE
# =========================================================

SESSION_DEFAULTS = {
    "raw_data": None,
    "cleaned_data": None,
    "data_report": None,
    "feature_table": None,
    "experiment_output": None,
    "loaded_source": None,
    "experiment_signature": None,
}

for key, value in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# 4. SIDEBAR
# =========================================================

st.sidebar.header("Project settings")

st.sidebar.markdown(
    """
    **Baseline experiment**
    - Window size: 128 samples
    - Window duration: approximately 2.56 seconds at 50 Hz
    - Features: 12 accelerometer statistics
    - Models: KNN, SVM, Decision Tree
    """
)

st.sidebar.divider()

dataset_path = st.sidebar.text_input(
    "Dataset folder",
    value=DEFAULT_DATASET_PATH,
)

uploaded_files = st.sidebar.file_uploader(
    "Or upload MAT/CSV files",
    type=["mat", "csv"],
    accept_multiple_files=True,
)

st.sidebar.divider()

n_runs = st.sidebar.number_input(
    "Repeated experiments",
    min_value=10,
    max_value=100,
    value=10,
    step=10,
)

test_size = st.sidebar.slider(
    "Test set proportion",
    min_value=0.10,
    max_value=0.40,
    value=0.25,
    step=0.05,
)

random_state = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    max_value=999999,
    value=42,
    step=1,
)

inner_cv_splits = st.sidebar.selectbox(
    "Hyperparameter-tuning CV folds",
    options=[2, 3, 4, 5],
    index=3,
)

st.sidebar.caption(
    "More runs and CV folds require more computation."
)


# =========================================================
# 5. LOAD DATA
# =========================================================

st.subheader("1. Load dataset")

load_col, reset_col = st.columns([3, 1])

with load_col:
    load_clicked = st.button(
        "Load dataset",
        type="primary",
        width="stretch",
    )

with reset_col:
    reset_clicked = st.button(
        "Reset results",
        width="stretch",
    )

if reset_clicked:
    clear_experiment_results()
    st.success("Experiment results cleared.")

if load_clicked:
    try:
        with st.spinner("Loading and preprocessing recordings..."):

            if uploaded_files:
                source_files = uploaded_files
                source_description = "Uploaded files"
            else:
                folder = Path(dataset_path)

                if not folder.exists():
                    raise FileNotFoundError(
                        f"Dataset folder does not exist: {folder}"
                    )

                source_files = sorted(
                    list(folder.glob("*.mat"))
                    + list(folder.glob("*.csv"))
                )

                if not source_files:
                    raise FileNotFoundError(
                        "No .mat or .csv files were found in the folder."
                    )

                source_description = str(folder)

            raw_df, initial_report = cast(
                tuple[pd.DataFrame, dict],
                load_uploaded_files(source_files),
            )

            cleaned_df, cleaning_report = clean_raw_data(
                raw_df
            )

            feature_df = extract_window_features(
                cleaned_df,
                window_size=128,
            )

            feature_df = validate_feature_table(feature_df)

            st.session_state["raw_data"] = raw_df
            st.session_state["cleaned_data"] = cleaned_df
            st.session_state["data_report"] = {
                "initial": initial_report,
                "cleaning": cleaning_report,
            }
            st.session_state["feature_table"] = feature_df
            st.session_state["loaded_source"] = source_description

            clear_experiment_results()

        st.success(
            f"Loaded {len(raw_df):,} raw rows and extracted "
            f"{len(feature_df):,} valid windows."
        )

    except Exception as error:
        show_exception(error)


raw_df = st.session_state["raw_data"]
cleaned_df = st.session_state["cleaned_data"]
feature_df = st.session_state["feature_table"]

if raw_df is None or cleaned_df is None or feature_df is None:
    st.info(
        "Select your dataset folder in the sidebar and click "
        "**Load dataset** to begin."
    )
    st.stop()


# =========================================================
# 6. TABS
# No Feature Analysis tab.
# =========================================================

overview_tab, quality_tab, signals_tab, models_tab, downloads_tab = (
    st.tabs(
        [
            "Overview",
            "Data Quality",
            "Raw Signals",
            "Model Experiments",
            "Downloads",
        ]
    )
)


# =========================================================
# 7. OVERVIEW
# =========================================================

with overview_tab:
    st.header("Dataset overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Raw samples", f"{len(raw_df):,}")
    col2.metric("Cleaned samples", f"{len(cleaned_df):,}")
    col3.metric("Valid windows", f"{len(feature_df):,}")
    col4.metric(
        "Activity classes",
        cleaned_df["activity"].nunique(),
    )

    st.write(
        "**Dataset source:**",
        st.session_state["loaded_source"],
    )

    st.subheader("Experimental workflow")

    st.markdown(
        """
        1. Load and clean Shimmer recordings.
        2. Inspect data quality and raw signals.
        3. Extract 12 baseline statistical features from each
           128-sample accelerometer window.
        4. Tune and evaluate KNN, SVM, and Decision Tree.
        5. Compare repeated-split performance and download results.
        """
    )

    st.subheader("Samples per activity")

    counts = make_activity_counts(cleaned_df)

    if not counts.empty:
        st.bar_chart(
            counts.set_index("Activity")["Samples"],
            width="stretch",
        )

        st.dataframe(
            counts,
            width="stretch",
            hide_index=True,
        )

    st.subheader("Windows per activity")

    window_counts = (
        feature_df["activity"]
        .value_counts()
        .sort_index()
        .rename_axis("Activity")
        .reset_index(name="Windows")
    )

    st.dataframe(
        window_counts,
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "Only complete 128-sample windows are retained. "
        "Incomplete trailing samples are discarded."
    )


# =========================================================
# 8. DATA QUALITY
# =========================================================

with quality_tab:
    st.header("Data quality and preprocessing")

    st.subheader("Before and after cleaning")

    quality_df = make_quality_report(raw_df, cleaned_df)

    st.dataframe(
        quality_df,
        width="stretch",
        hide_index=True,
    )

    st.subheader("Missing values after cleaning")

    missing_counts = (
        cleaned_df.isna()
        .sum()
        .rename_axis("Column")
        .reset_index(name="Missing values")
    )

    st.dataframe(
        missing_counts,
        width="stretch",
        hide_index=True,
    )

    st.subheader("Duplicate rows")

    st.write(
        "Duplicate rows in cleaned data:",
        int(cleaned_df.duplicated().sum()),
    )

    st.caption(
        "Repeated sensor readings are not automatically deleted. "
        "Their meaning should be checked against the dataset documentation."
    )

    st.subheader("Timestamp checks")

    timestamp_report = make_timestamp_report(raw_df)

    if timestamp_report.empty:
        st.info("No timestamp column is available.")
    else:
        st.dataframe(
            timestamp_report,
            width="stretch",
            hide_index=True,
        )

    st.warning(
        "Timestamp differences are descriptive only. Verify the "
        "timestamp units and acquisition behavior before interpreting "
        "gaps as recording boundaries."
    )

    st.subheader("Preprocessing report")

    reports = st.session_state.get("data_report")

    if reports is not None:
        st.json(
            {
                "initial_report": str(reports["initial"]),
                "cleaning_report": str(reports["cleaning"]),
            }
        )


# =========================================================
# 9. RAW SIGNALS
# =========================================================

with signals_tab:
    st.header("Raw accelerometer signals")

    st.write(
        "Inspect the first 1,000 samples of each activity."
    )

    available_activities = sorted(
        cleaned_df["activity"].astype(str).unique()
    )

    selected_activity = st.selectbox(
        "Select activity",
        options=available_activities,
    )

    activity_data = cleaned_df[
        cleaned_df["activity"].astype(str) == selected_activity
    ].copy()

    available_files = sorted(
        activity_data["source_file"].astype(str).unique()
    )

    selected_file = st.selectbox(
        "Select source recording",
        options=available_files,
    )

    signal_data = activity_data[
        activity_data["source_file"].astype(str) == selected_file
    ].copy()

    signal_data = signal_data.sort_values(
        "sample_index",
        kind="stable",
    ).head(1000)

    if signal_data.empty:
        st.warning("No samples are available for this selection.")

    else:
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(13, 9),
            sharex=True,
        )

        x_values = signal_data["sample_index"].to_numpy()

        signal_columns = [
            ("accel_x", "Accelerometer X"),
            ("accel_y", "Accelerometer Y"),
            ("accel_z", "Accelerometer Z"),
        ]

        for ax, (column, title) in zip(axes, signal_columns):
            if column not in signal_data.columns:
                ax.set_title(f"{title}: column unavailable")
                continue

            values = pd.to_numeric(
                signal_data[column],
                errors="coerce",
            )

            ax.plot(
                x_values,
                values,
                linewidth=0.8,
            )

            ax.set_title(title)
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Original sample index")
        fig.tight_layout()

        st.pyplot(fig, width="stretch")
        plt.close(fig)

        st.caption(
            f"Showing {len(signal_data):,} samples from "
            f"{selected_file}."
        )

        # FIXED: figure_to_png_bytes is defined above this code.
        st.download_button(
            "Download signal plot (PNG)",
            data=figure_to_png_bytes(signal_data),
            file_name=(
                f"{selected_activity}_raw_signals.png"
            ),
            mime="image/png",
            width="stretch",
        )

    st.subheader("Sample data")

    st.dataframe(
        signal_data.head(20),
        width="stretch",
        hide_index=True,
    )


# =========================================================
# 10. MODEL EXPERIMENTS
# =========================================================

with models_tab:
    st.header("Classical machine-learning experiments")

    st.markdown(
        """
        The models use the same repeated stratified train/test splits.

        Hyperparameters are tuned using cross-validation on the
        training partition only. The held-out test partition is
        not used for tuning.
        """
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Repeated runs", int(n_runs))
    col2.metric("Test proportion", f"{test_size:.0%}")
    col3.metric("CV folds", int(inner_cv_splits))

    st.warning(
        "Evaluation limitation: random splitting of windows can put "
        "related samples from the same recording into both training "
        "and test sets. This may inflate performance estimates. "
        "Discuss recording-aware validation with your supervisor."
    )

    signature = (
        len(feature_df),
        tuple(feature_df.columns),
        int(n_runs),
        float(test_size),
        int(random_state),
        int(inner_cv_splits),
    )

    if (
        st.session_state["experiment_signature"] is not None
        and st.session_state["experiment_signature"] != signature
    ):
        st.info(
            "The settings changed. Run the experiments again "
            "to update the results."
        )

    run_clicked = st.button(
        "Run model experiments",
        type="primary",
        width="stretch",
    )

    if run_clicked:
        try:
            start_time = time.perf_counter()

            with st.spinner(
                "Tuning and evaluating KNN, SVM, and Decision Tree..."
            ):
                output = repeated_random_split_experiments(
                    feature_df=feature_df,
                    n_runs=int(n_runs),
                    test_size=float(test_size),
                    random_state=int(random_state),
                    inner_cv_splits=int(inner_cv_splits),
                )

            elapsed = time.perf_counter() - start_time

            st.session_state["experiment_output"] = output
            st.session_state["experiment_signature"] = signature

            st.success(
                f"Completed {n_runs} runs for all three models "
                f"in {elapsed:.1f} seconds."
            )

        except Exception as error:
            st.session_state["experiment_output"] = None
            show_exception(error)

    output = st.session_state.get("experiment_output")

    if output is not None:
        st.divider()

        st.subheader("Model comparison")

        summary_df = output["summary"]

        display_columns = [
            "model",
            "accuracy_mean",
            "accuracy_std",
            "balanced_accuracy_mean",
            "balanced_accuracy_std",
            "precision_weighted_mean",
            "recall_weighted_mean",
            "f1_weighted_mean",
            "f1_weighted_std",
            "f1_macro_mean",
            "mcc_mean",
            "mcc_std",
            "mean_tuning_and_fit_seconds",
        ]

        display_columns = [
            column
            for column in display_columns
            if column in summary_df.columns
        ]

        st.dataframe(
            summary_df[display_columns].round(4),
            width="stretch",
            hide_index=True,
        )

        st.subheader("Balanced accuracy")

        st.bar_chart(
            summary_df.set_index("model")[
                ["balanced_accuracy_mean"]
            ],
            width="stretch",
        )

        st.subheader("Weighted F1-score")

        st.bar_chart(
            summary_df.set_index("model")[
                ["f1_weighted_mean"]
            ],
            width="stretch",
        )

        st.subheader("Per-run results")

        st.dataframe(
            output["results"],
            width="stretch",
            hide_index=True,
        )

        st.subheader("Selected hyperparameters")

        st.dataframe(
            output["best_parameters"],
            width="stretch",
            hide_index=True,
        )

        st.subheader("Confusion matrices")

        st.caption(output["confusion_matrix_note"])

        selected_model = st.selectbox(
            "Select model",
            options=list(output["confusion_matrices"].keys()),
        )

        labels = output["confusion_matrix_labels"]
        matrix = output["confusion_matrices"][selected_model]

        col_a, col_b = st.columns(2)

        with col_a:
            fig = make_confusion_matrix_figure(
                matrix,
                labels,
                f"{selected_model} — accumulated counts",
                normalize=False,
            )
            st.pyplot(fig, width="stretch")
            plt.close(fig)

        with col_b:
            fig = make_confusion_matrix_figure(
                matrix,
                labels,
                f"{selected_model} — normalized",
                normalize=True,
            )
            st.pyplot(fig, width="stretch")
            plt.close(fig)

        st.subheader("Per-class performance")

        per_class_df = output["per_class_metrics"]

        per_class_summary = (
            per_class_df
            .groupby(["model", "class"], as_index=False)
            .agg(
                precision_mean=("precision", "mean"),
                recall_mean=("recall", "mean"),
                f1_mean=("f1_score", "mean"),
            )
        )

        st.dataframe(
            per_class_summary.round(4),
            width="stretch",
            hide_index=True,
        )

        st.subheader("Evaluation limitations")

        st.write(output["evaluation_note"])

        st.caption(
            "Final models are fitted on all available windows after "
            "evaluation. They are intended for future predictions, "
            "not for estimating generalization performance."
        )


# =========================================================
# 11. DOWNLOADS
# =========================================================

with downloads_tab:
    st.header("Download project outputs")

    st.subheader("Data")

    st.download_button(
        "Download cleaned raw data",
        data=dataframe_to_csv(cleaned_df),
        file_name="shimmer_cleaned_data.csv",
        mime="text/csv",
        width="stretch",
    )

    st.download_button(
        "Download baseline features",
        data=dataframe_to_csv(feature_df),
        file_name="shimmer_baseline_features.csv",
        mime="text/csv",
        width="stretch",
    )

    output = st.session_state.get("experiment_output")

    if output is None:
        st.info(
            "Run the experiments to enable result downloads."
        )

    else:
        st.subheader("Experiment results")

        st.download_button(
            "Download per-run metrics",
            data=dataframe_to_csv(output["results"]),
            file_name="model_experiment_results.csv",
            mime="text/csv",
            width="stretch",
        )

        st.download_button(
            "Download model summary",
            data=dataframe_to_csv(output["summary"]),
            file_name="model_summary.csv",
            mime="text/csv",
            width="stretch",
        )

        st.download_button(
            "Download best hyperparameters",
            data=dataframe_to_csv(output["best_parameters"]),
            file_name="best_hyperparameters.csv",
            mime="text/csv",
            width="stretch",
        )

        st.download_button(
            "Download per-class metrics",
            data=dataframe_to_csv(output["per_class_metrics"]),
            file_name="per_class_metrics.csv",
            mime="text/csv",
            width="stretch",
        )

        for model_name, matrix in output[
            "confusion_matrices"
        ].items():
            matrix_df = pd.DataFrame(
                matrix,
                index=output["confusion_matrix_labels"],
                columns=output["confusion_matrix_labels"],
            )

            matrix_df.index.name = "True activity"

            st.download_button(
                f"Download {model_name} confusion matrix",
                data=dataframe_to_csv(
                    matrix_df.reset_index()
                ),
                file_name=(
                    f"{model_name.lower().replace(' ', '_')}_"
                    "confusion_matrix.csv"
                ),
                mime="text/csv",
                width="stretch",
                key=f"download_cm_{model_name}",
            )
