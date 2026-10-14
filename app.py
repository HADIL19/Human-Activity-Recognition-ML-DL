"""
SHIMMER ACTIVITY RECOGNITION
============================

Task 1:
    Data exploration, cleaning and feature extraction.

Task 2:
    Classical machine learning, hyperparameter tuning
    and repeated model evaluation.

Run:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from data_utils import (
    load_uploaded_files,
    clean_raw_data
)

from feature_extraction import (
    extract_window_features,
    validate_feature_table
)

from ml_experiments import (
    run_repeated_experiments
)


# ---------------------------------------------------------
# 1. STREAMLIT CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Shimmer Activity Recognition",
    page_icon="📈",
    layout="wide"
)

st.title(
    "Shimmer Sensor Activity Recognition"
)

st.caption(
    "Task 1: Data analysis and feature engineering | "
    "Task 2: Classical machine learning"
)


# ---------------------------------------------------------
# 2. SIDEBAR SETTINGS
# ---------------------------------------------------------

with st.sidebar:

    st.header("Dataset")

    uploaded_files = st.file_uploader(
        "Upload MAT or CSV files",
        type=["mat", "csv"],
        accept_multiple_files=True
    )

    st.divider()

    st.header("Feature extraction")

    window_size = st.number_input(
        "Window size in samples",
        min_value=16,
        max_value=4096,
        value=128,
        step=16
    )

    sampling_rate = st.number_input(
        "Sampling rate in Hz",
        min_value=1,
        max_value=1000,
        value=50
    )

    include_extended = st.checkbox(
        "Include extended features",
        value=False
    )

    st.divider()

    st.header("Model experiments")

    n_splits = st.number_input(
        "Repeated train/test splits",
        min_value=2,
        max_value=100,
        value=10
    )

    test_size = st.slider(
        "Test set proportion",
        min_value=0.1,
        max_value=0.4,
        value=0.2,
        step=0.05
    )

    tune_hyperparameters = st.checkbox(
        "Tune hyperparameters",
        value=True
    )

    selected_models = st.multiselect(
        "Select models",
        options=[
            "KNN",
            "SVM",
            "Decision Tree"
        ],
        default=[
            "KNN",
            "SVM",
            "Decision Tree"
        ]
    )


# ---------------------------------------------------------
# 3. ASK THE USER TO UPLOAD DATA
# ---------------------------------------------------------

if not uploaded_files:

    st.info(
        "Upload your Shimmer recordings to begin."
    )

    st.markdown(
        """
        ### Recommended workflow

        1. Upload your raw activity recordings.
        2. Inspect the signal plots and data quality.
        3. Extract 128-sample windows.
        4. Inspect the feature table.
        5. Run KNN, SVM and Decision Tree.
        6. Compare the repeated-run metrics.
        7. Download the results for your internship report.
        """
    )

    st.stop()


# ---------------------------------------------------------
# 4. LOAD THE DATA
# ---------------------------------------------------------

try:

    uploaded_df, data_type = load_uploaded_files(
        uploaded_files
    )

except Exception as error:

    st.error(
        f"Unable to load the uploaded data: {error}"
    )

    st.stop()


# ---------------------------------------------------------
# 5. PREPARE RAW DATA OR FEATURE DATA
# ---------------------------------------------------------

raw_df = None
cleaning_report = None

if data_type == "raw":

    # Clean raw sensor data.
    raw_df, cleaning_report = clean_raw_data(
        uploaded_df
    )

    # Recreate a sequential sample index within each recording.
    raw_df["sample_index"] = (
        raw_df.groupby("source_file").cumcount()
    )

    # Calculate magnitude.
    raw_df["magnitude"] = np.sqrt(
        raw_df["accel_x"] ** 2
        + raw_df["accel_y"] ** 2
        + raw_df["accel_z"] ** 2
    )

    # Extract the features.
    try:

        feature_df = extract_window_features(
            raw_df,
            window_size=int(window_size),
            include_extended=include_extended
        )

    except Exception as error:

        st.error(
            f"Feature extraction failed: {error}"
        )

        st.stop()

else:

    # If the uploaded CSV already contains features,
    # validate it rather than extracting features again.
    raw_df = None

    try:

        feature_df, feature_columns = (
            validate_feature_table(uploaded_df)
        )

    except Exception as error:

        st.error(
            f"Invalid feature table: {error}"
        )

        st.stop()

    st.info(
        "A precomputed feature table was detected. "
        "Raw-signal plots are available only for raw sensor data."
    )


# Validate the final feature table.
feature_df, feature_columns = validate_feature_table(
    feature_df
)


# ---------------------------------------------------------
# 6. ORGANIZE THE APP INTO TABS
# ---------------------------------------------------------

tabs = st.tabs([
    "Overview",
    "Task 1 — Signals",
    "Task 1 — Data Quality",
    "Task 1 — Features",
    "Task 2 — Models",
    "Task 2 — Feature Analysis",
    "Export"
])


# =========================================================
# TAB 1: OVERVIEW
# =========================================================

with tabs[0]:

    st.header("Dataset overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Raw samples",
        f"{len(raw_df):,}" if raw_df is not None else "N/A"
    )

    col2.metric(
        "Feature windows",
        f"{len(feature_df):,}"
    )

    col3.metric(
        "Activity classes",
        int(feature_df["activity"].nunique())
    )

    col4.metric(
        "Numeric features",
        len(feature_columns)
    )

    st.subheader("Activity distribution")

    counts = (
        feature_df["activity"]
        .value_counts()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(9, 4))

    sns.barplot(
        x=counts.index,
        y=counts.values,
        ax=ax
    )

    ax.set_xlabel("Activity")
    ax.set_ylabel("Number of windows")
    ax.set_title("Number of windows per activity")

    plt.xticks(rotation=35)
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Feature table preview")

    st.dataframe(
        feature_df.head(20),
        use_container_width=True
    )

    if cleaning_report is not None:

        st.subheader("Cleaning summary")

        st.json(cleaning_report)


# =========================================================
# TAB 2: TASK 1 — SIGNAL VISUALIZATION
# =========================================================

with tabs[1]:

    st.header("Sensor signals")

    if raw_df is None:

        st.warning(
            "Upload raw sensor files to inspect the signals."
        )

    else:

        activities = sorted(
            raw_df["activity"].astype(str).unique()
        )

        chosen_activity = st.selectbox(
            "Choose an activity",
            activities
        )

        number_samples = st.slider(
            "Number of samples to plot",
            min_value=100,
            max_value=1000,
            value=1000,
            step=100
        )

        selected = (
            raw_df[
                raw_df["activity"].astype(str)
                == chosen_activity
            ]
            .head(number_samples)
        )

        if selected.empty:

            st.warning(
                "No samples were found for this activity."
            )

        else:

            fig, axes = plt.subplots(
                4,
                1,
                figsize=(12, 9),
                sharex=True
            )

            # Plot acceleration along each axis.
            for ax, column, title in zip(
                axes[:3],
                ["accel_x", "accel_y", "accel_z"],
                [
                    "Acceleration X",
                    "Acceleration Y",
                    "Acceleration Z"
                ]
            ):

                ax.plot(
                    selected["sample_index"],
                    selected[column],
                    linewidth=0.8
                )

                ax.set_ylabel(title)
                ax.grid(True, alpha=0.25)

            # Plot the magnitude signal.
            axes[3].plot(
                selected["sample_index"],
                selected["magnitude"],
                linewidth=0.8
            )

            axes[3].set_ylabel("Magnitude")
            axes[3].set_xlabel("Sample index")
            axes[3].grid(True, alpha=0.25)

            fig.suptitle(
                f"{chosen_activity}: first {len(selected)} samples"
            )

            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

            st.latex(
                r"Mag = \sqrt{x^2 + y^2 + z^2}"
            )

            st.caption(
                "Inspect the activities individually. "
                "Unexpected patterns may indicate sensor issues, "
                "label errors or genuine differences in movement."
            )


# =========================================================
# TAB 3: TASK 1 — DATA QUALITY
# =========================================================

with tabs[2]:

    st.header("Data quality diagnostics")

    if raw_df is None:

        st.warning(
            "Raw sensor data is required for these diagnostics."
        )

    else:

        st.subheader("Timestamp analysis")

        if (
            "timestamp" in raw_df.columns
            and raw_df["timestamp"].notna().any()
        ):

            timestamp_df = raw_df.dropna(
                subset=["timestamp"]
            ).copy()

            timestamp_df = timestamp_df.sort_values(
                ["source_file", "sample_index"]
            )

            # Calculate time differences within each recording.
            timestamp_df["timestamp_delta"] = (
                timestamp_df.groupby("source_file")["timestamp"]
                .diff()
            )

            deltas = timestamp_df[
                "timestamp_delta"
            ].dropna()

            if not deltas.empty:

                st.dataframe(
                    deltas.describe().to_frame(
                        "timestamp_delta"
                    ),
                    use_container_width=True
                )

                st.write(
                    f"Median timestamp interval: "
                    f"{deltas.median():.4g} timestamp units"
                )

                st.caption(
                    "If timestamps are in milliseconds, an interval "
                    "near 20 ms suggests approximately 50 Hz. "
                    "Confirm the timestamp unit before interpreting "
                    "this as the true sampling rate."
                )

                fig, ax = plt.subplots(figsize=(10, 3))

                ax.plot(
                    deltas.to_numpy(),
                    linewidth=0.7
                )

                ax.set_title(
                    "Timestamp differences within recordings"
                )

                ax.set_xlabel("Sample interval index")
                ax.set_ylabel("Timestamp difference")
                ax.grid(True, alpha=0.25)

                fig.tight_layout()

                st.pyplot(fig)
                plt.close(fig)

            else:

                st.warning(
                    "There are insufficient timestamps to calculate intervals."
                )

        else:

            st.warning(
                "No valid timestamp column was found."
            )

        st.subheader("Sensor statistics")

        sensor_columns = [
            "accel_x",
            "accel_y",
            "accel_z",
            "magnitude"
        ]

        st.dataframe(
            raw_df[sensor_columns].describe().T,
            use_container_width=True
        )

        st.subheader("Potential outliers")

        # The IQR rule flags values for investigation.
        # It does not prove that these observations are errors.
        outlier_results = []

        for column in sensor_columns:

            q1 = raw_df[column].quantile(0.25)
            q3 = raw_df[column].quantile(0.75)

            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            number_outliers = int(
                (
                    (raw_df[column] < lower_bound)
                    | (raw_df[column] > upper_bound)
                ).sum()
            )

            outlier_results.append({

                "feature": column,

                "lower_bound": lower_bound,

                "upper_bound": upper_bound,

                "potential_outliers": number_outliers
            })

        st.dataframe(
            pd.DataFrame(outlier_results),
            use_container_width=True
        )

        st.caption(
            "Falls and jumps can naturally produce extreme acceleration. "
            "Do not delete these observations simply because the IQR "
            "rule identifies them."
        )


# =========================================================
# TAB 4: TASK 1 — FEATURE EXTRACTION
# =========================================================

with tabs[3]:

    st.header("Feature engineering")

    window_duration = (
        int(window_size) / float(sampling_rate)
    )

    st.write(
        f"Window size: **{int(window_size)} samples**"
    )

    st.write(
        f"Assumed sampling rate: **{sampling_rate} Hz**"
    )

    st.write(
        f"Window duration: **{window_duration:.3f} seconds**"
    )

    st.write(
        f"Number of extracted features: **{len(feature_columns)}**"
    )

    st.subheader("Model feature columns")

    st.code(
        ", ".join(feature_columns)
    )

    st.subheader("Extracted feature table")

    st.dataframe(
        feature_df.head(50),
        use_container_width=True
    )

    # Check whether any feature rows still have invalid values.
    invalid_rows = int(
        feature_df[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .isna()
        .any(axis=1)
        .sum()
    )

    st.metric(
        "Rows with invalid model features",
        invalid_rows
    )

    st.download_button(
        "Download feature table",
        data=feature_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="shimmer_features.csv",
        mime="text/csv"
    )


# =========================================================
# TAB 5: TASK 2 — TRAINING AND EVALUATION
# =========================================================

with tabs[4]:

    st.header("Classical machine learning")

    st.markdown(
        """
        We evaluate KNN, SVM and Decision Tree.

        **Primary metric: MCC.**

        We also report accuracy, macro precision, macro recall
        and macro F1 because a single metric does not fully
        describe a multiclass classifier.
        """
    )

    st.subheader("Class distribution")

    st.dataframe(
        feature_df["activity"]
        .value_counts()
        .rename_axis("activity")
        .reset_index(name="windows"),
        use_container_width=True
    )

    st.warning(
        "Evaluation limitation: repeated random window-level splits "
        "may place windows from the same original recording in both "
        "training and testing. This can inflate the scores. If possible, "
        "evaluate on entirely separate recordings or participants."
    )

    run_experiments = st.button(
        "Run machine-learning experiments",
        type="primary"
    )

    if run_experiments:

        if not selected_models:

            st.error(
                "Select at least one model in the sidebar."
            )

        else:

            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(value, message):

                progress_bar.progress(
                    min(1.0, max(0.0, value))
                )

                status_text.text(message)

            try:

                results_df, summary_df, details = (
                    run_repeated_experiments(

                        feature_df=feature_df,

                        feature_columns=feature_columns,

                        n_splits=int(n_splits),

                        test_size=float(test_size),

                        tune_hyperparameters=(
                            tune_hyperparameters
                        ),

                        cv_folds=5,

                        random_state=42,

                        selected_models=selected_models,

                        progress_callback=update_progress
                    )
                )

                # Save results in the session so changing tabs
                # does not immediately remove the experiment output.
                st.session_state["results_df"] = results_df

                st.session_state["summary_df"] = summary_df

                st.session_state["model_details"] = details

                st.success(
                    "Experiments completed successfully."
                )

            except Exception as error:

                st.error(
                    f"An error occurred during training: {error}"
                )

    # Display the most recent experiment results.
    if "summary_df" in st.session_state:

        summary_df = st.session_state["summary_df"]

        results_df = st.session_state["results_df"]

        details = st.session_state["model_details"]

        st.subheader("Model comparison")

        st.dataframe(
            summary_df.round(4),
            use_container_width=True
        )

        # Visual comparison of average MCC.
        fig, ax = plt.subplots(figsize=(8, 4))

        sns.barplot(
            data=summary_df,
            x="model",
            y="mcc_mean",
            ax=ax
        )

        ax.set_title(
            "Mean MCC across repeated splits"
        )

        ax.set_xlabel("Model")
        ax.set_ylabel("Mean MCC")
        ax.set_ylim(-1, 1)

        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Results for every run")

        st.dataframe(
            results_df.round(4),
            use_container_width=True
        )

        st.subheader("Detailed results from the final run")

        for model_name, detail in details.items():

            with st.expander(model_name):

                st.write("Best parameters")

                st.json(
                    detail["best_params"]
                )

                # Display the confusion matrix.
                fig, ax = plt.subplots(figsize=(7, 5))

                sns.heatmap(
                    detail["confusion_matrix"],
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=detail["labels"],
                    yticklabels=detail["labels"],
                    ax=ax
                )

                ax.set_xlabel("Predicted activity")
                ax.set_ylabel("True activity")

                ax.set_title(
                    f"{model_name}: final-run confusion matrix"
                )

                plt.xticks(rotation=35)

                fig.tight_layout()

                st.pyplot(fig)
                plt.close(fig)

                st.write(
                    "Per-class classification report"
                )

                report_df = pd.DataFrame(
                    detail["classification_report"]
                ).T

                st.dataframe(
                    report_df.round(4),
                    use_container_width=True
                )

        # Export the detailed results.
        st.download_button(
            "Download repeated-run metrics",
            data=results_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="classical_ml_repeated_runs.csv",
            mime="text/csv"
        )

        st.download_button(
            "Download summary metrics",
            data=summary_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="classical_ml_summary.csv",
            mime="text/csv"
        )

    else:

        st.info(
            "Choose your models and experiment settings, "
            "then click the training button."
        )


# =========================================================
# TAB 6: TASK 2 — FEATURE ANALYSIS
# =========================================================

with tabs[5]:

    st.header("Feature analysis")

    st.caption(
        "These plots are exploratory. Any feature selection used "
        "for final performance evaluation should be performed "
        "inside the training data, not using the held-out test labels."
    )

    chosen_features = st.multiselect(
        "Select features to inspect",
        options=feature_columns,
        default=feature_columns[
            :min(12, len(feature_columns))
        ]
    )

    if chosen_features:

        selected_feature = st.selectbox(
            "Feature to visualize",
            chosen_features
        )

        fig, ax = plt.subplots(figsize=(10, 4))

        for activity, group in feature_df.groupby(
            "activity"
        ):

            ax.hist(
                group[selected_feature].dropna(),
                bins=25,
                alpha=0.45,
                label=str(activity)
            )

        ax.set_title(
            f"{selected_feature} distribution by activity"
        )

        ax.set_xlabel(selected_feature)
        ax.set_ylabel("Count")

        ax.legend()

        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Feature correlation")

        if len(chosen_features) >= 2:

            correlation_matrix = feature_df[
                chosen_features
            ].corr()

            fig, ax = plt.subplots(
                figsize=(
                    max(8, len(chosen_features) * 0.6),
                    max(6, len(chosen_features) * 0.5)
                )
            )

            sns.heatmap(
                correlation_matrix,
                annot=False,
                cmap="coolwarm",
                center=0,
                ax=ax
            )

            ax.set_title(
                "Correlation between selected features"
            )

            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

        # ANOVA feature ranking is descriptive only.
        # It is not the model-comparison significance test
        # that we have postponed.
        st.subheader("Exploratory feature ranking")

        try:

            from sklearn.feature_selection import f_classif

            X_rank = feature_df[
                chosen_features
            ].replace(
                [np.inf, -np.inf],
                np.nan
            )

            valid = (
                X_rank.notna().all(axis=1)
                & feature_df["activity"].notna()
            )

            scores, p_values = f_classif(
                X_rank.loc[valid],
                feature_df.loc[valid, "activity"].astype(str)
            )

            ranking_df = pd.DataFrame({

                "feature": chosen_features,

                "anova_f_score": scores,

                "anova_p_value": p_values
            }).sort_values(
                "anova_f_score",
                ascending=False
            )

            st.dataframe(
                ranking_df.round(6),
                use_container_width=True
            )

            st.caption(
                "These ANOVA p-values describe feature/class associations. "
                "They do not establish that one classifier is significantly "
                "better than another."
            )

        except Exception as error:

            st.warning(
                f"Feature ranking could not be calculated: {error}"
            )


# =========================================================
# TAB 7: EXPORT
# =========================================================

with tabs[6]:

    st.header("Export project outputs")

    st.write(
        "Download the feature table and model results for your report."
    )

    st.download_button(
        "Download extracted features",
        data=feature_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="shimmer_features.csv",
        mime="text/csv"
    )

    if "results_df" in st.session_state:

        st.download_button(
            "Download all experiment results",
            data=st.session_state["results_df"].to_csv(
                index=False
            ).encode("utf-8"),
            file_name="classical_ml_repeated_runs.csv",
            mime="text/csv",
            key="export_all_runs"
        )

        st.download_button(
            "Download experiment summary",
            data=st.session_state["summary_df"].to_csv(
                index=False
            ).encode("utf-8"),
            file_name="classical_ml_summary.csv",
            mime="text/csv",
            key="export_summary"
        )