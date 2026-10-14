# Integrate visualizations into your existing app.py

Copy `eda_visualizations.py` and `model_visualizations.py` into the same folder as `app.py`.

## A. Add imports after your existing imports

```python
import eda_visualizations as eda
import model_visualizations as mv
```

## B. Replace the current tabs assignment

```python
(
    overview_tab, quality_tab, eda_tab, signals_tab,
    models_tab, comparison_tab, downloads_tab,
) = st.tabs([
    "Overview", "Data Quality", "EDA", "Raw Signals",
    "Model Experiments", "Model Comparison", "Downloads",
])
```

## C. Add the following section before `# 9. RAW SIGNALS`

```python
# =========================================================
# EXPLORATORY DATA ANALYSIS
# =========================================================
with eda_tab:
    st.header("Exploratory Data Analysis")

    def show_eda_figure(fig, title, key):
        st.subheader(title)
        st.pyplot(fig, width="stretch")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
        st.download_button("Download PNG", buf.getvalue(),
                           file_name=f"{key}.png", mime="image/png", key=f"eda_{key}")
        plt.close(fig)

    show_eda_figure(eda.activity_counts(cleaned_df, feature_df),
                    "Samples and windows", "activity_counts")
    show_eda_figure(eda.class_proportions(cleaned_df),
                    "Activity proportions", "activity_proportions")
    show_eda_figure(eda.missing_values(raw_df),
                    "Missing values before cleaning", "missing_values")
    show_eda_figure(eda.missing_pattern(raw_df),
                    "Missingness pattern", "missing_pattern")

    axis = st.selectbox("Accelerometer axis", eda.AXES, key="eda_axis")
    activity_options = sorted(cleaned_df['activity'].astype(str).unique())
    activities = st.multiselect("Activities to display", activity_options,
                                default=activity_options, key="eda_activities")
    if activities:
        show_eda_figure(eda.axis_distributions(cleaned_df, axis, activities, kind='hist'),
                        "Activity histograms", "histogram")
        show_eda_figure(eda.axis_distributions(cleaned_df, axis, activities, kind='kde'),
                        "Density curves", "density")
        show_eda_figure(eda.activity_boxplot(cleaned_df, axis, kind='box'),
                        "Activity boxplots", "boxplot")
        show_eda_figure(eda.activity_boxplot(cleaned_df, axis, kind='violin'),
                        "Activity violin plots", "violin")
        show_eda_figure(eda.signal_comparison(cleaned_df, activities, axis=axis),
                        "Side-by-side raw signals", "signal_comparison")
        show_eda_figure(eda.signal_comparison(cleaned_df, activities, axis='magnitude'),
                        "Magnitude comparison (exploration only)", "magnitude_comparison")

    if 'timestamp' in raw_df.columns and 'source_file' in raw_df.columns:
        source = st.selectbox("Timestamp recording", sorted(raw_df['source_file'].astype(str).unique()),
                              key="eda_timestamp_source")
        show_eda_figure(eda.timestamp_differences(raw_df, source),
                        "Timestamp differences (units unverified)", "timestamp_differences")

    st.caption("Plots describe raw sensor measurements. They do not alter the 12-feature ML baseline.")
```

## D. Add the following section before `# 11. DOWNLOADS`

```python
# =========================================================
# MODEL COMPARISON
# =========================================================
with comparison_tab:
    st.header("Model Comparison Visualizations")
    comparison_output = st.session_state.get('experiment_output')
    if comparison_output is None:
        st.info("Run the experiments in Model Experiments first.")
    else:
        summary = comparison_output['summary']
        runs = comparison_output['results']

        def show_model_figure(fig, title, key):
            st.subheader(title)
            st.pyplot(fig, width="stretch")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
            st.download_button("Download PNG", buf.getvalue(),
                               file_name=f"{key}.png", mime="image/png", key=f"model_{key}")
            plt.close(fig)

        show_model_figure(mv.metric_means(summary), "All evaluation metrics", "all_metrics")
        metric_options = [m for m in mv.METRICS if m in runs.columns and m+'_mean' in summary.columns and m+'_std' in summary.columns]
        if metric_options:
            metric = st.selectbox('Comparison metric', metric_options, index=metric_options.index('balanced_accuracy') if 'balanced_accuracy' in metric_options else 0)
            show_model_figure(mv.metric_errorbars(summary, metric), "Mean ± standard deviation", "metric_errorbars")
            show_model_figure(mv.run_distributions(runs, metric, 'box'), "Across-run boxplots", "run_boxplot")
            show_model_figure(mv.run_distributions(runs, metric, 'violin'), "Across-run violin plots", "run_violin")
            show_model_figure(mv.run_lines(runs, metric), "Metric per run", "run_lines")
        show_model_figure(mv.metrics_heatmap(summary), "Metrics heatmap", "metrics_heatmap")
        if 'mean_tuning_and_fit_seconds' in summary.columns:
            show_model_figure(mv.runtime_chart(summary), "Tuning and fitting runtime", "runtime")
        per_class = comparison_output['per_class_metrics']
        for metric in ['precision', 'recall', 'f1_score']:
            show_model_figure(mv.per_class_heatmap(per_class, metric),
                              f"Per-class {metric}", f"per_class_{metric}")
        show_model_figure(mv.parameter_frequency(comparison_output['best_parameters']),
                          "Hyperparameter selection", "hyperparameter_frequency")
        st.warning('Repeated random splits reuse windows across runs; results are not independent. '
                   'Windows from the same recording may appear in both train and test sets.')
```

## E. Run

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

**Notes:** The plots use only existing DataFrames and model outputs. They do not modify training or evaluation. The `parameter_frequency` chart auto-detects common column names; if your actual best-parameter column differs, update that helper. Some plotting APIs require recent Matplotlib versions.
