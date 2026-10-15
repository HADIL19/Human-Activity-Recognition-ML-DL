"""CDTA Shimmer HAR interactive research dashboard. Run: python -m streamlit run app.py"""
from pathlib import Path
import io
import time
import traceback
import json
from typing import cast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from data_utils import load_uploaded_files, clean_raw_data
from feature_extraction import extract_window_features, validate_feature_table
from ml_experiments import repeated_random_split_experiments
import eda_visualizations as eda
import model_visualizations as mv

st.set_page_config(page_title='CDTA Shimmer ML', page_icon='📊', layout='wide')
st.title('CDTA Shimmer Activity Recognition')
st.caption('Interactive sensor exploration · Data quality · Classical ML · Model comparison')
DEFAULT_PATH = r'C:\Users\HP\Downloads\shimmer dataset_cdta\shimmer dataset_cdta\mat'


def show_error(exc):
    st.error(f'{type(exc).__name__}: {exc}')
    with st.expander('Traceback'):
        st.code(traceback.format_exc())


def csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')


def show_plot(fig, filename, caption=None):
    """Render a plot IN the app and offer an optional PNG export."""
    if caption:
        st.caption(caption)
    st.pyplot(fig, width='stretch')
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=160, bbox_inches='tight')
    st.download_button('⬇ Download this chart (PNG)', buffer.getvalue(),
                       file_name=filename, mime='image/png', key=f'png_{filename}')
    plt.close(fig)


def quality_summary(raw, cleaned):
    return pd.DataFrame([{
        'Dataset': label, 'Rows': len(df), 'Columns': len(df.columns),
        'Missing cells': int(df.isna().sum().sum()),
        'Duplicate rows': int(df.duplicated().sum()),
    } for label, df in [('Raw', raw), ('Cleaned', cleaned)]])


def missing_values_summary(raw, cleaned):
    columns = list(dict.fromkeys([*raw.columns, *cleaned.columns]))
    raw_missing = raw.isna().sum()
    cleaned_missing = cleaned.isna().sum()
    return pd.DataFrame({
        'Column': columns,
        'Raw missing values': [int(raw_missing.get(column, 0)) for column in columns],
        'Cleaned missing values': [
            int(cleaned_missing.get(column, 0)) for column in columns
        ],
    })


def timestamp_summary(df):
    if not {'timestamp', 'source_file'}.issubset(df.columns):
        return pd.DataFrame()
    rows = []
    for source, group in df.groupby('source_file', sort=False, dropna=False):
        ts = pd.to_numeric(group['timestamp'], errors='coerce').to_numpy(dtype=float)
        d = np.diff(ts)
        d = d[np.isfinite(d)]
        positive = d[d > 0]
        median = float(np.median(positive)) if len(positive) else np.nan
        rows.append({'Source file': str(source), 'Rows': len(group),
                     'Missing timestamps': int(np.sum(~np.isfinite(ts))),
                     'Non-increasing differences': int(np.sum(d <= 0)),
                     'Median positive difference': median,
                     'Large positive differences (>5× median)':
                     int(np.sum(d > 5 * median)) if np.isfinite(median) and median > 0 else 0})
    return pd.DataFrame(rows)


def confusion_plot(matrix, labels, model, normalized=False):
    matrix = np.asarray(matrix, dtype=float)
    if normalized:
        sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, sums, out=np.zeros_like(matrix), where=sums != 0)
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix, cmap='Blues', vmin=0)
    fig.colorbar(image, ax=ax)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel('Predicted activity')
    ax.set_ylabel('True activity')
    ax.set_title(f'{model}: ' + ('row-normalized' if normalized else 'accumulated counts'))
    threshold = matrix.max() / 2 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f'{matrix[i,j]:.2f}' if normalized else f'{int(matrix[i,j])}',
                    ha='center', va='center', fontsize=8,
                    color='white' if matrix[i,j] > threshold else 'black')
    fig.tight_layout()
    return fig


for key in ['raw_data', 'cleaned_data', 'feature_table', 'data_report',
            'loaded_source', 'experiment_output', 'experiment_signature']:
    st.session_state.setdefault(key, None)

st.sidebar.header('Project settings')
st.sidebar.markdown('**Baseline:** 128 samples/window · 12 X/Y/Z statistics · KNN / SVM / Decision Tree')
folder_path = st.sidebar.text_input('Dataset folder', DEFAULT_PATH)
uploads = st.sidebar.file_uploader('Or upload MAT/CSV files', type=['mat', 'csv'], accept_multiple_files=True)
st.sidebar.divider()
runs = st.sidebar.number_input('Repeated experiments', min_value=10, max_value=100, value=10, step=10)
test_size = st.sidebar.slider('Test set proportion', 0.10, 0.40, 0.25, 0.05)
seed = st.sidebar.number_input('Random seed', min_value=0, max_value=999999, value=42)
folds = st.sidebar.selectbox('Tuning CV folds', [2, 3, 4, 5], index=3)
protocol = st.sidebar.selectbox('Evaluation protocol',
    ['Random window split (supervisor protocol)', 'Segment-grouped split (no leakage between segments)'])
split_mode = 'grouped' if protocol.startswith('Segment') else 'random'
n_perm = st.sidebar.number_input('Permutation test repetitions (0 = skip)', min_value=0, max_value=1000, value=100, step=50,
                                 help='Random protocol only. p-value resolution is 1/(repetitions+1).')
st.sidebar.caption('Repeated grid search can take several minutes.')

st.subheader('Load dataset')
load_col, reset_col = st.columns([3, 1])
with load_col:
    load_clicked = st.button('Load dataset', type='primary', width='stretch')
with reset_col:
    reset_clicked = st.button('Reset experiment results', width='stretch')
if reset_clicked:
    st.session_state['experiment_output'] = None
    st.session_state['experiment_signature'] = None
    st.success('Experiment results cleared.')
if load_clicked:
    try:
        with st.spinner('Loading, cleaning, and extracting windows...'):
            if uploads:
                sources = uploads
                description = 'Uploaded files'
            else:
                folder = Path(folder_path)
                if not folder.is_dir():
                    raise FileNotFoundError(f'Dataset folder does not exist: {folder}')
                sources = sorted([*folder.glob('*.mat'), *folder.glob('*.csv')])
                if not sources:
                    raise FileNotFoundError('No MAT or CSV files in selected folder.')
                description = str(folder)
            raw, initial = cast(tuple[pd.DataFrame, object], load_uploaded_files(sources))
            cleaned, cleaning = clean_raw_data(raw)
            features = validate_feature_table(extract_window_features(cleaned, window_size=128))
            st.session_state.update(raw_data=raw, cleaned_data=cleaned, feature_table=features,
                                    data_report={'initial': initial, 'cleaning': cleaning},
                                    loaded_source=description, experiment_output=None,
                                    experiment_signature=None)
        st.success(f'Loaded {len(raw):,} rows and extracted {len(features):,} windows.')
    except Exception as exc:
        show_error(exc)

raw = st.session_state['raw_data']
cleaned = st.session_state['cleaned_data']
features = st.session_state['feature_table']
if raw is None or cleaned is None or features is None:
    st.info('Select a dataset and click **Load dataset** to start.')
    st.stop()

(overview_tab, quality_tab, signals_tab, models_tab, comparison_tab, downloads_tab) = st.tabs([
    '📊 Overview', '🧹 Data Quality',
    '📈 Raw Signals', '🤖 ML Experiments', '🏆 Model Comparison', '📥 Downloads'
])
activities = sorted(cleaned['activity'].astype(str).unique().tolist())

with overview_tab:
    st.header('Dataset overview')
    a, b, c, d = st.columns(4)
    a.metric('Raw samples', f'{len(raw):,}')
    b.metric('Cleaned samples', f'{len(cleaned):,}')
    c.metric('Valid windows', f'{len(features):,}')
    d.metric('Activity classes', len(activities))
    st.write('**Dataset source:**', st.session_state['loaded_source'])
    show_plot(eda.activity_counts(cleaned, features), 'overview_activity_counts.png')
    c1, c2 = st.columns(2)
    with c1:
        st.subheader('Samples per activity')
        st.dataframe(cleaned['activity'].astype(str).value_counts().sort_index().rename_axis('Activity').reset_index(name='Samples'), hide_index=True, width='stretch')
    with c2:
        st.subheader('Windows per activity')
        st.dataframe(features['activity'].astype(str).value_counts().sort_index().rename_axis('Activity').reset_index(name='Windows'), hide_index=True, width='stretch')
    st.caption('Window duration is approximately 2.56 seconds only if the sampling rate is 50 Hz. Incomplete trailing windows are excluded.')

with quality_tab:
    st.header('Data quality & timestamp diagnostics')
    st.dataframe(quality_summary(raw, cleaned), hide_index=True, width='stretch')
    st.subheader('Missing values by column')
    st.dataframe(
        missing_values_summary(raw, cleaned),
        hide_index=True,
        width='stretch',
    )
    if {'timestamp', 'source_file'}.issubset(raw.columns):
        st.subheader('Timestamp analysis')
        st.dataframe(timestamp_summary(raw), hide_index=True, width='stretch')
        st.warning('Timestamp units have not been verified. Gaps are diagnostic, not automatically treated as recording boundaries.')
    else:
        st.info('Timestamp checks are unavailable because timestamp or source-file data is missing.')
    st.subheader('Preprocessing reports')
    st.json(st.session_state['data_report'])

with signals_tab:
    st.header('Raw accelerometer signals')
    activity = st.selectbox('Activity', activities, key='signal_activity')
    activity_rows = cleaned[cleaned['activity'].astype(str).eq(activity)]
    files = sorted(activity_rows['source_file'].astype(str).unique())
    source = st.selectbox('Source recording', files, key='signal_file')
    n_samples = st.slider('Number of samples', 128, 3000, 1000, 128)
    sub = activity_rows[activity_rows['source_file'].astype(str).eq(source)].sort_values('sample_index', kind='stable').head(n_samples).copy()
    if sub.empty:
        st.warning('No signal rows for this selection.')
    else:
        fig, axs = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
        xyz = sub[eda.AXES].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
        magnitude = np.sqrt(np.sum(xyz ** 2, axis=1))
        x = sub['sample_index'].to_numpy()
        for idx, axis_name in enumerate(eda.AXES):
            axs[idx].plot(x, xyz[:, idx], lw=.85)
            axs[idx].set_ylabel(axis_name)
            axs[idx].grid(alpha=.3)
        axs[3].plot(x, magnitude, lw=.85)
        axs[3].set_ylabel('Magnitude')
        axs[3].set_xlabel('Original sample index')
        axs[3].grid(alpha=.3)
        fig.suptitle(f'{activity} — {source}')
        fig.tight_layout()
        show_plot(fig, 'raw_signal_xyz_magnitude.png', 'Magnitude = √(X² + Y² + Z²). Used for visualization only.')
        st.dataframe(sub.head(20), hide_index=True, width='stretch')
    st.divider()
    st.subheader('Compare activities')
    compared = st.multiselect('Choose activities', activities,
                              default=activities[:min(2, len(activities))], key='signal_compare')
    compare_axis = st.selectbox('Signal to compare', [*eda.AXES, 'magnitude'], key='signal_compare_axis')
    if compared:
        show_plot(eda.signal_comparison(cleaned, compared, compare_axis, samples=n_samples),
                  'activity_signal_comparison.png',
                  'Each activity uses the first available recording; these are descriptive signal comparisons.')

with models_tab:
    st.header('Classical machine learning experiments')
    st.markdown('KNN, SVM and Decision Tree use matched stratified random splits. Hyperparameters are tuned **only on training windows**.')
    c1, c2, c3 = st.columns(3)
    c1.metric('Runs', int(runs))
    c2.metric('Test proportion', f'{test_size:.0%}')
    c3.metric('CV folds', int(folds))
    if split_mode == 'random':
        st.warning('Window-level splits may mix related windows from the same recording across train and test, potentially inflating performance.')
    else:
        st.info('Segment-grouped protocol: windows from the same timestamp-delimited segment stay together in train or test.')
    signature = (len(features), tuple(features.columns), int(runs), float(test_size), int(seed), int(folds), split_mode, int(n_perm))
    if st.session_state['experiment_output'] is not None and st.session_state['experiment_signature'] != signature:
        st.warning('Settings changed. Previous results are displayed until you run experiments again.')
    if st.button('Run model experiments', type='primary', width='stretch'):
        try:
            start = time.perf_counter()
            with st.spinner('Running nested tuning and repeated evaluations...'):
                result = repeated_random_split_experiments(feature_df=features,
                    n_runs=int(runs), test_size=float(test_size),
                    random_state=int(seed), inner_cv_splits=int(folds),
                    split_mode=split_mode, n_permutations=int(n_perm))
            st.session_state['experiment_output'] = result
            st.session_state['experiment_signature'] = signature
            st.success(f'Experiments completed in {time.perf_counter()-start:.1f} seconds.')
        except Exception as exc:
            st.session_state['experiment_output'] = None
            st.session_state['experiment_signature'] = None
            show_error(exc)
    result = st.session_state['experiment_output']
    if result is not None:
        st.subheader('Model summary')
        st.dataframe(result['summary'].round(4), hide_index=True, width='stretch')
        st.subheader('Per-run results')
        st.dataframe(result['results'].round(4), hide_index=True, width='stretch')
        st.subheader('Selected hyperparameters')
        st.dataframe(result['best_parameters'], hide_index=True, width='stretch')
        if 'final_model_parameters' in result:
            with st.expander('Final model parameters (trained on all windows)'):
                st.dataframe(result['final_model_parameters'], hide_index=True, width='stretch')
        st.subheader('Statistical tests')
        if result.get('chance_test') is not None:
            st.markdown('**Against chance** - permutation test (balanced accuracy, 5-fold, labels shuffled)')
            st.dataframe(result['chance_test'].round(4), hide_index=True, width='stretch')
        st.markdown('**Between models** - paired tests over the repeated runs (balanced accuracy)')
        st.dataframe(result['pairwise_tests'].round(4), hide_index=True, width='stretch')
        if result.get('friedman_test'):
            st.caption(f"Friedman test over all models: p = {result['friedman_test']['p_value']:.4g}")
        st.info(result.get('evaluation_note', 'Repeated splits are not independent held-out recordings.'))
        st.caption('See the Model Comparison tab for all interactive performance charts.')

with comparison_tab:
    st.header('Interactive model comparison')
    result = st.session_state['experiment_output']
    if result is None:
        st.info('Run the experiments in **ML Experiments** to populate comparison charts.')
    else:
        summary = result['summary']
        per_run = result['results']
        st.subheader('Comparison across all metrics')
        show_plot(mv.metric_means(summary), 'models_all_metrics.png')
        show_plot(mv.metrics_heatmap(summary), 'models_metric_heatmap.png')
        metric = st.selectbox('Choose evaluation metric', list(mv.METRICS),
                              format_func=lambda m: mv.METRICS[m], key='comparison_metric')
        show_plot(mv.metric_errorbars(summary, metric), f'models_{metric}_errorbars.png',
                  'Error bars show standard deviation across repeated splits (not confidence intervals).')
        a, b = st.columns(2)
        with a:
            show_plot(mv.run_distributions(per_run, metric, kind='box'), f'models_{metric}_boxplot.png')
        with b:
            show_plot(mv.run_distributions(per_run, metric, kind='violin'), f'models_{metric}_violin.png')
        show_plot(mv.run_lines(per_run, metric), f'models_{metric}_runs.png')
        if 'mean_tuning_and_fit_seconds' in summary.columns:
            show_plot(mv.runtime_chart(summary), 'models_runtime.png')
        st.divider()
        st.subheader('Confusion matrices')
        st.caption(result.get('confusion_matrix_note', 'Counts accumulated across repeated test splits; some windows appear more than once.'))
        model = st.selectbox('Model', list(result['confusion_matrices']), key='comparison_model')
        matrix = result['confusion_matrices'][model]
        labels = result['confusion_matrix_labels']
        a, b = st.columns(2)
        with a:
            show_plot(confusion_plot(matrix, labels, model, False), 'confusion_counts.png')
        with b:
            show_plot(confusion_plot(matrix, labels, model, True), 'confusion_normalized.png')
        st.divider()
        st.subheader('Per-class comparison')
        per_class = result['per_class_metrics']
        class_metric = st.selectbox('Per-class metric', ['precision', 'recall', 'f1_score'], key='comparison_class_metric')
        show_plot(mv.per_class_heatmap(per_class, class_metric), f'per_class_{class_metric}.png')
        st.dataframe(per_class.groupby(['model', 'class'], as_index=False)[['precision', 'recall', 'f1_score']].mean().round(4),
                     hide_index=True, width='stretch')
        st.subheader('Hyperparameter selection')
        show_plot(mv.parameter_frequency(result['best_parameters']), 'models_parameter_frequency.png')
        st.caption('The repeated train/test splits overlap. These plots show variability across random splits, not independent-study uncertainty.')

with downloads_tab:
    st.header('Download data and experiment results')
    st.download_button('Cleaned data (CSV)', csv_bytes(cleaned), 'shimmer_cleaned_data.csv', 'text/csv')
    st.download_button('Baseline features (CSV)', csv_bytes(features), 'shimmer_baseline_features.csv', 'text/csv')
    result = st.session_state['experiment_output']
    if result is None:
        st.info('Run experiments to enable ML result downloads. Each chart also has its own PNG download button in its tab.')
    else:
        for key, filename in [('results', 'per_run_results.csv'), ('summary', 'model_summary.csv'),
                              ('best_parameters', 'best_hyperparameters.csv'),
                              ('per_class_metrics', 'per_class_metrics.csv'),
                              ('pairwise_tests', 'pairwise_tests.csv'), ('chance_test', 'chance_test.csv'),
                              ('final_model_parameters', 'final_model_parameters.csv')]:
            if key in result and isinstance(result[key], pd.DataFrame):
                st.download_button(f'{key.replace("_", " ").title()} (CSV)', csv_bytes(result[key]),
                                   filename, 'text/csv', key=f'download_{key}')
        for model, matrix in result['confusion_matrices'].items():
            cm = pd.DataFrame(matrix, index=result['confusion_matrix_labels'], columns=result['confusion_matrix_labels'])
            cm.index.name = 'True activity'
            st.download_button(f'{model} confusion matrix (CSV)', csv_bytes(cm.reset_index()),
                               f'{str(model).lower().replace(" ", "_")}_confusion_matrix.csv',
                               'text/csv', key=f'download_cm_{model}')
