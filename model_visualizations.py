"""Model comparison plots using the existing ml_experiments output dictionary."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

METRICS = {
    'accuracy': 'Accuracy',
    'balanced_accuracy': 'Balanced accuracy',
    'precision_weighted': 'Weighted precision',
    'recall_weighted': 'Weighted recall',
    'f1_weighted': 'Weighted F1',
    'f1_macro': 'Macro F1',
    'mcc': 'MCC',
}


def metric_means(summary):
    names = [m for m in METRICS if m + '_mean' in summary.columns]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(summary))
    width = .8 / len(names)
    for j, m in enumerate(names):
        ax.bar(x - .4 + (j+.5)*width, summary[m+'_mean'], width, label=METRICS[m])
    ax.set_xticks(x, summary['model'])
    ax.set_ylabel('Mean metric value')
    ax.set_title('Model comparison across evaluation metrics')
    ax.legend(loc='lower center', bbox_to_anchor=(.5, -0.36), ncol=4)
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    return fig


def metric_errorbars(summary, metric='balanced_accuracy'):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(summary['model'], summary[metric+'_mean'], yerr=summary[metric+'_std'], capsize=6)
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title('Mean ± standard deviation across repeated splits')
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    return fig


def run_distributions(results, metric='balanced_accuracy', kind='box'):
    groups = [(str(name), group[metric].dropna().to_numpy()) for name, group in results.groupby('model', sort=True)]
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [n for n, _ in groups]
    vals = [v for _, v in groups]
    if kind == 'violin':
        ax.violinplot(vals, showmedians=True)
        ax.set_xticks(range(1, len(names)+1), names)
    else:
        ax.boxplot(vals, tick_labels=names)
    ax.set_ylabel(metric)
    ax.set_title(f'Per-run {metric} distribution')
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    return fig


def run_lines(results, metric='balanced_accuracy'):
    fig, ax = plt.subplots(figsize=(11, 5))
    run_col = next((c for c in ['run', 'run_id', 'iteration'] if c in results.columns), None)
    for name, group in results.groupby('model', sort=True):
        x = group[run_col] if run_col else np.arange(1, len(group)+1)
        ax.plot(x, group[metric], marker='o', ms=3, label=name)
    ax.set_xlabel('Run')
    ax.set_ylabel(metric)
    ax.set_title('Model performance across repeated splits')
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    return fig


def metrics_heatmap(summary):
    cols = [m+'_mean' for m in METRICS if m+'_mean' in summary.columns]
    values = summary[cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    im = ax.imshow(values, aspect='auto', vmin=-1, vmax=1, cmap='coolwarm')
    ax.set_xticks(range(len(cols)), [METRICS[c[:-5]] for c in cols], rotation=35, ha='right')
    ax.set_yticks(range(len(summary)), summary['model'])
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f'{values[i,j]:.3f}', ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=ax, label='Metric value (MCC may be negative)')
    ax.set_title('Mean evaluation metrics')
    fig.tight_layout()
    return fig


def runtime_chart(summary):
    col = 'mean_tuning_and_fit_seconds'
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(summary['model'], summary[col])
    ax.set_ylabel('Mean tuning + fit time (seconds)')
    ax.set_title('Computational cost comparison')
    fig.tight_layout()
    return fig


def per_class_heatmap(per_class_df, metric='f1_score'):
    pivot = per_class_df.groupby(['model', 'class'])[metric].mean().unstack('class')
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(pivot.to_numpy(), vmin=0, vmax=1, aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha='right')
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f'{pivot.iloc[i,j]:.2f}', ha='center', va='center', color='white', fontsize=8)
    fig.colorbar(im, ax=ax)
    ax.set_title(f'Per-class mean {metric} across runs')
    fig.tight_layout()
    return fig


def parameter_frequency(best_parameters):
    fig, ax = plt.subplots(figsize=(11, 5))
    df = best_parameters.copy()
    candidate = next((c for c in ['best_params', 'best_parameters', 'parameters', 'params'] if c in df.columns), None)
    if candidate is None:
        ax.text(.5, .5, 'Parameter column unavailable', ha='center', va='center')
    else:
        top = df.groupby(['model', candidate]).size().sort_values(ascending=False).head(12)
        ax.barh([f'{m}: {str(p)[:55]}' for m,p in top.index], top.values)
        ax.invert_yaxis()
        ax.set_xlabel('Number of runs')
    ax.set_title('Most frequently selected hyperparameter configurations')
    fig.tight_layout()
    return fig
