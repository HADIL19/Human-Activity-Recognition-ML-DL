"""Exploratory plots for the CDTA Shimmer Streamlit application."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

AXES = ['accel_x', 'accel_y', 'accel_z']


def _fig(w=11, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax


def activity_counts(df, feature_df):
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for ax, source, title in zip(axs, [df, feature_df], ['Samples by activity', 'Windows by activity']):
        c = source['activity'].astype(str).value_counts().sort_index()
        ax.bar(c.index, c.values)
        ax.set_title(title)
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=35)
        for i, v in enumerate(c.values):
            ax.text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    return fig


def class_proportions(df):
    counts = df['activity'].astype(str).value_counts().sort_index()
    fig, ax = _fig(8, 6)
    ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=90)
    ax.set_title('Activity proportions (samples)')
    fig.tight_layout()
    return fig


def missing_values(df):
    counts = df.isna().sum().sort_values(ascending=False)
    fig, ax = _fig()
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_ylabel('Missing cells')
    ax.set_title('Missing values by column')
    ax.tick_params(axis='x', rotation=55)
    fig.tight_layout()
    return fig


def missing_pattern(df, max_rows=350):
    subset = df.iloc[np.linspace(0, len(df)-1, min(len(df), max_rows), dtype=int)]
    fig, ax = _fig(12, 5)
    ax.imshow(subset.isna().to_numpy().T, aspect='auto', interpolation='nearest', cmap='Greys')
    ax.set_yticks(np.arange(len(df.columns)), df.columns)
    ax.set_xlabel('Sampled row positions (not continuous)')
    ax.set_title('Missingness map: dark = missing')
    fig.tight_layout()
    return fig


def axis_distributions(df, axis='accel_x', activities=None, max_per_activity=4000, kind='hist'):
    fig, ax = _fig(11, 6)
    activities = activities or sorted(df['activity'].astype(str).unique())
    for name in activities:
        series = pd.to_numeric(df.loc[df['activity'].astype(str).eq(name), axis], errors='coerce').dropna()
        if len(series) > max_per_activity:
            series = series.sample(max_per_activity, random_state=42)
        if series.empty:
            continue
        if kind == 'hist':
            ax.hist(series, bins=55, density=True, alpha=0.3, label=name)
        else:
            series.plot(kind='kde', ax=ax, label=name)
    ax.set_title(f'{axis}: activity distributions ({kind})')
    ax.set_xlabel(axis)
    ax.set_ylabel('Density')
    ax.legend()
    fig.tight_layout()
    return fig


def activity_boxplot(df, axis='accel_x', kind='box', max_per_activity=3000):
    activities = sorted(df['activity'].astype(str).unique())
    data = []
    names = []
    for name in activities:
        vals = pd.to_numeric(df.loc[df['activity'].astype(str).eq(name), axis], errors='coerce').dropna()
        if len(vals) > max_per_activity:
            vals = vals.sample(max_per_activity, random_state=42)
        if not vals.empty:
            data.append(vals.to_numpy())
            names.append(name)
    fig, ax = _fig(11, 6)
    if data:
        if kind == 'violin':
            ax.violinplot(data, showmedians=True)
            ax.set_xticks(range(1, len(names)+1), names)
        else:
            ax.boxplot(data, tick_labels=names, showfliers=False)
    ax.set_title(f'{axis}: {kind} comparison by activity')
    ax.set_ylabel(axis)
    fig.tight_layout()
    return fig


def signal_comparison(df, activities, axis='magnitude', samples=1000):
    fig, axs = plt.subplots(len(activities), 1, figsize=(13, max(3.5, len(activities)*2.6)), sharex=False, squeeze=False)
    for ax, activity in zip(axs[:, 0], activities):
        sub = df[df['activity'].astype(str).eq(activity)]
        if 'source_file' in sub.columns and not sub.empty:
            sub = sub[sub['source_file'].eq(sub['source_file'].iloc[0])]
        if 'sample_index' in sub.columns:
            sub = sub.sort_values('sample_index', kind='stable')
        sub = sub.head(samples)
        if axis == 'magnitude':
            xyz = sub[AXES].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
            vals = np.sqrt(np.sum(xyz**2, axis=1))
        else:
            vals = pd.to_numeric(sub[axis], errors='coerce').to_numpy(dtype=float)
        ax.plot(np.arange(len(vals)), vals, lw=.85)
        ax.set_title(activity)
        ax.set_ylabel(axis)
        ax.grid(alpha=.25)
    axs[-1, 0].set_xlabel('Sample position within selected recording')
    fig.suptitle('Activity signal comparison (first recording per activity)')
    fig.tight_layout()
    return fig


def timestamp_differences(df, source_file):
    sub = df[df['source_file'].astype(str).eq(str(source_file))]
    ts = pd.to_numeric(sub['timestamp'], errors='coerce').to_numpy(dtype=float)
    differences = np.diff(ts)
    differences = differences[np.isfinite(differences)]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    if len(differences):
        axs[0].plot(differences, lw=.6)
        pos = differences[differences > 0]
        if len(pos):
            cap = np.quantile(pos, .99)
            axs[1].hist(pos[pos <= cap], bins=60)
            axs[1].set_title('Positive differences (up to 99th percentile)')
    axs[0].set_title('Consecutive timestamp differences')
    axs[0].set_xlabel('Difference index')
    axs[1].set_xlabel('Timestamp difference (units unverified)')
    fig.tight_layout()
    return fig
