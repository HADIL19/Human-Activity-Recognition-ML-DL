"""
feature_extraction.py

Extract the 12 baseline statistical features required for classical
machine learning on the CDTA Shimmer activity-recognition dataset.

Each window contains 128 consecutive samples.

Features per axis:
    max, min, mean, std

Axes:
    accelerometer X, Y, Z

Total: 12 features per window.
"""

import numpy as np
import pandas as pd


DEFAULT_WINDOW_SIZE = 128

AXES = {
    "x": "accel_x",
    "y": "accel_y",
    "z": "accel_z",
}

STATISTICS = ("max", "min", "mean", "std")

FEATURE_COLUMNS = [
    f"{axis}_{stat}"
    for axis in AXES
    for stat in STATISTICS
]

METADATA_COLUMNS = [
    "activity",
    "source_file",
    "segment_id",
    "window_id",
    "window_start_sample",
    "window_end_sample",
    "window_size",
    "group_id",
    "crosses_break",
]


def get_feature_columns(df):
    """
    Return the 12 expected baseline feature columns present in df,
    preserving the predefined order.
    """
    return [
        column
        for column in FEATURE_COLUMNS
        if column in df.columns
    ]


def extract_features_from_window(window):
    """
    Extract 12 baseline features from a single window.

    Parameters
    ----------
    window : pandas.DataFrame
        Must contain accel_x, accel_y, and accel_z.

    Returns
    -------
    dict
        Dictionary containing exactly 12 statistical features.
    """
    required = list(AXES.values())
    missing = set(required) - set(window.columns)

    if missing:
        raise ValueError(
            f"Window is missing required columns: {sorted(missing)}"
        )

    if len(window) != DEFAULT_WINDOW_SIZE:
        raise ValueError(
            f"Expected {DEFAULT_WINDOW_SIZE} samples, "
            f"received {len(window)}."
        )

    features = {}

    for axis, column in AXES.items():
        values = pd.to_numeric(
            window[column],
            errors="coerce",
        ).to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite values found in accelerometer axis {axis.upper()}."
            )

        features[f"{axis}_max"] = float(np.max(values))
        features[f"{axis}_min"] = float(np.min(values))
        features[f"{axis}_mean"] = float(np.mean(values))

        # Sample standard deviation, matching pandas' default ddof=1.
        features[f"{axis}_std"] = float(np.std(values, ddof=1))

    return features


def _has_consecutive_indices(group):
    """Check whether sample indices form one uninterrupted sequence."""
    indices = pd.to_numeric(
        group["sample_index"],
        errors="coerce",
    ).to_numpy(dtype=float)

    if not np.isfinite(indices).all():
        return False

    if len(indices) < 2:
        return True

    return bool(np.all(np.diff(indices) == 1))


def extract_window_features(
    raw_df,
    window_size=DEFAULT_WINDOW_SIZE,
):
    """
    Extract baseline features from non-overlapping windows.

    Windows are constructed separately for each combination of
    source_file, segment_id, and activity.

    A window is accepted only if:
    - It contains exactly window_size rows.
    - Its sample_index values are consecutive.
    - All three acceleration axes contain finite numeric values.

    Incomplete windows are discarded.

    Parameters
    ----------
    raw_df : pandas.DataFrame
        Cleaned raw sensor data.
    window_size : int, default=128
        Number of samples per window.

    Returns
    -------
    pandas.DataFrame
        One row per accepted window, with activity metadata and features.
    """
    if not isinstance(raw_df, pd.DataFrame):
        raise TypeError("raw_df must be a pandas DataFrame.")

    if window_size != DEFAULT_WINDOW_SIZE:
        raise ValueError(
            f"This project uses {DEFAULT_WINDOW_SIZE}-sample windows "
            "as specified for the baseline experiment."
        )

    required = {
        "activity",
        "source_file",
        "segment_id",
        "sample_index",
        "accel_x",
        "accel_y",
        "accel_z",
    }

    missing = required - set(raw_df.columns)

    if missing:
        raise ValueError(
            f"Raw data is missing required columns: {sorted(missing)}"
        )

    if raw_df.empty:
        raise ValueError("The raw dataset is empty.")

    df = raw_df.copy()

    df["sample_index"] = pd.to_numeric(
        df["sample_index"],
        errors="coerce",
    )

    for column in AXES.values():
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "activity",
            "source_file",
            "segment_id",
            "sample_index",
        ]
    )

    df = df.sort_values(
        ["source_file", "segment_id", "activity", "sample_index"],
        kind="stable",
    )

    grouping_columns = [
        "source_file",
        "segment_id",
        "activity",
    ]

    feature_rows = []
    window_number = 0

    for group_keys, group in df.groupby(
        grouping_columns,
        sort=False,
        dropna=False,
    ):
        group = group.sort_values(
            "sample_index",
            kind="stable",
        ).copy()

        # Recording-segment counter: a timestamp that does not increase
        # (reset) starts a new recording segment. Used only to build
        # group_id for group-aware evaluation; windowing is unchanged.
        if "timestamp" in group.columns:
            ts = pd.to_numeric(
                group["timestamp"], errors="coerce"
            ).to_numpy(dtype=float)
            step = np.diff(ts)
            resets = np.where(np.isnan(step), False, step <= 0)
            group["_segment"] = np.concatenate(
                [[0], np.cumsum(resets)]
            ).astype(int)
        else:
            group["_segment"] = 0

        # Do not bridge missing indices. Split into uninterrupted runs.
        indices = group["sample_index"].to_numpy(dtype=float)
        breaks = np.flatnonzero(np.diff(indices) != 1) + 1

        runs = np.split(np.arange(len(group)), breaks)

        for run_indices in runs:
            run = group.iloc[run_indices]

            # Only complete, non-overlapping windows are used.
            number_of_windows = len(run) // window_size

            for window_number_in_run in range(number_of_windows):
                start = window_number_in_run * window_size
                end = start + window_size

                window = run.iloc[start:end]

                sensor_values = window[
                    ["accel_x", "accel_y", "accel_z"]
                ].to_numpy(dtype=float)

                if (
                    len(window) != window_size
                    or not np.isfinite(sensor_values).all()
                ):
                    continue

                features = extract_features_from_window(window)

                source_file, segment_id, activity = group_keys

                start_sample = int(window["sample_index"].iloc[0])
                end_sample = int(window["sample_index"].iloc[-1])

                window_number += 1

                row = {
                    "activity": str(activity),
                    "source_file": str(source_file),
                    "segment_id": str(segment_id),
                    "window_id": window_number,
                    "window_start_sample": start_sample,
                    "window_end_sample": end_sample,
                    "window_size": window_size,
                    # Windows of the same recording segment share a group_id
                    # so they can be kept together in train OR test.
                    "group_id": (
                        f"{source_file}#"
                        f"{int(window['_segment'].iloc[0])}"
                    ),
                    "crosses_break": bool(
                        window["_segment"].nunique() > 1
                    ),
                }

                row.update(features)
                feature_rows.append(row)

    if not feature_rows:
        raise ValueError(
            "No valid windows were extracted. Check the accelerometer "
            "columns, sample indices, and available sample counts."
        )

    result = pd.DataFrame(feature_rows)

    # Keep metadata first, followed by exactly the 12 baseline features.
    ordered_columns = METADATA_COLUMNS + FEATURE_COLUMNS

    result = result[ordered_columns]

    return result


def validate_feature_table(feature_df):
    """
    Validate a table containing the 12 baseline features and activity labels.

    Invalid rows are removed. The function raises an error if the
    resulting dataset cannot support classification.
    """
    if not isinstance(feature_df, pd.DataFrame):
        raise TypeError("feature_df must be a pandas DataFrame.")

    df = feature_df.copy()

    required = {"activity", *FEATURE_COLUMNS}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Feature table is missing required columns: "
            f"{sorted(missing)}"
        )

    df["activity"] = df["activity"].astype("string").str.strip()

    df = df.dropna(subset=["activity"])
    df = df[df["activity"] != ""]

    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = df.dropna(subset=FEATURE_COLUMNS)

    if len(df) < 4:
        raise ValueError(
            "At least four valid windows are required for validation."
        )

    if df["activity"].nunique() < 2:
        raise ValueError(
            "At least two different activity classes are required."
        )

    # Retain optional metadata columns when available.
    available_metadata = [
        column
        for column in METADATA_COLUMNS
        if column in df.columns
    ]

    return df[
        available_metadata + FEATURE_COLUMNS
    ].reset_index(drop=True)
