"""
FEATURE EXTRACTION
==================

Convert raw Shimmer recordings into a feature table.

Each row represents one fixed-length sensor window.
The default window size is 128 samples.
"""

import numpy as np
import pandas as pd

from scipy.stats import skew, kurtosis


# The three acceleration axes used for the baseline experiment.
AXES = {
    "x": "accel_x",
    "y": "accel_y",
    "z": "accel_z"
}


# ---------------------------------------------------------
# 1. EXTRACT FEATURES FROM ONE WINDOW
# ---------------------------------------------------------

def extract_features_from_window(
    window,
    include_extended=False
):
    """
    Calculate statistical features from a single window.

    Baseline features:
        max, min, mean and standard deviation for X, Y and Z.

    Extended features are optional.
    """

    features = {}

    axis_values = {}

    for axis, column in AXES.items():

        values = window[column].to_numpy(dtype=float)

        axis_values[axis] = values

        # Four baseline statistical features.
        features[f"{axis}_max"] = np.max(values)
        features[f"{axis}_min"] = np.min(values)
        features[f"{axis}_mean"] = np.mean(values)

        # Population standard deviation.
        features[f"{axis}_std"] = np.std(
            values,
            ddof=0
        )

        if include_extended:

            features[f"{axis}_median"] = np.median(values)

            features[f"{axis}_range"] = np.ptp(values)

            features[f"{axis}_iqr"] = (
                np.percentile(values, 75)
                - np.percentile(values, 25)
            )

            features[f"{axis}_rms"] = np.sqrt(
                np.mean(values ** 2)
            )

            features[f"{axis}_energy"] = np.sum(
                values ** 2
            )

            # Skewness and kurtosis are undefined or unstable
            # for constant signals, so handle that case.
            if np.std(values) > 0:

                features[f"{axis}_skew"] = skew(
                    values,
                    bias=False
                )

                features[f"{axis}_kurtosis"] = kurtosis(
                    values,
                    bias=False
                )

            else:

                features[f"{axis}_skew"] = 0.0
                features[f"{axis}_kurtosis"] = 0.0

    if include_extended:

        x = axis_values["x"]
        y = axis_values["y"]
        z = axis_values["z"]

        # Axis correlations describe how pairs of axes vary together.
        pairs = [
            ("x", x, "y", y),
            ("x", x, "z", z),
            ("y", y, "z", z)
        ]

        for axis1, values1, axis2, values2 in pairs:

            if (
                np.std(values1) > 0
                and np.std(values2) > 0
            ):

                correlation = np.corrcoef(
                    values1,
                    values2
                )[0, 1]

            else:
                correlation = 0.0

            features[
                f"{axis1}{axis2}_correlation"
            ] = correlation

        # Magnitude features.
        magnitude = np.sqrt(
            x ** 2 + y ** 2 + z ** 2
        )

        features["magnitude_mean"] = np.mean(magnitude)
        features["magnitude_std"] = np.std(magnitude)
        features["magnitude_max"] = np.max(magnitude)
        features["magnitude_min"] = np.min(magnitude)

    return features


# ---------------------------------------------------------
# 2. EXTRACT FEATURES FROM ALL RECORDINGS
# ---------------------------------------------------------

def extract_window_features(
    raw_df,
    window_size=128,
    include_extended=False
):
    """
    Divide each recording into non-overlapping windows.

    Windows never cross:
        - source files
        - activity labels
        - segment identifiers

    Incomplete final windows are discarded.

    Windows containing non-finite acceleration values are skipped.
    """

    if window_size < 2:
        raise ValueError(
            "Window size must be at least 2."
        )

    required_columns = {
        "activity",
        "source_file",
        "segment_id",
        "accel_x",
        "accel_y",
        "accel_z"
    }

    missing = required_columns - set(raw_df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    rows = []

    grouping_columns = [
        "source_file",
        "segment_id",
        "activity"
    ]

    for group_keys, group in raw_df.groupby(
        grouping_columns,
        sort=False,
        dropna=False
    ):

        source_file, segment_id, activity = group_keys

        # Preserve the temporal order of samples.
        if "sample_index" in group.columns:

            group = group.sort_values("sample_index")

        group = group.reset_index(drop=True)

        number_of_windows = len(group) // window_size

        for window_number in range(number_of_windows):

            start = window_number * window_size

            end = start + window_size

            window = group.iloc[start:end]

            sensor_values = window[
                ["accel_x", "accel_y", "accel_z"]
            ].to_numpy(dtype=float)

            # Never calculate features from NaN or infinite values.
            if not np.isfinite(sensor_values).all():
                continue

            features = extract_features_from_window(
                window,
                include_extended=include_extended
            )

            # Add metadata to help trace the feature row.
            features["activity"] = str(activity)

            features["source_file"] = str(source_file)

            features["segment_id"] = str(segment_id)

            features["window_start_sample"] = int(start)

            features["window_size"] = int(window_size)

            rows.append(features)

    feature_df = pd.DataFrame(rows)

    if feature_df.empty:

        raise ValueError(
            "No valid windows were extracted. "
            "Check your data length, column names and missing values."
        )

    return feature_df


# ---------------------------------------------------------
# 3. IDENTIFY MODEL FEATURES
# ---------------------------------------------------------

def get_feature_columns(feature_df):
    """
    Return numeric features while excluding labels and metadata.

    Metadata such as sample positions and filenames must not be
    used as predictors by the classification models.
    """

    excluded_columns = {
        "activity",
        "source_file",
        "segment_id",
        "recording_id",
        "window_start_sample",
        "window_size",
        "sample_index"
    }

    numeric_columns = feature_df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    return [
        column
        for column in numeric_columns
        if column not in excluded_columns
    ]


# ---------------------------------------------------------
# 4. VALIDATE A FEATURE TABLE
# ---------------------------------------------------------

def validate_feature_table(feature_df):
    """
    Validate a precomputed feature table before training.
    """

    feature_df = feature_df.copy()

    if "activity" not in feature_df.columns:

        raise ValueError(
            "The feature table must contain an activity column."
        )

    feature_df = feature_df.dropna(
        subset=["activity"]
    ).copy()

    feature_df["activity"] = (
        feature_df["activity"].astype(str)
    )

    feature_columns = get_feature_columns(feature_df)

    if not feature_columns:

        raise ValueError(
            "No numeric feature columns were found."
        )

    # Replace infinity with missing values.
    feature_df[feature_columns] = (
        feature_df[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
    )

    # Remove unusable feature rows instead of fabricating values.
    feature_df = feature_df.dropna(
        subset=feature_columns
    ).reset_index(drop=True)

    if feature_df["activity"].nunique() < 2:

        raise ValueError(
            "Classification requires at least two activity classes."
        )

    return feature_df, feature_columns