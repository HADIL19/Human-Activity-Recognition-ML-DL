"""
data_utils.py

Load and clean Shimmer accelerometer recordings.

Principles:
- Preserve original sample indices.
- Do not automatically delete repeated sensor readings.
- Report timestamp anomalies without assuming they are recording boundaries.
- Remove rows only when acceleration X/Y/Z contain invalid values.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.io import loadmat


MAT_COLUMNS = [
    "timestamp",
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "channel_08",
    "channel_09",
    "channel_10",
    "channel_11",
]

COLUMN_ALIASES = {
    "time": "timestamp",
    "time_stamp": "timestamp",
    "accelerometer_x": "accel_x",
    "accelerometer_y": "accel_y",
    "accelerometer_z": "accel_z",
    "acc_x": "accel_x",
    "acc_y": "accel_y",
    "acc_z": "accel_z",
    "accelx": "accel_x",
    "accely": "accel_y",
    "accelz": "accel_z",
    "gyroscope_x": "gyro_x",
    "gyroscope_y": "gyro_y",
    "gyroscope_z": "gyro_z",
    "gyrox": "gyro_x",
    "gyroy": "gyro_y",
    "gyroz": "gyro_z",
}


def normalize_columns(df):
    """Normalize column names and map common sensor aliases."""
    result = df.copy()
    normalized = []

    for column in result.columns:
        name = str(column).strip().lower()
        name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
        normalized.append(COLUMN_ALIASES.get(name, name))

    if len(set(normalized)) != len(normalized):
        raise ValueError(
            "Duplicate column names appeared after normalization."
        )

    result.columns = normalized
    return result


def infer_activity(filename):
    """Infer an activity label from a filename."""
    stem = Path(str(filename)).stem.strip()

    # Remove duplicate-download suffixes such as "(1)".
    stem = re.sub(r"\s*\(\d+\)\s*$", "", stem)

    # Remove suffixes such as "_2" or "-3".
    stem = re.sub(r"[_\-\s]+\d+$", "", stem)

    stem = re.sub(r"[_\-]+", " ", stem).strip()
    return stem.title()


def _mat_to_dataframe(uploaded_file, filename):
    """Read a MATLAB file containing an 11-column numeric matrix."""
    try:
        mat = loadmat(uploaded_file)
    except Exception as exc:
        raise ValueError(
            f"Could not read MAT file '{filename}': {exc}"
        ) from exc

    candidates = []

    for name, value in mat.items():
        if name.startswith("__"):
            continue

        array = np.asarray(value)

        if (
            array.ndim == 2
            and np.issubdtype(array.dtype, np.number)
        ):
            candidates.append((name, array))

    if not candidates:
        raise ValueError(
            f"No 2D numeric matrix was found in '{filename}'."
        )

    # Prefer the MATLAB variable named "data".
    selected = next(
        (item for item in candidates if item[0].lower() == "data"),
        None,
    )

    variable_name, array = selected or max(
        candidates,
        key=lambda item: item[1].size,
    )

    # Transpose only if the matrix appears to have channels as rows.
    if (
        array.shape[1] != len(MAT_COLUMNS)
        and array.shape[0] == len(MAT_COLUMNS)
    ):
        array = array.T

    if array.shape[1] != len(MAT_COLUMNS):
        raise ValueError(
            f"'{filename}' contains matrix '{variable_name}' "
            f"with shape {array.shape}; expected 11 columns."
        )

    df = pd.DataFrame(array, columns=MAT_COLUMNS)

    df["source_file"] = str(filename)
    df["activity"] = infer_activity(filename)
    df["segment_id"] = str(filename)
    df["sample_index"] = np.arange(len(df), dtype=np.int64)

    return df


def _read_uploaded_file(uploaded_file):
    """Read a single uploaded MAT or CSV file."""
    filename = getattr(uploaded_file, "name", None) or str(uploaded_file)
    suffix = Path(filename).suffix.lower()

    if suffix == ".mat":
        return _mat_to_dataframe(uploaded_file, filename)

    if suffix == ".csv":
        try:
            return pd.read_csv(uploaded_file)
        except Exception as exc:
            raise ValueError(
                f"Could not read CSV file '{filename}': {exc}"
            ) from exc

    raise ValueError(
        f"Unsupported file type '{suffix}'. Upload MAT or CSV files."
    )


def _looks_like_feature_table(df):
    """Identify a likely precomputed feature table."""
    if "activity" not in df.columns:
        return False

    if {"accel_x", "accel_y", "accel_z"}.issubset(df.columns):
        return False

    baseline_features = {
        "x_max", "x_min", "x_mean", "x_std",
        "y_max", "y_min", "y_mean", "y_std",
        "z_max", "z_min", "z_mean", "z_std",
    }

    return len(baseline_features.intersection(df.columns)) >= 4


def load_uploaded_files(uploaded_files):
    """
    Load and combine uploaded MAT/CSV files.

    Returns
    -------
    combined_df : pandas.DataFrame
    data_type : str
        'raw' or 'features'
    """
    if not uploaded_files:
        raise ValueError("Upload at least one MAT or CSV file.")

    frames = []

    for uploaded_file in uploaded_files:
        filename = (
            getattr(uploaded_file, "name", None)
            or str(uploaded_file)
        )

        frame = normalize_columns(
            _read_uploaded_file(uploaded_file)
        )

        if "source_file" not in frame.columns:
            frame["source_file"] = str(filename)

        if (
            "activity" not in frame.columns
            or frame["activity"].isna().all()
        ):
            frame["activity"] = infer_activity(filename)

        if "segment_id" not in frame.columns:
            frame["segment_id"] = str(filename)

        frames.append(frame)

    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    combined = normalize_columns(combined)

    if _looks_like_feature_table(combined):
        return combined, "features"

    required = {"accel_x", "accel_y", "accel_z"}
    missing = required - set(combined.columns)

    if missing:
        raise ValueError(
            "Raw data must contain accel_x, accel_y, and accel_z. "
            f"Missing columns: {sorted(missing)}"
        )

    for column in ("accel_x", "accel_y", "accel_z"):
        combined[column] = pd.to_numeric(
            combined[column],
            errors="coerce",
        )

    if "sample_index" not in combined.columns:
        combined["sample_index"] = (
            combined.groupby(
                "source_file",
                sort=False,
            ).cumcount()
        )
    else:
        combined["sample_index"] = pd.to_numeric(
            combined["sample_index"],
            errors="coerce",
        )

    return combined, "raw"


def clean_raw_data(df):
    """
    Remove rows with invalid acceleration measurements.

    Repeated sensor readings and timestamp anomalies are retained
    and reported rather than automatically deleted.

    Returns
    -------
    cleaned_df : pandas.DataFrame
    report : dict
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    working = normalize_columns(df)

    required = {"accel_x", "accel_y", "accel_z"}
    missing = required - set(working.columns)

    if missing:
        raise ValueError(
            f"Missing required acceleration columns: {sorted(missing)}"
        )

    rows_before = len(working)

    for column in ("accel_x", "accel_y", "accel_z"):
        working[column] = pd.to_numeric(
            working[column],
            errors="coerce",
        )

    sensor_values = working[
        ["accel_x", "accel_y", "accel_z"]
    ].to_numpy(dtype=float)

    valid_mask = np.isfinite(sensor_values).all(axis=1)

    invalid_rows_removed = int((~valid_mask).sum())

    # Preserve original sample_index values and row order.
    cleaned = working.loc[valid_mask].copy()

    if "source_file" not in cleaned.columns:
        cleaned["source_file"] = "unknown"

    if "activity" not in cleaned.columns:
        cleaned["activity"] = "Unknown"

    if "segment_id" not in cleaned.columns:
        cleaned["segment_id"] = cleaned["source_file"].astype(str)

    if "sample_index" not in cleaned.columns:
        cleaned["sample_index"] = (
            cleaned.groupby(
                "source_file",
                sort=False,
            ).cumcount()
        )
    else:
        cleaned["sample_index"] = pd.to_numeric(
            cleaned["sample_index"],
            errors="coerce",
        )

    duplicate_timestamps = 0
    nonincreasing_timestamps = 0
    large_timestamp_gaps = 0
    duplicate_readings = 0

    if "timestamp" in cleaned.columns:
        cleaned["timestamp"] = pd.to_numeric(
            cleaned["timestamp"],
            errors="coerce",
        )

        for _, group in cleaned.groupby(
            "source_file",
            sort=False,
            dropna=False,
        ):
            timestamps = group["timestamp"].to_numpy(dtype=float)
            timestamps = timestamps[np.isfinite(timestamps)]

            if len(timestamps) < 2:
                continue

            deltas = np.diff(timestamps)

            duplicate_timestamps += int(np.sum(deltas == 0))
            nonincreasing_timestamps += int(np.sum(deltas <= 0))

            positive_deltas = deltas[deltas > 0]

            if len(positive_deltas):
                median_delta = float(np.median(positive_deltas))

                if median_delta > 0:
                    large_timestamp_gaps += int(
                        np.sum(deltas > 1.5 * median_delta)
                    )

    ordered = cleaned.sort_values(
        ["source_file", "sample_index"],
        kind="stable",
    )

    for _, group in ordered.groupby(
        "source_file",
        sort=False,
        dropna=False,
    ):
        values = group[
            ["accel_x", "accel_y", "accel_z"]
        ].to_numpy(dtype=float)

        if len(values) > 1:
            duplicate_readings += int(
                np.all(values[1:] == values[:-1], axis=1).sum()
            )

    cleaned["magnitude"] = np.sqrt(
        cleaned["accel_x"].to_numpy(dtype=float) ** 2
        + cleaned["accel_y"].to_numpy(dtype=float) ** 2
        + cleaned["accel_z"].to_numpy(dtype=float) ** 2
    )

    if cleaned.empty:
        raise ValueError(
            "No valid rows remain after removing invalid acceleration values."
        )

    report = {
        "rows_before": int(rows_before),
        "rows_after": int(len(cleaned)),
        "invalid_sensor_rows_removed": invalid_rows_removed,
        "consecutive_duplicate_rows_removed": 0,
        "consecutive_duplicate_readings_detected": int(duplicate_readings),
        "duplicate_timestamps_detected": int(duplicate_timestamps),
        "nonincreasing_timestamp_steps_detected": int(
            nonincreasing_timestamps
        ),
        "large_timestamp_gaps_detected": int(large_timestamp_gaps),
        "timestamp_units_verified": False,
    }

    return cleaned, report
