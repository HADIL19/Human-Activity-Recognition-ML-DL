"""
DATA UTILITIES
==============

Responsibilities:
1. Load CSV and MATLAB (.mat) files.
2. Standardize column names.
3. Assign activity labels.
4. Combine multiple recordings.
5. Perform basic data cleaning.
6. Calculate acceleration magnitude.
"""

from io import BytesIO
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.io import loadmat


# ---------------------------------------------------------
# 1. EXPECTED COLUMNS FOR THE SHIMMER MAT FILE
# ---------------------------------------------------------

# This assumes that your MAT file contains 11 columns:
# timestamp, accelerometer X/Y/Z, gyroscope X/Y/Z,
# and four additional channels.
#
# IMPORTANT:
# Verify this mapping against your actual MAT files.

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


# ---------------------------------------------------------
# 2. STANDARDIZE COLUMN NAMES
# ---------------------------------------------------------

def normalize_columns(df):
    """
    Convert common column names to a consistent format.

    Examples:
        accX -> accel_x
        accY -> accel_y
        accZ -> accel_z
        timestamp_ms -> timestamp
        label -> activity
    """

    df = df.copy()

    rename_map = {}

    aliases = {
        "timestamp_ms": "timestamp",
        "time": "timestamp",
        "accx": "accel_x",
        "acc_x": "accel_x",
        "accelx": "accel_x",
        "accy": "accel_y",
        "acc_y": "accel_y",
        "accely": "accel_y",
        "accz": "accel_z",
        "acc_z": "accel_z",
        "accelz": "accel_z",
        "x": "accel_x",
        "y": "accel_y",
        "z": "accel_z",
        "label": "activity",
        "class": "activity",
        "activity_label": "activity",
    }

    for column in df.columns:

        # Standardize capitalization, spaces and hyphens.
        name = str(column).strip().lower()
        name = re.sub(r"[\s\-]+", "_", name)

        if name in aliases:
            rename_map[column] = aliases[name]
        else:
            rename_map[column] = name

    df = df.rename(columns=rename_map)

    return df


# ---------------------------------------------------------
# 3. INFER ACTIVITY FROM FILENAME
# ---------------------------------------------------------

def infer_activity(filename):
    """
    Infer an activity label from the filename.

    Examples:
        Fall(1).mat -> Fall
        Run.mat     -> Run
        Walk.csv    -> Walk
    """

    name = Path(filename).stem

    # Remove suffixes such as (1), (2), etc.
    name = re.sub(r"\(\d+\)$", "", name)

    # Remove a trailing separator and number, if present.
    name = re.sub(r"[_\- ]+\d+$", "", name)

    return name.strip().title() or "Unknown"


# ---------------------------------------------------------
# 4. LOAD A MATLAB FILE
# ---------------------------------------------------------

def load_mat_file(uploaded_file):
    """
    Load a MATLAB file.

    The preferred MATLAB variable is 'data'.

    If that variable is absent, the function searches for
    the largest two-dimensional numeric array.

    The exact column mapping must match your real files.
    """

    content = uploaded_file.getvalue()

    mat = loadmat(BytesIO(content))

    array = None

    # Prefer the variable named 'data'.
    if (
        "data" in mat
        and isinstance(mat["data"], np.ndarray)
        and mat["data"].ndim == 2
    ):
        array = mat["data"]

    # Otherwise, find a plausible numeric matrix.
    if array is None:

        candidates = []

        for key, value in mat.items():

            if key.startswith("__"):
                continue

            if (
                isinstance(value, np.ndarray)
                and value.ndim == 2
                and np.issubdtype(value.dtype, np.number)
            ):
                candidates.append(value)

        if not candidates:
            raise ValueError(
                f"No suitable numeric matrix found in {uploaded_file.name}."
            )

        array = max(candidates, key=lambda item: item.size)

    # Transpose if the expected 11 columns are stored as rows.
    if array.shape[1] != 11 and array.shape[0] == 11:
        array = array.T

    if array.shape[1] != 11:
        raise ValueError(
            f"{uploaded_file.name} contains {array.shape[1]} columns. "
            "This loader expects 11 columns. Verify the MAT structure "
            "and update MAT_COLUMNS if necessary."
        )

    return pd.DataFrame(array, columns=MAT_COLUMNS)


# ---------------------------------------------------------
# 5. LOAD AND COMBINE UPLOADED FILES
# ---------------------------------------------------------

def load_uploaded_files(uploaded_files):
    """
    Load multiple sensor recordings or a precomputed feature table.

    The caller should not mix raw recordings and precomputed
    feature tables in one upload.
    """

    if not uploaded_files:
        raise ValueError("Upload at least one file.")

    frames = []
    data_types = []

    for file in uploaded_files:

        filename = file.name
        extension = Path(filename).suffix.lower()

        # Load the file.
        if extension == ".mat":

            df = load_mat_file(file)
            data_type = "raw"

        elif extension == ".csv":

            df = pd.read_csv(file)
            df = normalize_columns(df)

            # Recognize an existing feature table.
            baseline_features = {
                f"{axis}_{stat}"
                for axis in ("x", "y", "z")
                for stat in ("max", "min", "mean", "std")
            }

            if (
                "activity" in df.columns
                and baseline_features.issubset(df.columns)
            ):
                data_type = "features"
            else:
                data_type = "raw"

        else:
            raise ValueError(
                f"Unsupported file type: {filename}"
            )

        df = normalize_columns(df)

        # Preserve the origin of every row.
        df["source_file"] = filename

        # If no activity label exists, infer it from the filename.
        if (
            "activity" not in df.columns
            or df["activity"].isna().all()
        ):
            df["activity"] = infer_activity(filename)

        else:
            df["activity"] = (
                df["activity"]
                .fillna(infer_activity(filename))
                .astype(str)
            )

        frames.append(df)
        data_types.append(data_type)

    # Do not combine raw samples with extracted features.
    if len(set(data_types)) != 1:
        raise ValueError(
            "Do not mix raw recordings and precomputed feature CSVs. "
            "Upload one type at a time."
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False
    )

    data_type = data_types[0]

    if data_type == "raw":

        required = {
            "accel_x",
            "accel_y",
            "accel_z"
        }

        missing = required - set(combined.columns)

        if missing:
            raise ValueError(
                f"Missing acceleration columns: {sorted(missing)}. "
                "Verify your CSV column names."
            )

        # Convert the important sensor columns to numeric values.
        for column in [
            "timestamp",
            "accel_x",
            "accel_y",
            "accel_z"
        ]:

            if column in combined.columns:
                combined[column] = pd.to_numeric(
                    combined[column],
                    errors="coerce"
                )

        # Generate a sample index within each recording.
        combined["sample_index"] = (
            combined.groupby("source_file").cumcount()
        )

        # Give each file its own segment identifier.
        if "segment_id" not in combined.columns:
            combined["segment_id"] = combined["source_file"]

    return combined, data_type


# ---------------------------------------------------------
# 6. BASIC SENSOR DATA CLEANING
# ---------------------------------------------------------

def clean_raw_data(df):
    """
    Clean raw sensor data conservatively.

    - Removes rows with missing X/Y/Z acceleration.
    - Removes consecutive duplicate sensor rows within
      the same recording.
    - Calculates acceleration magnitude.

    This does not automatically remove genuine high values.
    """

    df = df.copy()

    report = {
        "rows_before": len(df)
    }

    # Ensure numeric sensor values.
    for column in ["accel_x", "accel_y", "accel_z"]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Remove rows where an essential acceleration value is missing.
    missing_mask = df[
        ["accel_x", "accel_y", "accel_z"]
    ].isna().any(axis=1)

    report["missing_acceleration_rows_removed"] = int(
        missing_mask.sum()
    )

    df = df.loc[~missing_mask].copy()

    # Check for consecutive duplicate samples.
    comparison_columns = [
        column
        for column in [
            "timestamp",
            "accel_x",
            "accel_y",
            "accel_z"
        ]
        if column in df.columns
    ]

    if comparison_columns:

        previous = (
            df.groupby("source_file", sort=False)[
                comparison_columns
            ].shift(1)
        )

        # A duplicate must match the previous row in the same file.
        duplicate_mask = (
            df[comparison_columns]
            .eq(previous)
            .all(axis=1)
            & previous.notna().all(axis=1)
        )

        report["consecutive_duplicates_removed"] = int(
            duplicate_mask.sum()
        )

        df = df.loc[~duplicate_mask].copy()

    else:
        report["consecutive_duplicates_removed"] = 0

    df = df.reset_index(drop=True)

    # Calculate the magnitude of the three acceleration axes.
    df["magnitude"] = np.sqrt(
        df["accel_x"] ** 2
        + df["accel_y"] ** 2
        + df["accel_z"] ** 2
    )

    report["rows_after"] = len(df)

    return df, report