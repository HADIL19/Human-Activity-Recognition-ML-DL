"""Export/import Shimmer HAR fitted pipelines and predict on new feature windows.

Only load model bundles from sources you trust: joblib uses pickle.
"""
import io
import json
import zipfile

import joblib
import numpy as np
import pandas as pd
from scipy.io import loadmat

from ml_experiments import FEATURE_COLUMNS


def export_bundle(result):
    """Return ZIP bytes containing final fitted pipelines and inference metadata."""
    buffer = io.BytesIO()
    metadata = {
        "split_mode": result["split_mode"],
        "feature_columns": list(result["feature_columns"]),
        "window_size": 128,
        "classes": [str(x) for x in result["label_encoder"].classes_],
        "note": "Final pipelines refitted on all provided windows; reported evaluation scores come from separate repeated splits.",
    }
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metadata.json", json.dumps(metadata, indent=2))
        for name, model in result["final_models"].items():
            item = io.BytesIO()
            joblib.dump(model, item)
            archive.writestr(f"models/{name.lower().replace(' ', '_')}.joblib", item.getvalue())
        item = io.BytesIO()
        joblib.dump(result["label_encoder"], item)
        archive.writestr("label_encoder.joblib", item.getvalue())
    return buffer.getvalue()


def import_bundle(zip_bytes):
    """Load a trusted ZIP created by export_bundle."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        metadata = json.loads(archive.read("metadata.json"))
        if metadata.get("feature_columns") != FEATURE_COLUMNS or metadata.get("window_size") != 128:
            raise ValueError("Incompatible feature definition or window size.")
        models = {}
        for name in ("knn", "svm", "decision_tree"):
            path = f"models/{name}.joblib"
            if path in archive.namelist():
                models[name] = joblib.load(io.BytesIO(archive.read(path)))
        encoder = joblib.load(io.BytesIO(archive.read("label_encoder.joblib")))
        if not models:
            raise ValueError("No models found in bundle.")
    return metadata, models, encoder


def features_from_csv(file):
    """Accept a feature CSV with exactly named baseline feature columns (other columns allowed)."""
    df = pd.read_csv(file)
    missing = [name for name in FEATURE_COLUMNS if name not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing feature columns: {missing}")
    x = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if x.empty or x.isna().all(axis=1).any():
        raise ValueError("No windows, or at least one window has all features missing.")
    return x.replace([np.inf, -np.inf], np.nan)


def features_from_mat(file):
    """Extract baseline features from one Shimmer MATLAB recording.

    Assumes the supplied .mat has a numeric `data` matrix with timestamp in
    column 0 and accelerometer X/Y/Z in columns 1/2/3. Nonoverlapping windows.
    Do not use on concatenated recordings unless boundaries are already separated.
    """
    mat = loadmat(io.BytesIO(file.read()))
    if "data" not in mat:
        raise ValueError("MAT file must contain a 'data' matrix.")
    data = np.asarray(mat["data"])
    if data.ndim != 2 or data.shape[1] < 4 or not np.issubdtype(data.dtype, np.number):
        raise ValueError("Expected a numeric data matrix with timestamp, X, Y, Z columns.")
    xyz = data[:, 1:4].astype(float)
    n = len(xyz) // 128
    if n == 0:
        raise ValueError("Need at least 128 samples.")
    rows = []
    for i in range(n):
        window = xyz[i * 128:(i + 1) * 128]
        row = {}
        for j, axis in enumerate("xyz"):
            vals = window[:, j]
            row.update({f"{axis}_max": np.nanmax(vals), f"{axis}_min": np.nanmin(vals),
                        f"{axis}_mean": np.nanmean(vals), f"{axis}_std": np.nanstd(vals, ddof=1)})
        rows.append(row)
    x = pd.DataFrame(rows)[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    if x.isna().all(axis=1).any():
        raise ValueError("At least one 128-sample window has no valid accelerometer values.")
    return x


def predict_features(model, encoder, x):
    encoded = model.predict(x)
    return pd.DataFrame({"window": np.arange(1, len(x) + 1),
                         "predicted_activity": encoder.inverse_transform(encoded)})
