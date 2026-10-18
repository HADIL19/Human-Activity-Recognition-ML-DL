"""
model_io.py

Save, load and use the trained models.

The experiments (10 or more runs) are used to EVALUATE the models. Each run
trains a fresh model on a different split, so those models are not kept.
The model worth saving is the FINAL model: the selected algorithm with its
tuned hyperparameters, refitted on all windows. It is returned by
repeated_random_split_experiments() as result["final_models"].

A saved file ("bundle") contains everything needed to predict later:
    - the complete pipeline (imputer + scaler + classifier)
    - the class names (the pipeline predicts numbers 0..6)
    - the 12 feature names and the window size
    - protocol, date, tuned parameters, cross-run metrics, library versions
"""

import datetime as dt
import io
import json
import platform
import zipfile
from pathlib import Path
from typing import cast

import joblib
import numpy as np
import pandas as pd
import sklearn

from data_utils import clean_raw_data, load_uploaded_files

from feature_extraction import (
    DEFAULT_WINDOW_SIZE,
    FEATURE_COLUMNS,
    extract_features_from_window,
    extract_window_features,
)

REGISTRY_COLUMNS = [
    "saved_at", "model", "protocol", "file", "n_runs",
    "balanced_accuracy_mean", "mcc_mean",
]


def _slug(name):
    return str(name).lower().replace(" ", "_")


def build_bundles(result, n_windows=None):
    """Create one bundle per final model from an experiment result."""
    if "final_models" not in result:
        raise ValueError("The result has no final_models.")

    summary = result["summary"].set_index("model")
    params = result.get("final_model_parameters")
    params = (
        params.set_index("model")["best_parameters"].to_dict()
        if isinstance(params, pd.DataFrame)
        and "best_parameters" in params.columns
        else {}
    )
    stamp = dt.datetime.now().isoformat(timespec="seconds")
    classes = [str(c) for c in result["label_encoder"].classes_]

    bundles = {}
    for name, pipeline in result["final_models"].items():
        row = summary.loc[name] if name in summary.index else None
        bundles[name] = {
            "pipeline": pipeline,
            "model_name": str(name),
            "classes": classes,
            "feature_columns": list(FEATURE_COLUMNS),
            "window_size": DEFAULT_WINDOW_SIZE,
            "protocol": result.get("split_mode", "unknown"),
            "trained_at": stamp,
            "n_windows": None if n_windows is None else int(n_windows),
            "best_parameters": str(params.get(name, "")),
            "evaluation_over_runs": (
                None if row is None else {
                    k: float(row[k])
                    for k in row.index
                    if k.endswith("_mean") or k.endswith("_std")
                }
            ),
            "n_runs": (
                int(result["results"]["run"].nunique())
                if "results" in result else None
            ),
            "versions": {
                "python": platform.python_version(),
                "scikit_learn": sklearn.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "joblib": joblib.__version__,
            },
        }
    return bundles


def bundle_bytes(bundle):
    """Serialise a bundle to bytes (for a download button)."""
    buffer = io.BytesIO()
    joblib.dump(bundle, buffer, compress=3)
    return buffer.getvalue()


def save_final_models(result, out_dir="models", n_windows=None):
    """
    Save every final model to out_dir.

    For each model two files are written:
        <model>_<protocol>_<YYYYmmdd_HHMMSS>.joblib   (kept, never overwritten)
        <model>_<protocol>_latest.joblib              (always the newest)
    and one line per model is appended to models_registry.csv, so every
    retraining stays traceable.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    registry_rows, paths = [], []
    for name, bundle in build_bundles(result, n_windows).items():
        base = f"{_slug(name)}_{bundle['protocol']}"
        versioned = out / f"{base}_{stamp}.joblib"
        counter = 2
        while versioned.exists():  # never overwrite a previous save
            versioned = out / f"{base}_{stamp}_{counter}.joblib"
            counter += 1
        latest = out / f"{base}_latest.joblib"
        joblib.dump(bundle, versioned, compress=3)
        joblib.dump(bundle, latest, compress=3)
        paths.append(versioned)

        ev = bundle["evaluation_over_runs"] or {}
        registry_rows.append({
            "saved_at": bundle["trained_at"],
            "model": name,
            "protocol": bundle["protocol"],
            "file": versioned.name,
            "n_runs": bundle["n_runs"],
            "balanced_accuracy_mean": ev.get("balanced_accuracy_mean"),
            "mcc_mean": ev.get("mcc_mean"),
        })

    registry = out / "models_registry.csv"
    new = pd.DataFrame(registry_rows, columns=REGISTRY_COLUMNS)
    if registry.exists():
        new = pd.concat([pd.read_csv(registry), new], ignore_index=True)
    new.to_csv(registry, index=False)
    return paths


def load_model(path):
    """Load a saved bundle."""
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError("This is not a bundle created by save_final_models.")
    return bundle


def predict_window(bundle, window_xyz):
    """
    Predict the activity of ONE window.

    window_xyz : array-like of shape (128, 3), columns = accel X, Y, Z,
                 in the same units as the training data.
    Returns (activity name, {class: probability} or None).
    """
    window = np.asarray(window_xyz, dtype=float)
    size = bundle["window_size"]
    if window.shape != (size, 3):
        raise ValueError(f"Expected shape ({size}, 3), got {window.shape}.")

    features = extract_features_from_window(
        pd.DataFrame(window, columns=["accel_x", "accel_y", "accel_z"])
    )
    row = np.array([[features[c] for c in bundle["feature_columns"]]])

    pipeline = bundle["pipeline"]
    label = bundle["classes"][int(pipeline.predict(row)[0])]

    probabilities = None
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(row)[0]
        probabilities = {c: float(p) for c, p in zip(bundle["classes"], proba)}
    return label, probabilities


def predict_recording(bundle, xyz):
    """
    Predict a longer recording by cutting it into consecutive windows.

    xyz : array-like of shape (n_samples, 3). Incomplete last window dropped.
    Returns a DataFrame (one row per window) and the majority activity.
    """
    xyz = np.asarray(xyz, dtype=float)
    size = bundle["window_size"]
    rows = []
    for start in range(0, len(xyz) - size + 1, size):
        label, _ = predict_window(bundle, xyz[start:start + size])
        rows.append({"start_sample": start, "activity": label})
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError(f"Need at least {size} samples.")
    return table, table["activity"].value_counts().idxmax()


def load_model_bytes(data):
    """Load a bundle from bytes (an uploaded .joblib file)."""
    bundle = joblib.load(io.BytesIO(data))
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError("This is not a bundle created by save_final_models.")
    return bundle


def predict_files(bundle, files):
    """
    Predict the activity of every window in NEW recordings.

    files : paths or uploaded file objects (.mat with the same 11-column
            layout as the training files, or .csv with accel_x/accel_y/accel_z
            and optionally timestamp). The windows are built with exactly the
            same code as for training (128 samples, no overlap, 12 features).

    Returns
    -------
    windows : DataFrame, one row per window (file, start/end sample,
              predicted activity, confidence when the model gives probabilities)
    summary : DataFrame, one row per file (number of windows, majority
              activity, share of windows, mean confidence)
    """
    raw, kind = cast(
        tuple[pd.DataFrame, str], load_uploaded_files(files)
    )
    pipeline = bundle["pipeline"]
    classes = bundle["classes"]
    columns = bundle["feature_columns"]

    if kind == "features":
        missing = [c for c in columns if c not in raw.columns]
        if missing:
            raise ValueError(f"The feature table is missing columns: {missing}")
        feats = raw.reset_index(drop=True).copy()
        if "window_id" not in feats.columns:
            feats["window_id"] = np.arange(1, len(feats) + 1)
        for extra in ("window_start_sample", "window_end_sample"):
            if extra not in feats.columns:
                feats[extra] = np.nan
        X = (
            feats[columns]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .to_numpy(dtype=float)
        )
    else:
        clean, _ = clean_raw_data(raw)
        feats = extract_window_features(clean, window_size=bundle["window_size"])
        X = feats[columns].to_numpy(dtype=float)

    predicted = [classes[int(i)] for i in pipeline.predict(X)]

    windows = feats[
        ["source_file", "window_id", "window_start_sample", "window_end_sample"]
    ].copy()
    windows["predicted_activity"] = predicted
    if hasattr(pipeline, "predict_proba"):
        windows["confidence"] = pipeline.predict_proba(X).max(axis=1)

    # Label read from an 'activity' column or from the file name (Walk.mat).
    # It is only kept when it is one of the model's classes, so a made-up
    # file name never counts as a true label.
    lookup = {c.lower(): c for c in classes}
    labels = feats["activity"].astype(str).str.strip().str.lower().map(lookup)
    if labels.notna().any():
        windows["true_activity"] = labels.to_numpy()

    rows = []
    for name, group in windows.groupby("source_file", sort=False):
        counts = group["predicted_activity"].value_counts()
        row = {
            "file": name,
            "windows": int(len(group)),
            "majority_activity": counts.index[0],
            "share_of_windows": float(counts.iloc[0] / len(group)),
        }
        if "confidence" in group:
            row["mean_confidence"] = float(group["confidence"].mean())
        rows.append(row)

    return windows, pd.DataFrame(rows)


def bundles_zip_bytes(bundles):
    """
    Pack final models into ONE zip: one .joblib per model and model_card.json.
    bundles : dict returned by build_bundles().
    """
    buffer = io.BytesIO()
    card = []
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, bundle in bundles.items():
            zf.writestr(
                f"{_slug(name)}_{bundle['protocol']}.joblib", bundle_bytes(bundle)
            )
            card.append({k: v for k, v in bundle.items() if k != "pipeline"})
        zf.writestr("model_card.json", json.dumps(card, indent=2, default=str))
    return buffer.getvalue()


def load_bundles_from_upload(filename, data, max_bytes=500_000_000):
    """
    Read an uploaded model file: a .zip made by bundles_zip_bytes (or holding
    .joblib bundles) or a single .joblib. Returns {label: bundle}.
    """
    found = {}
    if str(filename).lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            members = [i for i in zf.infolist() if i.filename.lower().endswith(".joblib")]
            if not members:
                raise ValueError("The zip contains no .joblib model file.")
            if sum(i.file_size for i in members) > max_bytes:
                raise ValueError("The zip is too large.")
            for info in members:
                bundle = load_model_bytes(zf.read(info))
                found[f"{bundle['model_name']} ({bundle['protocol']})"] = bundle
    else:
        bundle = load_model_bytes(data)
        found[f"{bundle['model_name']} ({bundle['protocol']})"] = bundle
    return found
