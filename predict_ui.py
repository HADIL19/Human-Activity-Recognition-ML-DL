"""
predict_ui.py

The "Predict activities with a saved model" section, used by app.py and by
predict_app.py. It trains nothing: it loads a saved model bundle and predicts
the activity of every 128-sample window of new data.
"""

import pandas as pd
import streamlit as st
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

import model_io


def _show_labelled_check(windows):
    labelled = windows.dropna(subset=["true_activity"])
    if labelled.empty:
        return
    st.subheader("Check against the known activity")
    y_true, y_pred = labelled["true_activity"], labelled["predicted_activity"]
    a, b, c = st.columns(3)
    a.metric("Accuracy", f"{(y_true == y_pred).mean():.3f}")
    b.metric("Balanced accuracy", f"{balanced_accuracy_score(y_true, y_pred):.3f}")
    c.metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.3f}")
    st.dataframe(pd.crosstab(y_true.rename("true"), y_pred.rename("predicted")))
    st.caption(
        "The known activity comes from an 'activity' column or from the file name "
        "(for example Walk.mat). It is only meaningful if it really is the activity, "
        "and these scores say nothing if the file was used to train the model."
    )


def render():
    st.header("Predict activities with a saved model")
    with st.expander("Upload a saved model bundle and new Shimmer data"):
        st.caption("Only load model ZIPs that you created or fully trust (Joblib can execute code).")
        model_file = st.file_uploader("Trained model bundle (.zip)", type=["zip", "joblib"], key="pred_model_file")
        new_files = st.file_uploader("New data: baseline feature CSV or Shimmer MAT", type=["csv", "mat"],
                                     accept_multiple_files=True, key="pred_new_files")

        bundle = None
        if model_file is not None:
            try:
                found = model_io.load_bundles_from_upload(model_file.name, model_file.getvalue())
                choice = st.selectbox("Model in this bundle", list(found)) if len(found) > 1 else next(iter(found))
                bundle = found[choice]
            except Exception as exc:
                st.error(f"Could not load the model: {exc}")

        if bundle is not None:
            ev = bundle.get("evaluation_over_runs") or {}
            st.write(
                f"**{bundle['model_name']}** · trained with the {bundle['protocol']} protocol · "
                f"{bundle.get('n_windows') or '?'} windows · classes: {', '.join(bundle['classes'])}"
            )
            if "balanced_accuracy_mean" in ev:
                st.caption(f"Expected performance (mean balanced accuracy over the runs): {ev['balanced_accuracy_mean']:.2f}. "
                           "New data must come from the same sensor, units, sampling rate and body placement.")

        if st.button("Predict", type="primary", key="pred_button"):
            if bundle is None or not new_files:
                st.warning("Upload a model bundle and at least one data file first.")
            else:
                try:
                    st.session_state["pred_result"] = model_io.predict_files(bundle, new_files)
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

        if st.session_state.get("pred_result") is not None:
            windows, summary = st.session_state["pred_result"]
            st.subheader("One line per file")
            st.dataframe(summary.round(3), hide_index=True, width="stretch")
            st.caption("Majority activity = predicted for most windows of the file. "
                       "A low share or a low confidence means the model is unsure.")
            st.bar_chart(windows.groupby(["source_file", "predicted_activity"]).size().unstack(fill_value=0))
            _show_labelled_check(windows)
            st.subheader("One line per window")
            st.dataframe(windows.round(3), hide_index=True, width="stretch")
            st.download_button("Download window predictions (CSV)", windows.to_csv(index=False).encode("utf-8"),
                               "window_predictions.csv", "text/csv", key="pred_download")
