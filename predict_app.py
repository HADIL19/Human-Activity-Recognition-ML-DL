"""
predict_app.py

Stand-alone page: use a saved model on NEW recordings, nothing is trained.

    python -m streamlit run predict_app.py

The same section is also at the top of the main app (app.py).
"""

import streamlit as st

import predict_ui

st.set_page_config(page_title="Predict with a saved model", layout="wide")
st.title("CDTA Shimmer Activity Recognition")
predict_ui.render()
