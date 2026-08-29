import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from app.inference import load_models
from src import config

st.set_page_config(
    page_title="MSL Hand Gesture Recognition",
    page_icon="",
    layout="wide",
)

MODE_LABELS = {
    "alphabet": "Alphabet (A\u2013Z)",
    "number": "Number (0\u201310)",
}

with st.sidebar:
    st.header("Settings")

    selected_mode = st.radio(
        "Recognition mode",
        list(MODE_LABELS),
        format_func=MODE_LABELS.get,
    )

    input_mode = st.radio(
        "Input source",
        ["Upload image", "Camera snapshot", "Live camera"],
    )
    conf_threshold = st.slider("Low-confidence alert below (%)", 0, 100, 50, 5)

    st.divider()

    models, encoder = load_models(selected_mode)
    if models:
        st.success(f"**{MODE_LABELS[selected_mode]}** models loaded")
    else:
        st.error(
            f"No models found for {MODE_LABELS[selected_mode]}.\n\nRun:\n"
            f"`python -m src.landmark_extraction --mode {selected_mode}`\n"
            f"`python -m src.train_models --mode {selected_mode}`"
        )

tab_ref, tab_play, tab_quiz, tab_dev = st.tabs([
    "Reference Library",
    "Playground",
    "Quiz",
    "Developer Dashboard",
])

# Reference Library
with tab_ref:
    from app.tabs import reference
    reference.render(selected_mode, encoder)

# Playground
with tab_play:
    from app.tabs import playground
    playground.render(selected_mode, input_mode, conf_threshold, models, encoder)

# Quiz
with tab_quiz:
    from app.tabs import quiz
    quiz.render(selected_mode, input_mode, conf_threshold, models, encoder)

# Developer Mode
with tab_dev:
    from app.tabs import developer
    developer.render(selected_mode, input_mode, conf_threshold, models, encoder)

st.divider()
st.caption("Happy Learning ! This system is built by Brian Kam Ding Xian, Imam Mahdi Ali Ang Attuko, and Lee Boon Yew. @2026")
