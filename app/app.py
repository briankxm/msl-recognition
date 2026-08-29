"""
MSL Hand Gesture Recognition - Streamlit interface.

Run from the project root:
    streamlit run app/app.py

Tabs:
    Reference Library  - grid of reference images per class, no model needed.
    Playground         - free exploration with live prediction and confidence.
    Quiz               - perform a target sign and get proximity-based feedback.
    Developer Dashboard - 3-algorithm comparison, metrics, and confusion matrices.

Modes are chosen in the sidebar: alphabet (A-Z) and number (0-10) are two
independent classification tasks, each with its own trained models under
models/<mode>/.
"""
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

# ------------------------------------------------------------------ sidebar ---
with st.sidebar:
    st.header("Settings")

    selected_mode = st.radio(
        "Recognition mode",
        list(MODE_LABELS),
        format_func=MODE_LABELS.get,
    )

    for m in config.MODES:
        m_models, _ = load_models(m)
        if m == selected_mode:
            continue
        if m_models:
            st.caption(f"{MODE_LABELS[m]}: trained")
        else:
            st.caption(f"{MODE_LABELS[m]}: not trained yet")

    input_mode = st.radio(
        "Input source",
        ["Upload image", "Camera snapshot", "Live camera"],
    )
    conf_threshold = st.slider("Low-confidence alert below (%)", 0, 100, 50, 5)

    st.divider()

    models, encoder = load_models(selected_mode)
    if models:
        n_classes = len(encoder.classes_) if encoder is not None else "?"
        st.success(
            f"**{MODE_LABELS[selected_mode]} models loaded**\n\n"
            + "\n".join(f"- {n}" for n in models)
            + f"\n\n- classes: {n_classes}"
        )
    else:
        st.error(
            f"No models found for {MODE_LABELS[selected_mode]}.\n\nRun:\n"
            f"`python -m src.landmark_extraction --mode {selected_mode}`\n"
            f"`python -m src.train_models --mode {selected_mode}`"
        )

# --------------------------------------------------------------------- tabs ---
tab_ref, tab_play, tab_quiz, tab_dev = st.tabs([
    "Reference Library",
    "Playground",
    "Quiz",
    "Developer Dashboard",
])

with tab_ref:
    from app.tabs import reference
    reference.render(selected_mode, encoder)

with tab_play:
    from app.tabs import playground
    playground.render(selected_mode, input_mode, conf_threshold, models, encoder)

with tab_quiz:
    from app.tabs import quiz
    quiz.render(selected_mode, input_mode, conf_threshold, models, encoder)

with tab_dev:
    from app.tabs import developer
    developer.render(selected_mode, input_mode, conf_threshold, models, encoder)

st.divider()
st.caption("Happy Learning ! This system is built by Brian Kam Ding Xian, Imam Mahdi Ali Ang Attuko, and Lee Boon Yew. @2026")
