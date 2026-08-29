import random
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np
import streamlit as st

from app.inference import draw_hand_overlay, extract_features, predict_all
from app.ui_helpers import get_input_image

MODE_LABELS = {
    "alphabet": "Alphabet (A\u2013Z)",
    "number": "Number (0\u201310)",
}


def _score_label(score_pct):
    if score_pct >= 80:
        return "Excellent!"
    elif score_pct >= 60:
        return "Good"
    elif score_pct >= 40:
        return "Needs work"
    else:
        return "Not quite"


def render(selected_mode, input_mode, conf_threshold, models, encoder):
    st.subheader(f"Quiz \u2014 {MODE_LABELS[selected_mode]}")

    if not models or encoder is None:
        st.info("Train the models first to use Quiz mode (see sidebar).")
        return

    classes = [str(c) for c in encoder.classes_]

    def _on_quiz_class_change():
        st.session_state["quiz_target"] = st.session_state["quiz_select"]

    col_random, col_select = st.columns(2)
    with col_random:
        if st.button("Pick random class", width="stretch"):
            st.session_state["quiz_target"] = random.choice(classes)
    with col_select:
        st.selectbox(
            "Or choose a class to practice",
            classes,
            index=0,
            key="quiz_select",
            on_change=_on_quiz_class_change,
        )

    target = st.session_state.get("quiz_target")
    if not target:
        st.info("Click **Pick random class** or select a class above to begin.")
        return

    st.success(f"**Perform the sign for: {target}**")
    st.caption("Take a photo or upload an image of your attempt.")

    if input_mode == "Live camera":
        st.info("Quiz mode works with **Upload image** or **Camera snapshot**. "
                "Switch the input source in the sidebar.")
        return

    image_bgr = get_input_image(input_mode, key="quiz")

    if image_bgr is None:
        return

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    features, hand = extract_features(rgb)

    if features is None:
        st.warning(
            "No hand detected. Try better lighting, fill more "
            "of the frame with your hand, or use another photo."
        )
        return

    shown = draw_hand_overlay(image_bgr.copy(), hand)
    st.image(
        cv2.cvtColor(shown, cv2.COLOR_BGR2RGB),
        caption="Detected hand landmarks",
        width="stretch",
    )

    results = predict_all(models, encoder, features)

    st.divider()
    st.subheader("Results")

    X = np.asarray(features, dtype=np.float32).reshape(1, -1)

    algo_results = {}
    for name, model in models.items():
        target_conf = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            class_idx = None
            for i, cls in enumerate(model.classes_):
                cls_label = str(encoder.inverse_transform([i])[0]) if encoder is not None else str(cls)
                if cls_label == target:
                    class_idx = i
                    break
            if class_idx is not None:
                target_conf = float(proba[class_idx])

        # change name to best model
        algo_results[name] = {
            "predicted": results[name]["label"],
            "target_confidence": target_conf,
            "top": results[name]["top"],
        }

    cols = st.columns(len(algo_results))
    for col, (name, data) in zip(cols, algo_results.items()):
        with col:
            st.metric(name, f"Predicted: {data['predicted']}")

            if data["target_confidence"] is not None:
                score_pct = data["target_confidence"] * 100
                label = _score_label(score_pct)
                st.progress(
                    min(score_pct, 100) / 100,
                    text=f"{score_pct:.1f}% \u2014 {label}",
                )

                if data["predicted"] == target:
                    st.success("Correct!")
                else:
                    top_label = data["top"][0][0] if data["top"] else "?"
                    top_conf = data["top"][0][1] * 100 if data["top"] else 0
                    st.warning(
                        f"Your gesture most resembled **{top_label}** ({top_conf:.1f}%)"
                    )
            else:
                st.caption("Model does not support probability estimates.")

    if "quiz_history" not in st.session_state:
        st.session_state["quiz_history"] = []

    attempt = {
        "target": target,
        "scores": {
            name: data["target_confidence"]
            for name, data in algo_results.items()
            if data["target_confidence"] is not None
        },
        "predicted": {name: data["predicted"] for name, data in algo_results.items()},
    }
    st.session_state["quiz_history"].append(attempt)

    if len(st.session_state["quiz_history"]) > 1:
        st.divider()
        st.subheader("Attempt History")
        history = st.session_state["quiz_history"]
        for i, a in enumerate(reversed(history[-10:]), 1):
            avg_score = (
                np.mean(list(a["scores"].values())) * 100
                if a["scores"]
                else 0
            )
            st.caption(
                f"#{len(history) - i + 1}: Target **{a['target']}** \u2014 "
                f"avg proximity {avg_score:.1f}% \u2014 "
                f"predicted {', '.join(a['predicted'].values())}"
            )
