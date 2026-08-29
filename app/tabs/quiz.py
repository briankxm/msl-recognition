import random
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np
import streamlit as st

from app.inference import draw_hand_overlay, extract_features, predict_all
from app.ui_helpers import get_input_image, get_reference_images

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

    if st.button("Pick random class", width="stretch"):
        st.session_state["quiz_target"] = random.choice(classes)

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

    best_result = results.get("SVM")

    if best_result is None:
        st.info("No model could produce a prediction.")
    else:
        predicted = best_result["label"]
        st.markdown(f"### Prediction: **{predicted}**")

        target_conf = None
        for name, model in models.items():
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                for i, cls in enumerate(model.classes_):
                    cls_label = str(encoder.inverse_transform([i])[0]) if encoder else str(cls)
                    if cls_label == target:
                        prob = float(proba[i])
                        if target_conf is None or prob > target_conf:
                            target_conf = prob
                        break

        if target_conf is not None:
            score_pct = target_conf * 100
            label = _score_label(score_pct)
            st.progress(
                min(score_pct, 100) / 100,
                text=f"{score_pct:.1f}% \u2014 {label}",
            )

            if predicted == target:
                st.success("Correct!")
            else:
                top = [(l, p) for l, p in best_result["top"] if l != predicted]
                if top:
                    second_label, second_prob = top[0]
                    st.warning(
                        f"Your gesture most resembled **{second_label}** ({second_prob * 100:.1f}%)"
                    )
        else:
            if predicted == target:
                st.success("Correct!")
            else:
                st.warning(f"Your gesture most resembled **{predicted}**")
                
    ref_images = get_reference_images(selected_mode, target, max_samples=1)
    if ref_images:
        st.image(ref_images[0], caption=f"Reference: {target}", width=250)


    if "quiz_history" not in st.session_state:
        st.session_state["quiz_history"] = []

    attempt = {
        "target": target,
        "predicted": predicted if best_result else None,
        "confidence": target_conf if best_result else None,
    }
    st.session_state["quiz_history"].append(attempt)

    if len(st.session_state["quiz_history"]) > 1:
        st.divider()
        st.subheader("Attempt History")
        history = st.session_state["quiz_history"]
        for i, a in enumerate(reversed(history[-10:]), 1):
            conf_pct = a["confidence"] * 100 if a["confidence"] else 0
            status = "Correct" if a["predicted"] == a["target"] else f"Predicted {a['predicted']}"
            st.caption(
                f"#{len(history) - i + 1}: Target **{a['target']}** \u2014 "
                f"{status} ({conf_pct:.1f}%)"
            )
