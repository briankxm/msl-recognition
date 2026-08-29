import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import pandas as pd
import plotly.express as px
import streamlit as st

from app.inference import (
    draw_hand_overlay,
    extract_features,
    predict_all,
)
from app.ui_helpers import get_input_image, render_prediction_panel

MODE_LABELS = {
    "alphabet": "Alphabet (A to Z)",
    "number": "Number (0 to 10)",
}


def render(selected_mode, input_mode, conf_threshold, models, encoder):
    st.subheader(f"Developer Dashboard \u2014 {MODE_LABELS[selected_mode]}")

    if not models:
        st.info("Train models first (see sidebar).")
        return

    st.markdown("### Live Prediction")

    if input_mode == "Live camera":
        st.info(
            "You can toggle Live Camera Input Method in Playground Tab. "
            "If you want detailed analysis, please upload static image or use the camera to capture static images."
            "Allowed File Format: JPG, PNG."
        )
    else:
        image_bgr = get_input_image(input_mode, key="developer")

        if image_bgr is not None:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            features, hand = extract_features(rgb)

            if features is None:
                st.warning(
                    "No hand detected. Try better lighting, fill more "
                    "of the frame with your hand, or use another photo."
                )
            else:
                left, right = st.columns([1, 2])
                with left:
                    shown = draw_hand_overlay(image_bgr.copy(), hand)
                    st.image(
                        cv2.cvtColor(shown, cv2.COLOR_BGR2RGB),
                        caption="Detected hand landmarks",
                        width="stretch",
                    )
                with right:
                    results = predict_all(models, encoder, features)
                    render_prediction_panel(results, conf_threshold, developer=True)

    st.divider()

    # --- evaluation metrics ---
    st.markdown("### Model Comparison Report")
    from src import config

    mode_paths = config.mode_paths(selected_mode)
    results_csv = os.path.join(mode_paths["results_dir"], "evaluation_results.csv")

    if not os.path.exists(results_csv):
        st.info(
            f"No evaluation results for {MODE_LABELS[selected_mode]} yet.\n\nRun:\n"
            f"```bash\npython -m src.landmark_extraction --mode {selected_mode}\n"
            f"python -m src.train_models --mode {selected_mode}\n```\n"
            f"then reload this page."
        )
        return

    df = pd.read_csv(results_csv).set_index("algorithm")

    # Metrics table
    metric_cols = [
        "test_accuracy", "test_precision", "test_recall", "test_f1",
        "test_top3_accuracy", "test_top5_accuracy",
    ]
    df_display = df[metric_cols + ["val_accuracy", "training_time_sec", "avg_inference_ms"]]
    st.dataframe(df_display.style.format("{:.4f}"), width="stretch")

    # Comparison charts
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Test metrics by algorithm")
        st.bar_chart(df[metric_cols])
    with c2:
        st.caption("Validation vs test accuracy")
        st.bar_chart(df[["val_accuracy", "test_accuracy"]])

    st.caption("Average inference time per sample (ms) \u2014 lower is better")
    st.bar_chart(df["avg_inference_ms"])

    # --- confusion matrix ---
    st.divider()
    st.markdown("### Confusion Matrix (test split)")

    landmarks_csv = mode_paths["csv"]
    if not os.path.exists(landmarks_csv):
        st.caption(
            f"`{os.path.relpath(landmarks_csv)}` not found \u2014 "
            f"re-run landmark extraction to enable this chart."
        )
        return

    algo = st.selectbox("Algorithm", list(models))

    @st.cache_data(show_spinner="Preparing test split...")
    def test_split(csv_path):
        from src.train_models import load_dataset, split_dataset

        X, y, enc = load_dataset(csv_path)
        _, _, X_test, _, _, y_test = split_dataset(X, y)
        return X_test, y_test, list(enc.classes_)

    X_test, y_test, class_names = test_split(landmarks_csv)
    y_pred = models[algo].predict(X_test)

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred)
    show_text = len(class_names) <= 20
    fig = px.imshow(
        cm,
        x=class_names,
        y=class_names,
        text_auto=".0f" if show_text else False,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="True", color="count"),
        title=f"{algo} \u2014 confusion matrix on the held-out test set",
    )
    fig.update_layout(height=620)
    st.plotly_chart(fig, width="stretch")
