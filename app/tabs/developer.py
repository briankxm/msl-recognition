import json
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

    st.markdown("### Model Comparison Report")
    from src import config

    mode_paths = config.mode_paths(selected_mode)
    results_json = os.path.join(mode_paths["results_dir"], "evaluation_results.json")

    if not os.path.exists(results_json):
        st.info(
            f"No evaluation results for {MODE_LABELS[selected_mode]} yet.\n\nRun:\n"
            f"```bash\npython -m src.landmark_extraction --mode {selected_mode}\n"
            f"python -m src.train_models --mode {selected_mode}\n```\n"
            f"then reload this page."
        )
        return

    with open(results_json, "r") as f:
        eval_data = json.load(f)

    rows = []
    for algo_name, algo_data in eval_data.items():
        rows.append({
            "algorithm": algo_name,
            "training_time_sec": algo_data["training_time_sec"],
            "val_accuracy": algo_data["validation"]["accuracy"],
            "test_accuracy": algo_data["test"]["accuracy"],
            "test_precision": algo_data["test"]["precision"],
            "test_recall": algo_data["test"]["recall"],
            "test_f1": algo_data["test"]["f1_score"],
            "test_top3_accuracy": algo_data["top3_accuracy"],
            "test_top5_accuracy": algo_data["top5_accuracy"],
            "avg_inference_ms": algo_data["test"]["avg_inference_time_ms"],
        })
    df = pd.DataFrame(rows).set_index("algorithm")

    metric_cols = [
        "test_accuracy", "test_precision", "test_recall", "test_f1",
        "test_top3_accuracy", "test_top5_accuracy",
    ]
    df_display = df[metric_cols + ["val_accuracy", "training_time_sec", "avg_inference_ms"]]
    st.dataframe(df_display.style.format("{:.4f}"), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Test metrics by algorithm")
        st.bar_chart(df[metric_cols])
    with c2:
        st.caption("Validation vs test accuracy")
        st.bar_chart(df[["val_accuracy", "test_accuracy"]])

    st.caption("Average inference time per sample (ms) \u2014 lower is better")
    st.bar_chart(df["avg_inference_ms"])

    st.divider()
    st.markdown("### Confusion Matrix")

    algo = st.selectbox("Algorithm", list(eval_data.keys()))

    algo_data = eval_data[algo]
    cm = algo_data["confusion_matrix"]
    class_names = algo_data["classes"]

    show_text = len(class_names) <= 20
    fig = px.imshow(
        cm,
        x=class_names,
        y=class_names,
        text_auto=".0f" if show_text else False,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="True", color="count"),
        title=f"{algo} \u2014 confusion matrix",
    )
    fig.update_layout(height=620)
    st.plotly_chart(fig, width="stretch")

    error_pairs = algo_data.get("error_pairs", [])
    n_errors = algo_data.get("n_errors", 0)

    if n_errors > 0:
        st.caption(f"**{n_errors}** misclassified sample(s) in the test set:")
        for ep in error_pairs:
            st.write(f"- True **{ep['true']}** \u2192 Predicted **{ep['pred']}** ({ep['count']}x)")
    else:
        st.success("No misclassifications \u2014 perfect accuracy on the test set!")
