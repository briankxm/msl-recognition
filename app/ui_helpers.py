import glob
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from app.inference import (
    draw_hand_overlay,
    extract_features,
    extract_features_video,
    load_models,
    predict_all,
)
from src import config


def pil_to_bgr(pil_image):
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def get_input_image(input_mode, key="upload"):
    image_bgr = None

    if input_mode == "Upload image":
        uploaded = st.file_uploader(
            "Upload a hand photo (JPG/PNG)", type=["jpg", "jpeg", "png"],
            key=f"{key}_file",
        )
        if uploaded is not None:
            image_bgr = pil_to_bgr(Image.open(io.BytesIO(uploaded.getvalue())))

    elif input_mode == "Camera snapshot":
        shot = st.camera_input("Take a photo of your hand", key=f"{key}_camera")
        if shot is not None:
            image_bgr = pil_to_bgr(Image.open(io.BytesIO(shot.getvalue())))

    return image_bgr


def get_reference_images(mode, class_label, max_samples=8):
    class_dir = os.path.join(config.mode_paths(mode)["raw_dir"], class_label)
    if not os.path.isdir(class_dir):
        return []
    extensions = ("*.jpg", "*.jpeg", "*.png")
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(class_dir, ext)))
    files.sort()
    return files[:max_samples]


def render_prediction_panel(results, conf_threshold, developer=False):
    if not results:
        return

    best_conf = max(
        (r["confidence"] for r in results.values() if r["confidence"] is not None),
        default=None,
    )

    if not developer:
        best_result = max(
            (r for r in results.values() if r["confidence"] is not None),
            key=lambda r: r["confidence"],
            default=None,
        )
        if best_result is None:
            st.info("No model could produce a prediction.")
            return

        if best_conf is not None and best_conf * 100 < conf_threshold:
            st.info(
                f"Not confident enough — best confidence is {best_conf * 100:.1f}%, "
                f"below the {conf_threshold}% threshold. Try holding the sign more clearly."
            )

        st.markdown(f"### Prediction: **{best_result['label']}**")

        top = [(label, prob) for label, prob in best_result["top"] if label != best_result["label"]]
        if top:
            second_label, second_prob = top[0]
            st.markdown(f"Second most similar: **{second_label}** ({second_prob * 100:.1f}%)")
        return

    if best_conf is not None and best_conf * 100 < conf_threshold:
        st.info(
            f"Not confident enough — best confidence is {best_conf * 100:.1f}%, "
            f"below the {conf_threshold}% threshold. Try holding the sign more clearly."
        )

    labels = [r["label"] for r in results.values()]
    if len(set(labels)) == 1:
        st.success(f"All {len(labels)} algorithms agree: **{labels[0]}**")
    else:
        st.warning(
            "Algorithms disagree: "
            + ", ".join(f"**{n}** → {r['label']}" for n, r in results.items())
        )

    cols = st.columns(len(results))
    for col, (name, res) in zip(cols, results.items()):
        with col:
            st.metric(name, res["label"])
            conf = res["confidence"]
            if conf is not None:
                pct = conf * 100
                st.progress(min(pct, 100) / 100, text=f"confidence: {pct:.1f}%")
                top = dict(res["top"])
                top.pop(res["label"], None)
                if top:
                    st.caption("Similar signs:")
                    st.bar_chart(pd.Series(top), height=220)


def render_hand_and_predictions(image_bgr, features, hand, models, encoder, conf_threshold, developer=False):
    if image_bgr is None or features is None:
        return None

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
        render_prediction_panel(results, conf_threshold, developer=developer)
    return results
