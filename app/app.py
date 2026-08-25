"""
MSL Hand Gesture Recognition - Streamlit interface.

Run from the project root:
    streamlit run app/app.py

Tabs:
    Prediction        - upload / snapshot / live camera, all 3 algorithms
                        side by side with top-5 "similar signs".
    Model Comparison  - metrics table + charts from results/<mode>/ and
                        per-algorithm confusion matrices on the test split.

Modes are chosen in the sidebar: alphabet (A-Z) and number (0-10) are two
independent classification tasks, each with its own trained models under
models/<mode>/.
"""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    import av
    from streamlit_webrtc import WebRtcMode, webrtc_streamer

    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

import plotly.express as px

from app.inference import (
    draw_hand_overlay,
    extract_features,
    extract_features_video,
    get_static_hands,
    load_models,
    predict_all,
)
from src import config

st.set_page_config(
    page_title="MSL Hand Gesture Recognition",
    page_icon="🤟",
    layout="wide",
)

MODE_LABELS = {
    "alphabet": "🔤 Alphabet (A–Z)",
    "number": "🔢 Number (0–10)",
}

# Shared with the webrtc worker thread (dict assignment is atomic under GIL).
LIVE_STATE = {"results": None}


def pil_to_bgr(pil_image):
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def render_predictions(results):
    """Three columns - one per algorithm - each with label, confidence and
    a top-K similar-signs bar chart. Plus an agreement badge on top."""
    labels = [r["label"] for r in results.values()]
    if len(set(labels)) == 1:
        st.success(f"✅ All {len(labels)} algorithms agree: **{labels[0]}**")
    else:
        st.warning(f"⚠️ Algorithms disagree: " + ", ".join(
            f"**{n}** → {r['label']}" for n, r in results.items()
        ))

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


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    annotated = img.copy()

    features, hand = extract_features_video(rgb)
    if features is None:
        cv2.putText(annotated, "No hand detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)
    else:
        draw_hand_overlay(annotated, hand)
        results = predict_all(models, encoder, features)
        LIVE_STATE["results"] = results
        text = " | ".join(f"{n}: {r['label']}" for n, r in results.items())
        cv2.putText(annotated, text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ---------------------------------------------------------------- sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")

    # ---- recognition mode: two independent tasks, two model sets ----
    selected_mode = st.radio(
        "Recognition mode",
        list(MODE_LABELS),
        format_func=MODE_LABELS.get,
    )

    # overview of which modes are trained (cached, cheap after first run)
    for m in config.MODES:
        m_models, _ = load_models(m)
        if m == selected_mode:
            continue
        if m_models:
            st.caption(f"{MODE_LABELS[m]}: ✅ trained")
        else:
            st.caption(f"{MODE_LABELS[m]}: ⬜ not trained yet")

    input_mode = st.radio(
        "Input source",
        ["📁 Upload image", "📸 Camera snapshot", "🎥 Live camera"],
    )
    conf_threshold = st.slider("Low-confidence alert below (%)", 0, 100, 50, 5)

    st.divider()

    # load the ACTIVE mode's models (cached per mode)
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

# ------------------------------------------------------------------- tabs ---
tab_pred, tab_report = st.tabs(["🔮 Prediction", "📊 Model Comparison"])

# ---------------------------------------------------------- prediction tab --
with tab_pred:
    st.subheader(f"Recognise a hand sign — {MODE_LABELS[selected_mode]}")
    if encoder is not None:
        st.caption("Classes: " + ", ".join(str(c) for c in encoder.classes_))

    image_bgr = None

    if input_mode == "📁 Upload image":
        uploaded = st.file_uploader("Upload a hand photo (JPG/PNG)",
                                    type=["jpg", "jpeg", "png"])
        if uploaded is not None and models:
            image_bgr = pil_to_bgr(Image.open(io.BytesIO(uploaded.getvalue())))

    elif input_mode == "📸 Camera snapshot":
        shot = st.camera_input("Take a photo of your hand")
        if shot is not None and models:
            image_bgr = pil_to_bgr(Image.open(io.BytesIO(shot.getvalue())))

    else:  # live camera
        if not WEBRTC_AVAILABLE:
            st.error("`streamlit-webrtc` is not installed.\n\nRun: "
                     "`pip install streamlit-webrtc`")
        elif not models:
            st.info("Train the models first (see sidebar).")
        else:
            webrtc_streamer(
                key=f"msl-live-{selected_mode}",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration={"iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]}
                ]},
                media_stream_constraints={"video": True, "audio": False},
                video_frame_callback=video_frame_callback,
                async_processing=True,
            )
            st.caption("Predictions are drawn live on the video feed "
                       "(SVM | KNN | RandomForest). The detailed panel below "
                       "shows the latest analysed frame.")
            if LIVE_STATE["results"]:
                render_predictions(LIVE_STATE["results"])

    if not models and image_bgr is None:
        st.info("👋 Welcome! This app recognises MSL hand gestures using the "
                "trained SVM / KNN / Random Forest models.\n\n"
                "**To get started:**\n"
                f"1. Add images to `data/raw/{selected_mode}/<class>/` "
                f"(e.g. `A/`, `B/`, ... or `0/`...`10/`)\n"
                "2. `python -m src.landmark_extraction --mode all`\n"
                "3. `python -m src.train_models --mode all`\n"
                "4. Reload this page")

    if image_bgr is not None:
        if not models:
            st.error("Models are missing - train them first (see sidebar).")
        else:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            features, hand = extract_features(rgb)

            if features is None:
                st.warning("✋ No hand detected. Try better lighting, fill more "
                           "of the frame with your hand, or use another photo.")
            else:
                left, right = st.columns([1, 2])
                with left:
                    shown = draw_hand_overlay(image_bgr.copy(), hand)
                    st.image(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB),
                             caption="Detected hand landmarks",
                             use_container_width=True)
                with right:
                    results = predict_all(models, encoder, features)
                    best_conf = max(
                        (r["confidence"] for r in results.values() if r["confidence"]),
                        default=None,
                    )
                    if best_conf is not None and best_conf * 100 < conf_threshold:
                        st.info(f"🐢 Low confidence ({best_conf * 100:.1f}% < "
                                f"{conf_threshold}%). Try holding the sign more "
                                f"clearly.")
                    render_predictions(results)

# ------------------------------------------------------------- report tab ---
with tab_report:
    st.subheader(f"3-algorithm comparison report — {MODE_LABELS[selected_mode]}")
    mode_paths = config.mode_paths(selected_mode)
    results_csv = os.path.join(mode_paths["results_dir"], "evaluation_results.csv")

    if not os.path.exists(results_csv):
        st.info(f"No evaluation results for {MODE_LABELS[selected_mode]} yet.\n\nRun:\n"
                f"```bash\npython -m src.landmark_extraction --mode {selected_mode}\n"
                f"python -m src.train_models --mode {selected_mode}\n```\n"
                f"then reload this page.")
    else:
        df = pd.read_csv(results_csv).set_index("algorithm")
        metric_cols = ["test_accuracy", "test_precision", "test_recall",
                       "test_f1"]
        df_display = df[metric_cols + ["val_accuracy", "training_time_sec",
                                       "avg_inference_ms"]]
        st.dataframe(
            df_display.style.format("{:.4f}"),
            use_container_width=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.caption("Test metrics by algorithm")
            st.bar_chart(df[metric_cols])
        with c2:
            st.caption("Validation vs test accuracy")
            st.bar_chart(df[["val_accuracy", "test_accuracy"]])

        st.caption("Average inference time per sample (ms) - lower is better")
        st.bar_chart(df["avg_inference_ms"])

        # ---- confusion matrix on the exact same test split as training ----
        st.divider()
        st.subheader("Confusion matrix (test split)")
        landmarks_csv = mode_paths["csv"]
        if not os.path.exists(landmarks_csv):
            st.caption(f"`{os.path.relpath(landmarks_csv)}` not found - "
                       f"re-run landmark extraction to enable this chart.")
        elif not models:
            st.caption("Load models first to compute confusion matrices.")
        else:
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
                title=f"{algo} - confusion matrix on the held-out test set",
            )
            fig.update_layout(height=620)
            st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("Labels come straight from each mode's "
           "`models/<mode>/label_encoder.pkl` - alphabet uses folder names "
           "A-Z, number uses 0-10.")
