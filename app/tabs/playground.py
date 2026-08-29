import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import streamlit as st

try:
    import av
    from streamlit_webrtc import WebRtcMode, webrtc_streamer

    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

from app.inference import (
    draw_hand_overlay,
    extract_features,
    extract_features_video,
    predict_all,
)
from app.ui_helpers import get_input_image, render_prediction_panel, pil_to_bgr
from src import config

MODE_LABELS = {
    "alphabet": "Alphabet (A\u2013Z)",
    "number": "Number (0\u201310)",
}

LIVE_STATE = {"results": None}


def _video_frame_callback(frame, models, encoder):
    img = frame.to_ndarray(format="bgr24")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    annotated = img.copy()

    features, hand = extract_features_video(rgb)
    if features is None:
        cv2.putText(
            annotated, "No hand detected", (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2,
        )
    else:
        draw_hand_overlay(annotated, hand)
        results = predict_all(models, encoder, features)
        LIVE_STATE["results"] = results
        text = " | ".join(f"{n}: {r['label']}" for n, r in results.items())
        cv2.putText(
            annotated, text, (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2,
        )

    return av.VideoFrame.from_ndarray(annotated, format="bgr24")


def render(selected_mode, input_mode, conf_threshold, models, encoder):
    st.subheader(f"Playground \u2014 {MODE_LABELS[selected_mode]}")
    if encoder is not None:
        st.caption("Classes: " + ", ".join(str(c) for c in encoder.classes_))

    if not models:
        st.info(
            "Welcome! This app recognises MSL hand gestures by using "
            "our best lightweight AI model.\n\n"
            "**To get started:**\n"
            f"1. Add images to `data/raw/{selected_mode}/<class>/` "
            f"(e.g. `A/`, `B/`, ... or `0/`...`10/`)\n"
            "2. `python -m src.landmark_extraction --mode all`\n"
            "3. `python -m src.train_models --mode all`\n"
            "4. Reload this page"
        )
        return

    if input_mode == "Live camera":
        if not WEBRTC_AVAILABLE:
            st.error(
                "`streamlit-webrtc` is not installed.\n\n"
                "Run: `pip install streamlit-webrtc`"
            )
            return

        def frame_callback(frame):
            return _video_frame_callback(frame, models, encoder)

        webrtc_streamer(
            key=f"msl-live-{selected_mode}",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]}
            ]},
            media_stream_constraints={"video": True, "audio": False},
            video_frame_callback=frame_callback,
            async_processing=True,
        )
        st.caption(
            "Predictions are drawn live on the video feed. "
            "The detailed panel below "
            "shows the latest analysed frame."
        )
        if LIVE_STATE["results"]:
            render_prediction_panel(LIVE_STATE["results"], conf_threshold, developer=False)
        return

    # --- static image modes (upload / snapshot) ---
    image_bgr = get_input_image(input_mode, key="playground")

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
                render_prediction_panel(results, conf_threshold, developer=False)
