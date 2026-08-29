"""
Inference helpers for the Streamlit app.

Loads the classifiers saved by src/train_models.py and turns raw
camera/uploaded images into normalised 63-feature landmark vectors using the
exact same MediaPipe settings + normalisation as training, so live
predictions stay consistent with the offline evaluation.

Detection uses the MediaPipe Tasks HandLandmarker via src/hand_detector.py
(mediapipe >= 1.0 removed the legacy mp.solutions API).
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import joblib
import numpy as np
import streamlit as st

from src import config
from src import hand_detector
from src.normalization import normalise_landmarks

MODEL_FILES = {
    "svm_model.pkl": "SVM",
    "mlp_model.pkl": "MLP",
    "randomforest_model.pkl": "RandomForest",
}

TOP_K = 5

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),                   # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),                   # index finger
    (5, 9), (9, 10), (10, 11), (11, 12),              # middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),            # ring finger
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # pinky + palm base
]


@st.cache_resource(show_spinner="Loading models...")
def load_models(mode):
    models_dir = config.mode_paths(mode)["models_dir"]

    models = {}
    for fname, display_name in MODEL_FILES.items():
        path = os.path.join(models_dir, fname)
        if os.path.exists(path):
            models[display_name] = joblib.load(path)

    encoder = None
    encoder_path = os.path.join(models_dir, "label_encoder.pkl")
    if os.path.exists(encoder_path):
        encoder = joblib.load(encoder_path)
    return models, encoder


@st.cache_resource(show_spinner="Starting MediaPipe HandLandmarker...")
def get_static_hands():
    return hand_detector.create(hand_detector.RUNNING_MODE_IMAGE)


_live_hands = None
_live_lock = threading.Lock()
_live_last_ts = 0


def get_live_hands():
    global _live_hands
    with _live_lock:
        if _live_hands is None:
            _live_hands = hand_detector.create(hand_detector.RUNNING_MODE_VIDEO)
    return _live_hands


def extract_features(image_rgb):
    landmarks, handedness = hand_detector.detect(get_static_hands(), image_rgb)
    if landmarks is None:
        return None, None
    coords = [(lm.x, lm.y, lm.z) for lm in landmarks]
    return normalise_landmarks(coords, handedness), landmarks


def extract_features_video(image_rgb):
    global _live_last_ts

    ts = time.perf_counter() * 1000
    if ts <= _live_last_ts:
        ts = _live_last_ts + 1
    _live_last_ts = ts

    landmarks, handedness = hand_detector.detect_video(get_live_hands(), image_rgb, ts)
    if landmarks is None:
        return None, None
    coords = [(lm.x, lm.y, lm.z) for lm in landmarks]
    return normalise_landmarks(coords, handedness), landmarks


def predict_all(models, encoder, features):
    X = np.asarray(features, dtype=np.float32).reshape(1, -1)
    results = {}

    for name, model in models.items():
        pred_idx = int(model.predict(X)[0])

        if encoder is not None:
            label = str(encoder.inverse_transform([pred_idx])[0])
        else:
            label = str(pred_idx)

        confidence, top = None, []
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            confidence = float(proba[pred_idx]) if pred_idx < len(proba) else float(proba.max())

            order = np.argsort(proba)[::-1][:TOP_K]
            classes = model.classes_[order]
            names = encoder.inverse_transform(classes) if encoder is not None else classes
            top = [(str(n), float(p)) for n, p in zip(names, proba[order])]

        results[name] = {"label": label, "confidence": confidence, "top": top}

    return results


def draw_hand_overlay(image_bgr, normalized_landmarks):
    height, width = image_bgr.shape[:2]
    pts = [(int(lm.x * width), int(lm.y * height)) for lm in normalized_landmarks]

    for a, b in HAND_CONNECTIONS:
        cv2.line(image_bgr, pts[a], pts[b], (255, 255, 255), 2)
    for p in pts:
        cv2.circle(image_bgr, p, 3, (0, 255, 0), -1)

    return image_bgr
