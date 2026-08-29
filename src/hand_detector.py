import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mediapipe as mp

from src import config

RUNNING_MODE_IMAGE = mp.tasks.vision.RunningMode.IMAGE
RUNNING_MODE_VIDEO = mp.tasks.vision.RunningMode.VIDEO


def ensure_model_file():
    if os.path.exists(config.HAND_LANDMARKER_TASK):
        return
    os.makedirs(os.path.dirname(config.HAND_LANDMARKER_TASK), exist_ok=True)
    print(f"Downloading MediaPipe hand model -> {config.HAND_LANDMARKER_TASK}")
    urllib.request.urlretrieve(config.HAND_LANDMARKER_URL, config.HAND_LANDMARKER_TASK)
    print("Download complete.")


def create(running_mode=RUNNING_MODE_IMAGE):
    ensure_model_file()

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=config.HAND_LANDMARKER_TASK,
        ),
        running_mode=running_mode,
        num_hands=config.MP_MAX_NUM_HANDS,
        min_hand_detection_confidence=config.MP_MIN_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=config.MP_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MP_MIN_DETECTION_CONFIDENCE,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def detect(detector, image_rgb):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = detector.detect(mp_image)
    if not result.hand_landmarks:
        return None, None
    landmarks = result.hand_landmarks[0]
    mp_hand = result.handedness[0][0].category_name  # Left or Right
    true_hand = "Right" if mp_hand == "Left" else "Left"
    return landmarks, true_hand


def detect_video(detector, image_rgb, timestamp_ms):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = detector.detect_for_video(mp_image, int(timestamp_ms))
    if not result.hand_landmarks:
        return None, None
    landmarks = result.hand_landmarks[0]
    mp_hand = result.handedness[0][0].category_name  # Left or Right
    true_hand = "Right" if mp_hand == "Left" else "Left"
    return landmarks, true_hand
