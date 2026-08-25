"""
Central configuration for the MSL Recognition System.
Edit these paths/settings to match your dataset and environment.
"""
import os

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root

RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
LANDMARKS_CSV = os.path.join(PROCESSED_DATA_DIR, "landmarks_dataset.csv")  # legacy

MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ---- Modes ----
# Two independent classification tasks, each with its own dataset, models
# and evaluation results:
#   "alphabet" -> data/raw/alphabet/A ... Z        (26 classes)
#   "number"   -> data/raw/number/0 ... 10         (11 classes)
MODES = ["alphabet", "number"]


def mode_paths(mode):
    """Per-mode dataset/model/result locations.

    Returns a dict with keys: raw_dir, csv, models_dir, results_dir.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}. Choose one of {MODES}.")
    return {
        "raw_dir": os.path.join(RAW_DATA_DIR, mode),
        "csv": os.path.join(PROCESSED_DATA_DIR, f"landmarks_{mode}.csv"),
        "models_dir": os.path.join(MODELS_DIR, mode),
        "results_dir": os.path.join(RESULTS_DIR, mode),
    }

# ---- MediaPipe HandLandmarker (Tasks API, mediapipe >= 1.0) ----
# The legacy mp.solutions API was removed; detection now uses the
# HandLandmarker task, whose .task model file is downloaded once on first use.
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
HAND_LANDMARKER_TASK = os.path.join(MODELS_DIR, "hand_landmarker.task")

# ---- MediaPipe Hands settings (shared by training + app) ----
MP_MAX_NUM_HANDS = 1
MP_MIN_DETECTION_CONFIDENCE = 0.5

# ---- Landmark feature settings ----
NUM_LANDMARKS = 21
NUM_COORDS = 3          # x, y, z
FEATURE_LENGTH = NUM_LANDMARKS * NUM_COORDS  # 63

# ---- Train / Validation / Test split ----
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_STATE = 42

# ---- Algorithms to train/compare ----
ALGORITHMS = ["SVM", "KNN", "RandomForest"]
