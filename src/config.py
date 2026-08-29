import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root

RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
LANDMARKS_CSV = os.path.join(PROCESSED_DATA_DIR, "landmarks_dataset.csv")  # legacy

MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

MODES = ["alphabet", "number"]


def mode_paths(mode):
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}. Choose one of {MODES}.")
    return {
        "raw_dir": os.path.join(RAW_DATA_DIR, mode),
        "csv": os.path.join(PROCESSED_DATA_DIR, f"landmarks_{mode}.csv"),
        "models_dir": os.path.join(MODELS_DIR, mode),
        "results_dir": os.path.join(RESULTS_DIR, mode),
    }

HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
HAND_LANDMARKER_TASK = os.path.join(MODELS_DIR, "hand_landmarker.task")

MP_MAX_NUM_HANDS = 1
MP_MIN_DETECTION_CONFIDENCE = 0.5

NUM_LANDMARKS = 21
NUM_COORDS = 3
FEATURE_LENGTH = NUM_LANDMARKS * NUM_COORDS

TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_STATE = 42

ALGORITHMS = ["SVM", "MLP", "RandomForest"]
