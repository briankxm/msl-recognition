import numpy as np


def normalise_landmarks(landmarks, handedness=None):
    pts = np.array(landmarks, dtype=np.float32)

    # Mirror left hands to right hands for consistent training data
    if handedness == "Left":
        pts[:, 0] = 1.0 - pts[:, 0]  # Horizontal flip

    # 1. Translate relative to the wrist (landmark index 0)
    wrist = pts[0].copy()
    pts -= wrist

    # 2. Scale by the largest distance from the wrist to any landmark
    distances = np.linalg.norm(pts, axis=1)
    max_dist = distances.max()
    if max_dist > 1e-6:  # avoid divide-by-zero on degenerate detections
        pts /= max_dist

    return pts.flatten()
