"""
Landmark normalisation.

MediaPipe returns 21 hand landmarks, each with (x, y, z) in image-relative
coordinates. Raw coordinates are NOT directly comparable across hand sizes,
distances from the camera, or positions in frame - so every feature vector
is normalised before it is fed to the classifiers (and before inference,
using this exact same function, so train and inference stay consistent).

Method:
  1. Translate - shift every landmark so the wrist (landmark 0) sits at the
                 origin. Removes dependence on WHERE the hand is in frame.
  2. Scale     - divide every coordinate by the largest distance from the
                 wrist to any other landmark. Removes dependence on hand
                 size / distance from camera.
"""
import numpy as np


def normalise_landmarks(landmarks):
    """
    Args:
        landmarks: iterable of 21 (x, y, z) tuples from MediaPipe.

    Returns:
        np.ndarray of shape (63,) - flattened, translation- and
        scale-invariant feature vector, ready for the classifiers.
    """
    pts = np.array(landmarks, dtype=np.float32)  # shape (21, 3)

    # 1. Translate relative to the wrist (landmark index 0)
    wrist = pts[0].copy()
    pts -= wrist

    # 2. Scale by the largest distance from the wrist to any landmark
    distances = np.linalg.norm(pts, axis=1)
    max_dist = distances.max()
    if max_dist > 1e-6:  # avoid divide-by-zero on degenerate detections
        pts /= max_dist

    return pts.flatten()  # (63,)
