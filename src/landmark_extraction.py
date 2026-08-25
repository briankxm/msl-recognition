"""
Implements the first part of the training algorithm:

  LOAD dataset
  FOR each image in dataset:
      DETECT hand
      IF hand detected:
          EXTRACT 21 MediaPipe landmarks
          NORMALISE landmarks
          CONVERT landmarks into 63-feature vector
          STORE feature vector + label
  -> saved to data/processed/landmarks_<mode>.csv

Expected raw dataset layout (one folder per class, per mode):

    data/raw/
        alphabet/
            A/   img001.jpg  img002.jpg  ...
            B/   ...
            ...  Z/
        number/
            0/   img001.jpg  ...
            1/   ...
            ...  10/

Run:
    python -m src.landmark_extraction --mode alphabet
    python -m src.landmark_extraction --mode number
    python -m src.landmark_extraction --mode all      # every mode found
"""
import argparse
import os
import csv

import cv2
from tqdm import tqdm

from src import config
from src import hand_detector
from src.normalization import normalise_landmarks


def list_class_folders(raw_dir, mode="alphabet"):
    """Each subfolder of the mode's raw dir is treated as one class label."""
    if not os.path.isdir(raw_dir):
        example = "A, B, C... Z" if mode == "alphabet" else "0, 1, 2... 10"
        raise FileNotFoundError(
            f"Raw data folder not found: {raw_dir}\n"
            f"Create it and add one subfolder per class (e.g. {example}) "
            f"containing the training images."
        )
    return sorted(
        d for d in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, d))
    )


def extract_landmarks_from_image(detector, image_path):
    """
    DETECT hand + EXTRACT 21 landmarks for a single image.
    Returns a (63,) normalised feature vector, or None if no hand detected.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    landmarks = hand_detector.detect(detector, image_rgb)
    if landmarks is None:
        return None  # no hand detected -> skip, per pseudocode's IF

    coords = [(lm.x, lm.y, lm.z) for lm in landmarks]
    return normalise_landmarks(coords)


def build_dataset(mode, raw_dir=None, output_csv=None):
    """Extract landmarks for one mode. raw_dir/output_csv default to the
    mode's configured locations (overridable for tests)."""
    paths = config.mode_paths(mode)
    raw_dir = raw_dir or paths["raw_dir"]
    output_csv = output_csv or paths["csv"]

    class_labels = list_class_folders(raw_dir, mode)
    print(f"\n=== [{mode}] Found {len(class_labels)} classes: {class_labels}")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    rows = []
    skipped = 0

    detector = hand_detector.create(hand_detector.RUNNING_MODE_IMAGE)
    try:
        for label in class_labels:
            class_dir = os.path.join(raw_dir, label)
            image_files = [
                f for f in os.listdir(class_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            for fname in tqdm(image_files, desc=f"Class {label}"):
                image_path = os.path.join(class_dir, fname)
                features = extract_landmarks_from_image(detector, image_path)

                if features is None:
                    skipped += 1
                    continue

                rows.append(list(features) + [label])
    finally:
        detector.close()

    # ---- Write CSV: feature_0 ... feature_62, label ----
    feature_cols = [f"feature_{i}" for i in range(config.FEATURE_LENGTH)]
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(feature_cols + ["label"])
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} samples -> {output_csv}")
    print(f"Skipped {skipped} images (no hand detected)")


def main():
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe hand landmarks from a raw dataset."
    )
    parser.add_argument(
        "--mode", choices=config.MODES + ["all"], default="alphabet",
        help="Which dataset to process (default: alphabet). "
             "'all' processes every mode whose raw folder exists.",
    )
    args = parser.parse_args()

    modes = config.MODES if args.mode == "all" else [args.mode]
    for mode in modes:
        raw_dir = config.mode_paths(mode)["raw_dir"]
        if not os.path.isdir(raw_dir):
            print(f"[skip] {mode}: no dataset at {raw_dir}")
            continue
        build_dataset(mode)


if __name__ == "__main__":
    main()
