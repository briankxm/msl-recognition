"""
Implements the second part of the training algorithm:

  SPLIT dataset into Training / Validation / Testing sets
  FOR each algorithm in [SVM, KNN, RandomForest]:
      TRAIN using training set
      VALIDATE
      TEST using test set
      CALCULATE accuracy, precision, recall, F1, inference time
      SAVE trained model
      SAVE evaluation results

The saved results/<mode>/evaluation_results.(json|csv) files are exactly what
the Streamlit app reads to render the 3-algorithm comparison report (step 9),
per selected mode.

Run:
    python -m src.train_models --mode alphabet
    python -m src.train_models --mode number
    python -m src.train_models --mode all
"""
import argparse
import os
import json
import time

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src import config


def load_dataset(csv_path=config.LANDMARKS_CSV):
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    X = df[feature_cols].values
    y_raw = df["label"].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    return X, y, encoder


def split_dataset(X, y):
    """70 / 15 / 15 train / val / test, stratified so every class is
    represented proportionally in each split."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        train_size=config.TRAIN_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    relative_val_size = config.VAL_SIZE / (config.VAL_SIZE + config.TEST_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        train_size=relative_val_size,
        random_state=config.RANDOM_STATE,
        stratify=y_temp,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_algorithms():
    """The three classifiers being compared, with reasonable defaults.
    Tune these (e.g. via GridSearchCV on the validation set) if accuracy
    needs improving."""
    return {
        "SVM": SVC(kernel="rbf", C=10, gamma="scale", probability=True),
        "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=config.RANDOM_STATE
        ),
    }


def evaluate(model, X, y):
    """Predict + accuracy / precision / recall / F1 (macro-averaged, since
    this is multi-class: 26 letters or 10 digits) + inference time."""
    start = time.perf_counter()
    y_pred = model.predict(X)
    elapsed = time.perf_counter() - start

    n_samples = len(X)
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y, y_pred, average="macro", zero_division=0),
        "f1_score": f1_score(y, y_pred, average="macro", zero_division=0),
        "total_inference_time_sec": elapsed,
        "avg_inference_time_ms": (elapsed / n_samples) * 1000 if n_samples else 0,
    }


def train_and_evaluate_all(mode, csv_path=None, models_dir=None, results_dir=None):
    """Train + evaluate all algorithms for one mode. Paths default to the
    mode's configured locations (overridable for tests)."""
    paths = config.mode_paths(mode)
    csv_path = csv_path or paths["csv"]
    models_dir = models_dir or paths["models_dir"]
    results_dir = results_dir or paths["results_dir"]

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n=== [{mode}] training ===")
    X, y, encoder = load_dataset(csv_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    print(f"Dataset: {len(X)} samples, {len(encoder.classes_)} classes")
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}\n")

    algorithms = get_algorithms()
    all_results = {}

    for name, model in algorithms.items():
        print(f"--- Training {name} ---")

        train_start = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - train_start

        val_metrics = evaluate(model, X_val, y_val)
        test_metrics = evaluate(model, X_test, y_test)

        all_results[name] = {
            "training_time_sec": train_time,
            "validation": val_metrics,
            "test": test_metrics,
        }

        print(f"  Val accuracy:  {val_metrics['accuracy']:.4f}")
        print(f"  Test accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Test F1:       {test_metrics['f1_score']:.4f}")
        print(f"  Avg inference: {test_metrics['avg_inference_time_ms']:.3f} ms/sample")

        # ---- SAVE trained model ----
        model_path = os.path.join(models_dir, f"{name.lower()}_model.pkl")
        joblib.dump(model, model_path)
        print(f"  Saved model -> {model_path}\n")

    # Save label encoder too - the Streamlit app needs it to turn predicted
    # class indices back into letters/numbers at inference time.
    joblib.dump(encoder, os.path.join(models_dir, "label_encoder.pkl"))

    # ---- SAVE evaluation results (feeds step 9: the comparison report) ----
    results_json = os.path.join(results_dir, "evaluation_results.json")
    with open(results_json, "w") as f:
        json.dump(all_results, f, indent=2)

    flat_rows = [
        {
            "algorithm": name,
            "training_time_sec": res["training_time_sec"],
            "val_accuracy": res["validation"]["accuracy"],
            "test_accuracy": res["test"]["accuracy"],
            "test_precision": res["test"]["precision"],
            "test_recall": res["test"]["recall"],
            "test_f1": res["test"]["f1_score"],
            "avg_inference_ms": res["test"]["avg_inference_time_ms"],
        }
        for name, res in all_results.items()
    ]
    results_csv = os.path.join(results_dir, "evaluation_results.csv")
    pd.DataFrame(flat_rows).to_csv(results_csv, index=False)

    print(f"Saved comparison results -> {results_json}\n                            -> {results_csv}")
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Train + evaluate SVM / KNN / RandomForest for one mode."
    )
    parser.add_argument(
        "--mode", choices=config.MODES + ["all"], default="alphabet",
        help="Which dataset to train on (default: alphabet). "
             "'all' trains every mode whose extracted CSV exists.",
    )
    args = parser.parse_args()

    modes = config.MODES if args.mode == "all" else [args.mode]
    for mode in modes:
        csv_path = config.mode_paths(mode)["csv"]
        if not os.path.exists(csv_path):
            print(f"[skip] {mode}: no landmarks CSV at {csv_path} - "
                  f"run `python -m src.landmark_extraction --mode {mode}` first")
            continue
        train_and_evaluate_all(mode)


if __name__ == "__main__":
    main()
