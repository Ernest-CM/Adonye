"""
Speech data loader — UCI Oxford Parkinson's Disease Detection Dataset.

Dataset: 195 voice recordings from 31 subjects (23 PD, 8 HC).
Each row is one sustained /a/ phonation recording.
22 pre-extracted acoustic features per recording.

Download from:
  https://archive.ics.uci.edu/dataset/174/parkinsons
Place the CSV file at: data/raw/speech/parkinsons.data
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib
from typing import Tuple, Dict

from src.config import (
    SPEECH_FEATURES, RANDOM_SEED, SPLIT_RATIOS,
    SPEECH_PROCESSED_DIR, SPEECH_SPLITS_PATH,
)
from src.data.split_utils import (
    subject_wise_stratified_split,
    collect_samples_for_split,
    save_splits,
    load_splits,
)


def load_speech_data(
    raw_dir: str,
    processed_dir: str = SPEECH_PROCESSED_DIR,
    splits_path: str = SPEECH_SPLITS_PATH,
    force_reprocess: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (X_train, y_train, X_val, y_val, X_test, y_test).
    Fits StandardScaler on training subjects only.
    """
    cache_file = os.path.join(processed_dir, "speech_splits.npz")

    if not force_reprocess and os.path.exists(cache_file):
        data = np.load(cache_file)
        return (
            data["X_train"], data["y_train"],
            data["X_val"], data["y_val"],
            data["X_test"], data["y_test"],
        )

    csv_path = _find_csv(raw_dir)
    df = pd.read_csv(csv_path)

    # UCI dataset has a 'name' column (subject identifier) and 'status' (0=HC, 1=PD)
    df.columns = [c.strip() for c in df.columns]
    subject_col = "name"
    label_col = "status"

    subjects = df[subject_col].tolist()
    labels = df[label_col].tolist()
    feature_df = df[SPEECH_FEATURES]

    # Build subject→label mapping (one recording per subject may appear multiple times)
    subject_label = {}
    for s, l in zip(subjects, labels):
        subject_label[s] = int(l)

    unique_subjects = list(subject_label.keys())
    unique_labels = [subject_label[s] for s in unique_subjects]

    splits = subject_wise_stratified_split(
        unique_subjects, unique_labels,
        train_ratio=SPLIT_RATIOS["train"],
        val_ratio=SPLIT_RATIOS["val"],
        seed=RANDOM_SEED,
    )
    save_splits(splits, splits_path)

    # Collect sample indices for each partition
    split_sets = {k: set(v) for k, v in splits.items()}
    idx_map: Dict[str, list] = {"train": [], "val": [], "test": []}
    for i, subj in enumerate(subjects):
        for part, sid_set in split_sets.items():
            if subj in sid_set:
                idx_map[part].append(i)
                break

    X_raw = feature_df.values.astype(np.float32)
    y = df[label_col].values.astype(np.float32)

    X_train_raw = X_raw[idx_map["train"]]
    X_val_raw   = X_raw[idx_map["val"]]
    X_test_raw  = X_raw[idx_map["test"]]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val   = scaler.transform(X_val_raw).astype(np.float32)
    X_test  = scaler.transform(X_test_raw).astype(np.float32)

    y_train = y[idx_map["train"]]
    y_val   = y[idx_map["val"]]
    y_test  = y[idx_map["test"]]

    os.makedirs(processed_dir, exist_ok=True)
    np.savez(cache_file,
             X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val,
             X_test=X_test, y_test=y_test)

    scaler_path = os.path.join(processed_dir, "speech_scaler.pkl")
    joblib.dump(scaler, scaler_path)

    _print_split_summary("Speech", y_train, y_val, y_test)
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_speech_scaler(processed_dir: str = SPEECH_PROCESSED_DIR) -> StandardScaler:
    path = os.path.join(processed_dir, "speech_scaler.pkl")
    return joblib.load(path)


def get_speech_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    classes = np.array([0, 1])
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    return {0: float(weights[0]), 1: float(weights[1])}


def _find_csv(raw_dir: str) -> str:
    for fname in os.listdir(raw_dir):
        if fname.endswith(".data") or fname.endswith(".csv"):
            return os.path.join(raw_dir, fname)
    raise FileNotFoundError(
        f"No .data or .csv file found in {raw_dir}. "
        "Download parkinsons.data from https://archive.ics.uci.edu/dataset/174/parkinsons "
        "and place it in data/raw/speech/"
    )


def _print_split_summary(name: str, y_tr, y_v, y_te) -> None:
    print(f"\n[{name}] Split summary:")
    for label, arr in [("train", y_tr), ("val", y_v), ("test", y_te)]:
        n = len(arr)
        pd_count = int(arr.sum())
        hc_count = n - pd_count
        print(f"  {label:5s}: {n:4d} samples  |  PD={pd_count}  HC={hc_count}")
