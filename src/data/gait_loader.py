"""
Gait data loader — PhysioNet Gait in Neurodegenerative Disease Database.

Database: 93 PD, 73 HC, and other neurodegenerative subjects.
This loader uses ONLY PD (files prefixed 'pd') and Control (files prefixed 'co') subjects.
Signal: 16-channel vertical ground reaction force at 100 Hz.
Column order: ElapsedTime, L1..L8 (left foot), R1..R8 (right foot)  [18 columns total]

Download from: https://physionet.org/content/gaitpdb/1.0.0/
Place .ts files at: data/raw/gait/
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib
from typing import List, Tuple, Dict, Optional

from src.config import (
    GAIT_SEQUENCE_LEN, GAIT_N_FEATURES, GAIT_TRIM_SECONDS,
    GAIT_SAMPLING_RATE, RANDOM_SEED, SPLIT_RATIOS,
    GAIT_PROCESSED_DIR, GAIT_SPLITS_PATH,
)
from src.data.split_utils import subject_wise_stratified_split, save_splits

_LEFT_COLS  = list(range(1, 9))   # columns 1–8: left foot sensors
_RIGHT_COLS = list(range(9, 17))  # columns 9–16: right foot sensors


def load_gait_data(
    raw_dir: str,
    processed_dir: str = GAIT_PROCESSED_DIR,
    splits_path: str = GAIT_SPLITS_PATH,
    seq_len: int = GAIT_SEQUENCE_LEN,
    trim_seconds: int = GAIT_TRIM_SECONDS,
    sampling_rate: int = GAIT_SAMPLING_RATE,
    force_reprocess: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (X_train, y_train, X_val, y_val, X_test, y_test).
    X arrays have shape (N, seq_len, GAIT_N_FEATURES), dtype float32.
    """
    cache_file = os.path.join(processed_dir, "gait_splits.npz")

    if not force_reprocess and os.path.exists(cache_file):
        data = np.load(cache_file)
        return (
            data["X_train"], data["y_train"],
            data["X_val"], data["y_val"],
            data["X_test"], data["y_test"],
        )

    ts_files = _collect_ts_files(raw_dir)
    if len(ts_files) == 0:
        raise FileNotFoundError(
            f"No .ts files found in {raw_dir}. "
            "Download the PhysioNet Gait-in-PD database and place .ts files at data/raw/gait/"
        )

    samples = []  # (subject_id, feature_seq, label)
    for fpath, sid, label in ts_files:
        seq = _process_file(fpath, trim_seconds, sampling_rate, seq_len)
        if seq is not None:
            samples.append((sid, seq, label))

    if len(samples) == 0:
        raise RuntimeError("No valid gait sequences could be extracted.")

    subject_ids = [s[0] for s in samples]
    labels      = [s[2] for s in samples]

    unique_subjects = list(dict.fromkeys(subject_ids))
    subj_label = {sid: lbl for sid, lbl in zip(subject_ids, labels)}
    unique_labels = [subj_label[s] for s in unique_subjects]

    splits = subject_wise_stratified_split(
        unique_subjects, unique_labels,
        train_ratio=SPLIT_RATIOS["train"],
        val_ratio=SPLIT_RATIOS["val"],
        seed=RANDOM_SEED,
    )
    save_splits(splits, splits_path)

    split_sets = {k: set(v) for k, v in splits.items()}
    part_seqs: Dict[str, List[np.ndarray]] = {"train": [], "val": [], "test": []}
    part_lbls: Dict[str, List[int]]        = {"train": [], "val": [], "test": []}

    for sid, seq, lbl in samples:
        for part, sid_set in split_sets.items():
            if sid in sid_set:
                part_seqs[part].append(seq)
                part_lbls[part].append(lbl)
                break

    X_train_raw = np.stack(part_seqs["train"]).astype(np.float32)
    X_val_raw   = np.stack(part_seqs["val"]).astype(np.float32)
    X_test_raw  = np.stack(part_seqs["test"]).astype(np.float32)

    # Normalize per feature across training subjects (fit on flattened train sequences)
    scaler = StandardScaler()
    n_tr, sl, nf = X_train_raw.shape
    X_train = scaler.fit_transform(X_train_raw.reshape(-1, nf)).reshape(n_tr, sl, nf).astype(np.float32)
    X_val   = scaler.transform(X_val_raw.reshape(-1, nf)).reshape(*X_val_raw.shape).astype(np.float32)
    X_test  = scaler.transform(X_test_raw.reshape(-1, nf)).reshape(*X_test_raw.shape).astype(np.float32)

    y_train = np.array(part_lbls["train"], dtype=np.float32)
    y_val   = np.array(part_lbls["val"],   dtype=np.float32)
    y_test  = np.array(part_lbls["test"],  dtype=np.float32)

    os.makedirs(processed_dir, exist_ok=True)
    np.savez_compressed(cache_file,
                        X_train=X_train, y_train=y_train,
                        X_val=X_val, y_val=y_val,
                        X_test=X_test, y_test=y_test)

    scaler_path = os.path.join(processed_dir, "gait_scaler.pkl")
    joblib.dump(scaler, scaler_path)

    _print_split_summary("Gait", y_train, y_val, y_test)
    return X_train, y_train, X_val, y_val, X_test, y_test


def extract_stride_features(
    grf_signal: np.ndarray,
    sampling_rate: int = GAIT_SAMPLING_RATE,
) -> np.ndarray:
    """
    Input:  (T, 16) GRF signal after trimming.
    Output: (N_strides, 8) matrix.
    8 features per stride: left/right stride_interval, swing, stance, double_support.
    """
    left_force  = grf_signal[:, _LEFT_COLS[0] - 1 : _LEFT_COLS[-1]].sum(axis=1)
    right_force = grf_signal[:, _RIGHT_COLS[0] - 1 : _RIGHT_COLS[-1]].sum(axis=1)

    min_dist = int(0.3 * sampling_rate)   # 300 ms minimum between heel strikes
    left_peaks,  _ = find_peaks(left_force,  distance=min_dist, height=0)
    right_peaks, _ = find_peaks(right_force, distance=min_dist, height=0)

    strides = _compute_stride_features(left_peaks, right_peaks, left_force, right_force, sampling_rate)
    return np.array(strides, dtype=np.float32) if strides else np.empty((0, GAIT_N_FEATURES), dtype=np.float32)


def _compute_stride_features(
    left_peaks, right_peaks, left_force, right_force, sampling_rate
) -> List[List[float]]:
    strides = []
    dt = 1.0 / sampling_rate

    for i in range(len(left_peaks) - 1):
        lp0, lp1 = left_peaks[i], left_peaks[i + 1]
        l_stride = (lp1 - lp0) * dt

        # Find the right heel strike within this left stride window
        rp_in_window = right_peaks[(right_peaks > lp0) & (right_peaks < lp1)]
        if len(rp_in_window) == 0:
            continue
        rp = rp_in_window[0]

        r_stride_idx = np.searchsorted(right_peaks, rp)
        if r_stride_idx + 1 >= len(right_peaks):
            continue
        r_stride = (right_peaks[r_stride_idx + 1] - rp) * dt

        # Swing: time foot is off the ground (force below threshold)
        threshold = 0.05 * left_force.max()
        l_swing = float(np.sum(left_force[lp0:lp1] < threshold)) * dt
        r_swing = float(np.sum(right_force[rp:right_peaks[r_stride_idx + 1]] < threshold)) * dt

        l_stance = l_stride - l_swing
        r_stance = r_stride - r_swing

        # Double support: both feet on ground
        l_dbl = max(0.0, l_stance + r_stance - l_stride)
        r_dbl = l_dbl

        strides.append([l_stride, r_stride, l_swing, r_swing, l_stance, r_stance, l_dbl, r_dbl])

    return strides


def _process_file(
    fpath: str,
    trim_seconds: int,
    sampling_rate: int,
    seq_len: int,
) -> Optional[np.ndarray]:
    try:
        # PhysioNet files may be tab/space delimited; skip header lines starting with '#'
        with open(fpath, "r") as f:
            lines = [l for l in f if not l.startswith("#") and l.strip()]

        rows = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 17:
                rows.append([float(x) for x in parts[:17]])

        if len(rows) < (2 * trim_seconds * sampling_rate + 100):
            return None

        signal = np.array(rows)
        trim_n = trim_seconds * sampling_rate
        signal = signal[trim_n:-trim_n, 1:]   # drop elapsed time column, trim edges

        strides = extract_stride_features(signal, sampling_rate)
        if len(strides) < 10:
            return None

        # Pad or truncate to seq_len
        if len(strides) >= seq_len:
            return strides[:seq_len]
        else:
            pad = np.zeros((seq_len - len(strides), GAIT_N_FEATURES), dtype=np.float32)
            return np.vstack([strides, pad])

    except Exception:
        return None


def _collect_ts_files(raw_dir: str) -> List[Tuple[str, str, int]]:
    """Return list of (filepath, subject_id, label) for PD and HC subjects only."""
    results = []
    for fname in sorted(os.listdir(raw_dir)):
        lower = fname.lower()
        if not (lower.endswith(".ts") or lower.endswith(".txt")):
            continue
        if lower.startswith("pd"):
            label = 1
        elif lower.startswith("co"):
            label = 0
        else:
            continue
        sid = os.path.splitext(fname)[0]
        results.append((os.path.join(raw_dir, fname), sid, label))
    return results


def load_gait_scaler(processed_dir: str = GAIT_PROCESSED_DIR) -> StandardScaler:
    path = os.path.join(processed_dir, "gait_scaler.pkl")
    return joblib.load(path)


def get_gait_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    classes = np.array([0, 1])
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    return {0: float(weights[0]), 1: float(weights[1])}


def _print_split_summary(name: str, y_tr, y_v, y_te) -> None:
    print(f"\n[{name}] Split summary:")
    for label, arr in [("train", y_tr), ("val", y_v), ("test", y_te)]:
        n = len(arr)
        pd_count = int(arr.sum())
        hc_count = n - pd_count
        print(f"  {label:5s}: {n:4d} samples  |  PD={pd_count}  HC={hc_count}")
