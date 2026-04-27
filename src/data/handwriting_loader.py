"""
Handwriting data loader — HandPD / NewHandPD dataset.

Dataset: Spiral drawing images (JPG) in PD and HC folders.
Expected directory layout (either convention works):

  Convention A — class subfolders:
    data/raw/handwriting/PD/subject_001_spiral.jpg
    data/raw/handwriting/HC/subject_002_spiral.jpg

  Convention B — filename prefix:
    data/raw/handwriting/P_spiral_001.jpg   (P = PD)
    data/raw/handwriting/H_spiral_001.jpg   (H = HC)

Download HandPD from: https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/
Place images at: data/raw/handwriting/
"""

import os
import re
import json
import numpy as np
from PIL import Image, ImageOps
from sklearn.utils.class_weight import compute_class_weight
import joblib
from typing import Tuple, Dict, List

from src.config import (
    HANDWRITING_IMG_SIZE, RANDOM_SEED, SPLIT_RATIOS,
    HANDWRITING_PROCESSED_DIR, HANDWRITING_SPLITS_PATH,
)
from src.data.split_utils import subject_wise_stratified_split, save_splits


_IMG_H, _IMG_W, _IMG_C = HANDWRITING_IMG_SIZE


def load_handwriting_data(
    raw_dir: str,
    processed_dir: str = HANDWRITING_PROCESSED_DIR,
    splits_path: str = HANDWRITING_SPLITS_PATH,
    force_reprocess: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (X_train, y_train, X_val, y_val, X_test, y_test).
    X arrays have shape (N, 224, 224, 3), dtype float32, values in [0, 1].
    Augmentation is applied to training split only (at load time — deterministic seed).
    """
    cache_file = os.path.join(processed_dir, "handwriting_splits.npz")

    if not force_reprocess and os.path.exists(cache_file):
        data = np.load(cache_file)
        return (
            data["X_train"], data["y_train"],
            data["X_val"], data["y_val"],
            data["X_test"], data["y_test"],
        )

    samples = _collect_samples(raw_dir)   # list of (subject_id, filepath, label)
    if len(samples) == 0:
        raise FileNotFoundError(
            f"No images found in {raw_dir}. "
            "Download HandPD and place images at data/raw/handwriting/"
        )

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
    partitioned: Dict[str, List] = {"train": [], "val": [], "test": []}
    for sid, fpath, lbl in samples:
        for part, sid_set in split_sets.items():
            if sid in sid_set:
                partitioned[part].append((fpath, lbl))
                break

    X_train, y_train = _load_images(partitioned["train"], augment=True)
    X_val,   y_val   = _load_images(partitioned["val"],   augment=False)
    X_test,  y_test  = _load_images(partitioned["test"],  augment=False)

    os.makedirs(processed_dir, exist_ok=True)
    np.savez_compressed(cache_file,
                        X_train=X_train, y_train=y_train,
                        X_val=X_val, y_val=y_val,
                        X_test=X_test, y_test=y_test)

    _print_split_summary("Handwriting", y_train, y_val, y_test)
    return X_train, y_train, X_val, y_val, X_test, y_test


def _collect_samples(raw_dir: str) -> List[Tuple[str, str, int]]:
    """
    Returns list of (subject_id, filepath, label) tuples.
    Handles both subfolder convention (PD/HC dirs) and filename prefix convention (P_/H_).
    """
    samples = []

    pd_dir = os.path.join(raw_dir, "PD")
    hc_dir = os.path.join(raw_dir, "HC")

    if os.path.isdir(pd_dir) and os.path.isdir(hc_dir):
        for lbl, folder in [(1, pd_dir), (0, hc_dir)]:
            for fname in sorted(os.listdir(folder)):
                if _is_image(fname):
                    sid = _extract_subject_id(fname)
                    fpath = os.path.join(folder, fname)
                    samples.append((sid, fpath, lbl))
        return samples

    # Flat directory with prefix naming
    for fname in sorted(os.listdir(raw_dir)):
        if not _is_image(fname):
            continue
        upper = fname.upper()
        if upper.startswith("P"):
            lbl = 1
        elif upper.startswith("H"):
            lbl = 0
        else:
            continue
        sid = _extract_subject_id(fname)
        fpath = os.path.join(raw_dir, fname)
        samples.append((sid, fpath, lbl))

    return samples


def _extract_subject_id(fname: str) -> str:
    """Extract numeric subject ID from filename, fallback to full stem."""
    stem = os.path.splitext(fname)[0]
    nums = re.findall(r"\d+", stem)
    return nums[0] if nums else stem


def _is_image(fname: str) -> bool:
    return fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))


def _load_images(
    entries: List[Tuple[str, int]], augment: bool
) -> Tuple[np.ndarray, np.ndarray]:
    imgs, lbls = [], []
    rng = np.random.default_rng(RANDOM_SEED)

    for fpath, lbl in entries:
        img = _preprocess_image(fpath)
        imgs.append(img)
        lbls.append(lbl)

        if augment:
            aug = _augment_image(img, rng)
            imgs.append(aug)
            lbls.append(lbl)

    X = np.stack(imgs, axis=0).astype(np.float32)
    y = np.array(lbls, dtype=np.float32)
    return X, y


def _preprocess_image(fpath: str) -> np.ndarray:
    img = Image.open(fpath).convert("RGB")
    img = img.resize((_IMG_W, _IMG_H), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def _augment_image(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Horizontal flip, rotation ±15°, brightness ±0.1."""
    pil_img = Image.fromarray((img * 255).astype(np.uint8))

    if rng.random() > 0.5:
        pil_img = ImageOps.mirror(pil_img)

    angle = rng.uniform(-15, 15)
    pil_img = pil_img.rotate(angle, resample=Image.BILINEAR, fillcolor=(128, 128, 128))

    arr = np.array(pil_img, dtype=np.float32) / 255.0
    brightness_delta = rng.uniform(-0.1, 0.1)
    arr = np.clip(arr + brightness_delta, 0.0, 1.0)
    return arr


def get_handwriting_class_weights(y_train: np.ndarray) -> Dict[int, float]:
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
