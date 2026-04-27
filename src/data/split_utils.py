import json
import numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Dict, List, Tuple


def subject_wise_stratified_split(
    subject_ids: List[str],
    labels: List[int],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Split at subject level so no subject's recordings appear in more than one partition.
    Returns dict with keys 'train', 'val', 'test' each containing a list of subject IDs.
    """
    unique_subjects = list({sid: lbl for sid, lbl in zip(subject_ids, labels)}.keys())
    subject_label_map = {}
    for sid, lbl in zip(subject_ids, labels):
        subject_label_map[sid] = lbl

    unique_subjects = sorted(unique_subjects)
    subject_labels = [subject_label_map[s] for s in unique_subjects]

    test_ratio = 1.0 - train_ratio - val_ratio

    # First split: isolate test set
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    train_val_idx, test_idx = next(sss1.split(unique_subjects, subject_labels))

    train_val_subjects = [unique_subjects[i] for i in train_val_idx]
    train_val_labels = [subject_labels[i] for i in train_val_idx]
    test_subjects = [unique_subjects[i] for i in test_idx]

    # Second split: isolate val from remaining train+val
    relative_val_ratio = val_ratio / (train_ratio + val_ratio)
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=relative_val_ratio, random_state=seed
    )
    train_idx, val_idx = next(sss2.split(train_val_subjects, train_val_labels))

    train_subjects = [train_val_subjects[i] for i in train_idx]
    val_subjects = [train_val_subjects[i] for i in val_idx]

    splits = {"train": train_subjects, "val": val_subjects, "test": test_subjects}
    leakage_check(splits)
    return splits


def leakage_check(splits: Dict[str, List[str]]) -> bool:
    """
    Asserts no subject ID appears in more than one partition.
    Raises ValueError if leakage is detected.
    Returns True if clean.
    """
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    train_val = train_set & val_set
    train_test = train_set & test_set
    val_test = val_set & test_set

    if train_val or train_test or val_test:
        raise ValueError(
            f"Data leakage detected! "
            f"train∩val={train_val}, train∩test={train_test}, val∩test={val_test}"
        )
    return True


def save_splits(splits: Dict[str, List[str]], path: str) -> None:
    with open(path, "w") as f:
        json.dump(splits, f, indent=2)


def load_splits(path: str) -> Dict[str, List[str]]:
    with open(path, "r") as f:
        return json.load(f)


def collect_samples_for_split(
    all_samples: List[Tuple],
    splits: Dict[str, List[str]],
    subject_col: int = 0,
) -> Dict[str, List[Tuple]]:
    """
    Given a list of (subject_id, *other_fields) tuples and a splits dict,
    returns {'train': [...], 'val': [...], 'test': [...]}.
    subject_col is the index of the subject_id in each tuple.
    """
    split_sets = {k: set(v) for k, v in splits.items()}
    result = defaultdict(list)
    for sample in all_samples:
        sid = sample[subject_col]
        for partition, sid_set in split_sets.items():
            if sid in sid_set:
                result[partition].append(sample)
                break
    return dict(result)
