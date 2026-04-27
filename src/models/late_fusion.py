"""
Late Fusion: decision-level weighted probability averaging.
No gradient training — weights are optimised by grid search on the validation set.
"""

from itertools import product
from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import f1_score

from src.config import LATE_FUSION_WEIGHT_GRID, LATE_FUSION_DEFAULT_WEIGHTS


class LateFusionPredictor:
    """
    Combines three modality-specific prediction probabilities via weighted average.
    The weights are optimised on the validation set.
    """

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or dict(LATE_FUSION_DEFAULT_WEIGHTS)

    def predict(
        self,
        p_speech: np.ndarray,
        p_handwriting: np.ndarray,
        p_gait: np.ndarray,
        threshold: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (binary_predictions, fused_probabilities).
        Each probability array should have shape (N,) or (N, 1).
        """
        p_s = p_speech.flatten()
        p_h = p_handwriting.flatten()
        p_g = p_gait.flatten()

        w_s = self.weights["speech"]
        w_h = self.weights["handwriting"]
        w_g = self.weights["gait"]
        total = w_s + w_h + w_g

        p_fused = (w_s * p_s + w_h * p_h + w_g * p_g) / total
        preds = (p_fused >= threshold).astype(np.int32)
        return preds, p_fused


def optimise_late_fusion_weights(
    speech_model,
    handwriting_model,
    gait_model,
    val_speech_X: np.ndarray,
    val_hw_X: np.ndarray,
    val_gait_X: np.ndarray,
    val_y: np.ndarray,
    weight_grid: list = None,
) -> Dict[str, float]:
    """
    Grid-search over weight combinations to maximise validation F1-score.
    Returns the best weight dict.
    """
    grid = weight_grid or LATE_FUSION_WEIGHT_GRID

    p_s = speech_model.predict(val_speech_X, verbose=0).flatten()
    p_h = handwriting_model.predict(val_hw_X, verbose=0).flatten()
    p_g = gait_model.predict(val_gait_X, verbose=0).flatten()

    best_f1 = -1.0
    best_weights = dict(LATE_FUSION_DEFAULT_WEIGHTS)

    for w_s, w_h, w_g in product(grid, grid, grid):
        total = w_s + w_h + w_g
        if total == 0:
            continue
        p_fused = (w_s * p_s + w_h * p_h + w_g * p_g) / total
        preds = (p_fused >= 0.5).astype(np.int32)
        f1 = f1_score(val_y.astype(np.int32), preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_weights = {"speech": w_s, "handwriting": w_h, "gait": w_g}

    print(f"\n[Late Fusion] Best val F1={best_f1:.4f} with weights={best_weights}")
    return best_weights
