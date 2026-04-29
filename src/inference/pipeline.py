"""
Prototype inference pipeline for early Parkinson's Disease detection.

Accepts available speech, handwriting, and/or gait inputs,
executes preprocessing and multimodal prediction,
outputs a PD risk classification, and logs per-run latency.

Usage:
    from src.inference.pipeline import PDInferencePipeline

    pipeline = PDInferencePipeline()
    result = pipeline.predict(
        speech_features=np.array([...22 values...]),
        handwriting_image_path="patient_spiral.jpg",
        gait_file_path="patient_walk.ts",
        fusion_strategy="hybrid",
    )
    print(result)
"""

import os
import json
import time
import platform
import datetime
import numpy as np
import tensorflow as tf
from typing import Optional, Dict, Any

from src.config import (
    SPEECH_FEATURE_DIM, HANDWRITING_IMG_SIZE, GAIT_SEQUENCE_LEN, GAIT_N_FEATURES,
    SAVED_MODELS_DIR, SPEECH_PROCESSED_DIR, GAIT_PROCESSED_DIR,
    LATENCY_P95_THRESHOLD_MS,
)
from src.models.speech_model import get_speech_embedding_model
from src.models.handwriting_model import get_handwriting_embedding_model
from src.models.gait_model import get_gait_embedding_model
from src import safe_load_model
from src.models.late_fusion import LateFusionPredictor
from src.models.hybrid_fusion import load_hybrid_fusion_model, predict_hybrid


class PDInferencePipeline:
    """
    End-to-end prototype inference pipeline.

    Supports six inference strategies:
        'speech'      — speech unimodal model only
        'handwriting' — handwriting unimodal model only
        'gait'        — gait unimodal model only
        'early'       — Early Fusion (all three modalities required)
        'late'        — Late Fusion  (all three modalities required)
        'hybrid'      — Hybrid Fusion (all three modalities required)
    """

    def __init__(self, models_dir: str = SAVED_MODELS_DIR):
        self._models_dir = models_dir
        self._loaded: Dict[str, Any] = {}
        self._speech_scaler = None
        self._gait_scaler = None
        self._late_weights = None
        self._load_scalers()

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict(
        self,
        speech_features: Optional[np.ndarray] = None,
        handwriting_image_path: Optional[str] = None,
        gait_file_path: Optional[str] = None,
        fusion_strategy: str = "late",
        log_path: Optional[str] = None,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Run inference. Returns a result dict with prediction, probability, label, latency_ms.

        For unimodal strategies ('speech', 'handwriting', 'gait'), only the
        matching input is required.
        For fusion strategies ('early', 'late', 'hybrid'), all three inputs are required.
        """
        strategy = fusion_strategy.lower()

        start = time.perf_counter()

        if strategy == "speech":
            prob = self._predict_speech(speech_features)
        elif strategy == "handwriting":
            prob = self._predict_handwriting(handwriting_image_path)
        elif strategy == "gait":
            prob = self._predict_gait(gait_file_path)
        elif strategy == "early":
            prob = self._predict_early(speech_features, handwriting_image_path, gait_file_path)
        elif strategy == "late":
            prob = self._predict_late(speech_features, handwriting_image_path, gait_file_path)
        elif strategy == "hybrid":
            prob = self._predict_hybrid(speech_features, handwriting_image_path, gait_file_path)
        else:
            raise ValueError(f"Unknown fusion_strategy: '{fusion_strategy}'. "
                             "Choose from: speech, handwriting, gait, early, late, hybrid")

        latency_ms = (time.perf_counter() - start) * 1000.0
        prediction = int(prob >= threshold)
        label = "Parkinson's Disease" if prediction == 1 else "Healthy Control"

        result = {
            "prediction":    prediction,
            "probability":   round(float(prob), 4),
            "label":         label,
            "model_used":    strategy,
            "latency_ms":    round(latency_ms, 2),
            "threshold":     threshold,
            "within_target": latency_ms < LATENCY_P95_THRESHOLD_MS,
            "timestamp":     datetime.datetime.now().isoformat(),
        }

        if log_path:
            self.log_result(result, log_path)

        return result

    def log_result(self, result: Dict, log_path: str) -> None:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(result) + "\n")

    # ── Preprocessing ──────────────────────────────────────────────────────────

    def preprocess_speech(self, raw_features: np.ndarray) -> np.ndarray:
        """Normalize a (22,) or (1, 22) speech feature vector."""
        arr = np.array(raw_features, dtype=np.float32).reshape(1, -1)
        if self._speech_scaler is not None:
            arr = self._speech_scaler.transform(arr)
        return arr.astype(np.float32)

    def preprocess_handwriting(self, image_path: str) -> np.ndarray:
        """Load, resize, and normalise a handwriting image to (1, 224, 224, 3)."""
        from PIL import Image
        h, w, c = HANDWRITING_IMG_SIZE
        img = Image.open(image_path).convert("RGB")
        img = img.resize((w, h))
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr[np.newaxis]

    def preprocess_gait(self, gait_file_path: str) -> np.ndarray:
        """Load a PhysioNet .ts file and extract stride features → (1, 300, 8)."""
        from src.data.gait_loader import _process_file
        from src.config import GAIT_TRIM_SECONDS, GAIT_SAMPLING_RATE
        seq = _process_file(gait_file_path, GAIT_TRIM_SECONDS, GAIT_SAMPLING_RATE, GAIT_SEQUENCE_LEN)
        if seq is None:
            raise ValueError(f"Could not extract valid gait features from {gait_file_path}")
        arr = seq.astype(np.float32)[np.newaxis]
        if self._gait_scaler is not None:
            arr_flat = arr.reshape(1, -1)
            arr = self._gait_scaler.transform(arr_flat).reshape(1, GAIT_SEQUENCE_LEN, GAIT_N_FEATURES)
        return arr.astype(np.float32)

    # ── Unimodal inference ─────────────────────────────────────────────────────

    def _predict_speech(self, features) -> float:
        x = self.preprocess_speech(features)
        m = self._get_model("speech_baseline")
        return float(m.predict(x, verbose=0).flatten()[0])

    def _predict_handwriting(self, image_path) -> float:
        x = self.preprocess_handwriting(image_path)
        m = self._get_model("handwriting_baseline")
        return float(m.predict(x, verbose=0).flatten()[0])

    def _predict_gait(self, gait_path) -> float:
        x = self.preprocess_gait(gait_path)
        m = self._get_model("gait_baseline")
        return float(m.predict(x, verbose=0).flatten()[0])

    # ── Fusion inference ───────────────────────────────────────────────────────

    def _predict_early(self, speech_f, hw_path, gait_path) -> float:
        xs = self.preprocess_speech(speech_f)
        xh = self.preprocess_handwriting(hw_path)
        xg = self.preprocess_gait(gait_path)
        m  = self._get_model("early_fusion")
        return float(m.predict([xs, xh, xg], verbose=0).flatten()[0])

    def _predict_late(self, speech_f, hw_path, gait_path) -> float:
        ps = self._predict_speech(speech_f)
        ph = self._predict_handwriting(hw_path)
        pg = self._predict_gait(gait_path)
        predictor = self._get_late_predictor()
        _, p_fused = predictor.predict(
            np.array([[ps]]), np.array([[ph]]), np.array([[pg]])
        )
        return float(p_fused[0])

    def _predict_hybrid(self, speech_f, hw_path, gait_path) -> float:
        xs = self.preprocess_speech(speech_f)
        xh = self.preprocess_handwriting(hw_path)
        xg = self.preprocess_gait(gait_path)

        sm = self._get_model("speech_baseline")
        hm = self._get_model("handwriting_baseline")
        gm = self._get_model("gait_baseline")

        se = get_speech_embedding_model(sm).predict(xs, verbose=0)
        he = get_handwriting_embedding_model(hm).predict(xh, verbose=0)
        ge = get_gait_embedding_model(gm).predict(xg, verbose=0)

        hybrid_m = self._get_hybrid_model()
        _, probs = predict_hybrid(hybrid_m, se, he, ge)
        return float(probs[0])

    # ── Model loading (lazy, cached) ───────────────────────────────────────────

    def _get_model(self, name: str):
        if name not in self._loaded:
            path = os.path.join(self._models_dir, name, "best_model.keras")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Trained model not found at {path}. "
                    "Run the corresponding training script first."
                )
            self._loaded[name] = safe_load_model(path)
        return self._loaded[name]

    def _get_late_predictor(self) -> LateFusionPredictor:
        if "late_fusion" not in self._loaded:
            weights_path = os.path.join(self._models_dir, "late_fusion", "weights.json")
            weights = None
            if os.path.exists(weights_path):
                with open(weights_path) as f:
                    weights = json.load(f)
            self._loaded["late_fusion"] = LateFusionPredictor(weights=weights)
        return self._loaded["late_fusion"]

    def _get_hybrid_model(self):
        if "hybrid_fusion" not in self._loaded:
            path = os.path.join(self._models_dir, "hybrid_fusion", "best_model.pt")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Hybrid fusion model not found at {path}")
            self._loaded["hybrid_fusion"] = load_hybrid_fusion_model(path)
        return self._loaded["hybrid_fusion"]

    def _load_scalers(self) -> None:
        import joblib
        speech_scaler_path = os.path.join(SPEECH_PROCESSED_DIR, "speech_scaler.pkl")
        if os.path.exists(speech_scaler_path):
            self._speech_scaler = joblib.load(speech_scaler_path)

        gait_scaler_path = os.path.join(GAIT_PROCESSED_DIR, "gait_scaler.pkl")
        if os.path.exists(gait_scaler_path):
            self._gait_scaler = joblib.load(gait_scaler_path)
