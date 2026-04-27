"""
Inference latency benchmark for all six model configurations.
Runs 100+ trials per model, reports mean/median/p95, validates against 1000ms threshold.

Usage:
    python -m src.evaluation.latency_benchmark
"""

import os
import json
import time
import platform
import numpy as np
import pandas as pd
import psutil
import tensorflow as tf

from src.config import (
    RANDOM_SEED,
    SPEECH_FEATURE_DIM, HANDWRITING_IMG_SIZE, GAIT_SEQUENCE_LEN, GAIT_N_FEATURES,
    SPEECH_EMBEDDING_DIM, HANDWRITING_EMBEDDING_DIM, GAIT_EMBEDDING_DIM,
    LATENCY_TRIALS, LATENCY_WARMUP_TRIALS, LATENCY_P95_THRESHOLD_MS,
    SAVED_MODELS_DIR, RESULTS_DIR,
)
from src.models.speech_model import get_speech_embedding_model
from src.models.handwriting_model import get_handwriting_embedding_model
from src.models.gait_model import get_gait_embedding_model
from src.models.late_fusion import LateFusionPredictor
from src.models.hybrid_fusion import load_hybrid_fusion_model, predict_hybrid


def benchmark_single_inference(
    predict_fn,
    sample_input,
    n_trials: int = LATENCY_TRIALS,
    warmup: int = LATENCY_WARMUP_TRIALS,
) -> dict:
    """
    Times predict_fn(sample_input) for n_trials after warmup_trials warm-up calls.
    Returns dict: mean_ms, median_ms, p95_ms, min_ms, max_ms, std_ms.
    """
    for _ in range(warmup):
        predict_fn(sample_input)

    latencies = []
    for _ in range(n_trials):
        start = time.perf_counter()
        predict_fn(sample_input)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies.append(elapsed_ms)

    arr = np.array(latencies)
    return {
        "mean_ms":   round(float(np.mean(arr)),           2),
        "median_ms": round(float(np.median(arr)),         2),
        "p95_ms":    round(float(np.percentile(arr, 95)), 2),
        "min_ms":    round(float(np.min(arr)),            2),
        "max_ms":    round(float(np.max(arr)),            2),
        "std_ms":    round(float(np.std(arr)),            2),
    }


def run_full_benchmark() -> pd.DataFrame:
    np.random.seed(RANDOM_SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    _print_hardware_profile()

    # ── Load models ───────────────────────────────────────────────────────────
    def _load_keras(subdir):
        path = os.path.join(SAVED_MODELS_DIR, subdir, "best_model.keras")
        return tf.keras.models.load_model(path)

    speech_model = _load_keras("speech_baseline")
    hw_model     = _load_keras("handwriting_baseline")
    gait_model   = _load_keras("gait_baseline")
    ef_model     = _load_keras("early_fusion")

    lf_weights_path = os.path.join(SAVED_MODELS_DIR, "late_fusion", "weights.json")
    lf_weights = json.load(open(lf_weights_path)) if os.path.exists(lf_weights_path) else None
    late_predictor = LateFusionPredictor(weights=lf_weights)

    hf_model = load_hybrid_fusion_model(
        os.path.join(SAVED_MODELS_DIR, "hybrid_fusion", "best_model.pt")
    )
    speech_emb_model = get_speech_embedding_model(speech_model)
    hw_emb_model     = get_handwriting_embedding_model(hw_model)
    gait_emb_model   = get_gait_embedding_model(gait_model)

    # ── Single-sample inputs ──────────────────────────────────────────────────
    rng = np.random.default_rng(RANDOM_SEED)
    s_sample  = rng.standard_normal((1, SPEECH_FEATURE_DIM)).astype(np.float32)
    hw_sample = rng.random((1, *HANDWRITING_IMG_SIZE)).astype(np.float32)
    g_sample  = rng.standard_normal((1, GAIT_SEQUENCE_LEN, GAIT_N_FEATURES)).astype(np.float32)

    results = {}

    def _bench(name, fn, inp):
        print(f"\nBenchmarking: {name} ...")
        stats = benchmark_single_inference(fn, inp)
        status = "PASS" if stats["p95_ms"] < LATENCY_P95_THRESHOLD_MS else "FAIL"
        print(f"  mean={stats['mean_ms']}ms  median={stats['median_ms']}ms  "
              f"p95={stats['p95_ms']}ms  [{status}]")
        results[name] = {**stats, "p95_threshold_ms": LATENCY_P95_THRESHOLD_MS, "status": status}

    _bench("Speech Baseline",      lambda x: speech_model.predict(x, verbose=0), s_sample)
    _bench("Handwriting Baseline", lambda x: hw_model.predict(x, verbose=0),     hw_sample)
    _bench("Gait Baseline",        lambda x: gait_model.predict(x, verbose=0),   g_sample)

    # Early Fusion: three inputs simultaneously
    _bench(
        "Early Fusion",
        lambda _: ef_model.predict([s_sample, hw_sample, g_sample], verbose=0),
        None,
    )

    # Late Fusion: run three models + combine (measure full pipeline)
    def lf_predict(_):
        ps = speech_model.predict(s_sample, verbose=0)
        ph = hw_model.predict(hw_sample, verbose=0)
        pg = gait_model.predict(g_sample, verbose=0)
        return late_predictor.predict(ps, ph, pg)
    _bench("Late Fusion", lf_predict, None)

    # Hybrid Fusion: extract embeddings + cross-attention (measure full pipeline)
    def hf_predict(_):
        se = speech_emb_model.predict(s_sample, verbose=0)
        he = hw_emb_model.predict(hw_sample, verbose=0)
        ge = gait_emb_model.predict(g_sample, verbose=0)
        return predict_hybrid(hf_model, se, he, ge)
    _bench("Hybrid Fusion", hf_predict, None)

    df = pd.DataFrame(results).T
    df.index.name = "model"
    save_path = os.path.join(RESULTS_DIR, "latency_report.csv")
    df.to_csv(save_path)
    print(f"\nLatency report saved to {save_path}")
    print(df[["mean_ms", "median_ms", "p95_ms", "status"]].to_string())
    return df


def _print_hardware_profile() -> None:
    print("\n=== Hardware Profile ===")
    print(f"  OS:        {platform.system()} {platform.release()}")
    print(f"  CPU:       {platform.processor()}")
    print(f"  RAM:       {psutil.virtual_memory().total / 1e9:.1f} GB")
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    print(f"  GPUs:      {len(gpus)} detected")
    print(f"  TF:        {tf.__version__}")
    import torch
    print(f"  PyTorch:   {torch.__version__}")
    print(f"  Trials:    {LATENCY_TRIALS} (warmup={LATENCY_WARMUP_TRIALS})")
    print(f"  Threshold: p95 < {LATENCY_P95_THRESHOLD_MS} ms")


if __name__ == "__main__":
    run_full_benchmark()
