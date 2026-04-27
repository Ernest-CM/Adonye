import os

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Data split ratios ────────────────────────────────────────────────────────
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# ── Speech configuration ─────────────────────────────────────────────────────
SPEECH_FEATURE_DIM = 22
SPEECH_EMBEDDING_DIM = 32
SPEECH_FEATURES = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR",
    "RPDE", "DFA", "spread1", "spread2", "D2", "PPE",
]

# ── Handwriting configuration ─────────────────────────────────────────────────
HANDWRITING_IMG_SIZE = (224, 224, 3)
HANDWRITING_EMBEDDING_DIM = 256
DENSENET_FREEZE_LAYERS = 430        # freeze all DenseNet121 base layers in Phase 1
DENSENET_UNFREEZE_TOP = 50          # unfreeze top N layers in Phase 2
HANDWRITING_PHASE1_EPOCHS = 10
HANDWRITING_PHASE1_LR = 1e-3
HANDWRITING_PHASE2_LR = 1e-5

# ── Gait configuration ────────────────────────────────────────────────────────
GAIT_SEQUENCE_LEN = 300             # strides after trim/pad/truncate
GAIT_N_FEATURES = 8                 # stride-level features per timestep
GAIT_TRIM_SECONDS = 20              # drop first and last N seconds from each walk
GAIT_SAMPLING_RATE = 100            # Hz
GAIT_EMBEDDING_DIM = 32

# ── Training hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
LEARNING_RATE = 1e-3
DROPOUT_RATE_SPEECH = 0.4
DROPOUT_RATE_HANDWRITING = 0.5
DROPOUT_RATE_GAIT = 0.3
DROPOUT_RATE_FUSION = 0.4
L2_LAMBDA = 1e-4

# ── Hybrid Fusion (PyTorch) ───────────────────────────────────────────────────
HYBRID_D_MODEL = 128
HYBRID_NUM_HEADS = 4

# ── Late Fusion weight search ────────────────────────────────────────────────
LATE_FUSION_WEIGHT_GRID = [0.0, 0.5, 1.0, 1.5, 2.0]
LATE_FUSION_DEFAULT_WEIGHTS = {"speech": 1.0, "handwriting": 1.0, "gait": 1.0}

# ── Latency benchmark ────────────────────────────────────────────────────────
LATENCY_TRIALS = 100
LATENCY_WARMUP_TRIALS = 10
LATENCY_P95_THRESHOLD_MS = 1000.0

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW_DIR = os.path.join(_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(_ROOT, "data", "processed")
SPLITS_DIR = os.path.join(_ROOT, "data", "splits")

SPEECH_RAW_DIR = os.path.join(DATA_RAW_DIR, "speech")
SPEECH_PROCESSED_DIR = os.path.join(DATA_PROCESSED_DIR, "speech")
SPEECH_SPLITS_PATH = os.path.join(SPLITS_DIR, "speech_splits.json")

HANDWRITING_RAW_DIR = os.path.join(DATA_RAW_DIR, "handwriting")
HANDWRITING_PROCESSED_DIR = os.path.join(DATA_PROCESSED_DIR, "handwriting")
HANDWRITING_SPLITS_PATH = os.path.join(SPLITS_DIR, "handwriting_splits.json")

GAIT_RAW_DIR = os.path.join(DATA_RAW_DIR, "gait")
GAIT_PROCESSED_DIR = os.path.join(DATA_PROCESSED_DIR, "gait")
GAIT_SPLITS_PATH = os.path.join(SPLITS_DIR, "gait_splits.json")

SAVED_MODELS_DIR = os.path.join(_ROOT, "saved_models")
RESULTS_DIR = os.path.join(_ROOT, "results")
LOGS_DIR = os.path.join(_ROOT, "logs")
