import sys
sys.path.insert(0, ".")

print("=== Speech ===")
from src.data.speech_loader import load_speech_data
X_tr, y_tr, X_v, y_v, X_te, y_te = load_speech_data("data/raw/speech", force_reprocess=True)
print(f"  train={X_tr.shape}  val={X_v.shape}  test={X_te.shape}")
print(f"  PD/HC — train: {int(y_tr.sum())}/{len(y_tr)-int(y_tr.sum())}  val: {int(y_v.sum())}/{len(y_v)-int(y_v.sum())}  test: {int(y_te.sum())}/{len(y_te)-int(y_te.sum())}")

print("\n=== Handwriting ===")
from src.data.handwriting_loader import load_handwriting_data
X_tr, y_tr, X_v, y_v, X_te, y_te = load_handwriting_data("data/raw/handwriting", force_reprocess=True)
print(f"  train={X_tr.shape}  val={X_v.shape}  test={X_te.shape}")
print(f"  PD/HC — train: {int(y_tr.sum())}/{len(y_tr)-int(y_tr.sum())}  val: {int(y_v.sum())}/{len(y_v)-int(y_v.sum())}  test: {int(y_te.sum())}/{len(y_te)-int(y_te.sum())}")

print("\n=== Gait ===")
from src.data.gait_loader import load_gait_data
X_tr, y_tr, X_v, y_v, X_te, y_te = load_gait_data("data/raw/gait", force_reprocess=True)
print(f"  train={X_tr.shape}  val={X_v.shape}  test={X_te.shape}")
print(f"  PD/HC — train: {int(y_tr.sum())}/{len(y_tr)-int(y_tr.sum())}  val: {int(y_v.sum())}/{len(y_v)-int(y_v.sum())}  test: {int(y_te.sum())}/{len(y_te)-int(y_te.sum())}")

print("\n=== ALL LOADERS OK ===")
