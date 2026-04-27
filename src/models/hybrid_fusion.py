"""
Hybrid Fusion: pairwise cross-attention between all three modality embeddings.
Implemented in PyTorch. Embeddings are pre-computed from trained Keras models.

Architecture:
  6 CrossModalAttentionBlocks (all pairwise directions) → concatenate 6×128 = 768-dim
  → Linear(768→256) → ReLU → Dropout → LayerNorm → Linear(256→64) → ReLU → Linear(64→1) → Sigmoid
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional

from src.config import (
    RANDOM_SEED,
    SPEECH_EMBEDDING_DIM, HANDWRITING_EMBEDDING_DIM, GAIT_EMBEDDING_DIM,
    HYBRID_D_MODEL, HYBRID_NUM_HEADS,
    BATCH_SIZE, EPOCHS, EARLY_STOPPING_PATIENCE, DROPOUT_RATE_FUSION,
)


class CrossModalAttentionBlock(nn.Module):
    """
    Attends from query_modality toward context_modality.
    Both are projected to d_model before attention.
    """

    def __init__(self, d_query: int, d_context: int, d_model: int, num_heads: int):
        super().__init__()
        self.q_proj = nn.Linear(d_query, d_model)
        self.k_proj = nn.Linear(d_context, d_model)
        self.v_proj = nn.Linear(d_context, d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query_emb: torch.Tensor, context_emb: torch.Tensor) -> torch.Tensor:
        # Shape: (B, d_model) → (B, 1, d_model) for MultiheadAttention
        Q = self.q_proj(query_emb).unsqueeze(1)
        K = self.k_proj(context_emb).unsqueeze(1)
        V = self.v_proj(context_emb).unsqueeze(1)
        attended, _ = self.attn(Q, K, V)
        return self.norm(attended.squeeze(1))


class HybridFusionModel(nn.Module):
    """
    Six pairwise cross-attention blocks across speech, handwriting, and gait embeddings.
    """

    def __init__(
        self,
        speech_dim: int = SPEECH_EMBEDDING_DIM,
        hw_dim: int = HANDWRITING_EMBEDDING_DIM,
        gait_dim: int = GAIT_EMBEDDING_DIM,
        d_model: int = HYBRID_D_MODEL,
        num_heads: int = HYBRID_NUM_HEADS,
        dropout: float = DROPOUT_RATE_FUSION,
    ):
        super().__init__()

        # 6 directional attention blocks: A→B, A→C, B→A, B→C, C→A, C→B
        self.s_from_hw   = CrossModalAttentionBlock(speech_dim, hw_dim,   d_model, num_heads)
        self.s_from_gait = CrossModalAttentionBlock(speech_dim, gait_dim, d_model, num_heads)
        self.hw_from_s   = CrossModalAttentionBlock(hw_dim, speech_dim,   d_model, num_heads)
        self.hw_from_g   = CrossModalAttentionBlock(hw_dim, gait_dim,     d_model, num_heads)
        self.g_from_s    = CrossModalAttentionBlock(gait_dim, speech_dim, d_model, num_heads)
        self.g_from_hw   = CrossModalAttentionBlock(gait_dim, hw_dim,     d_model, num_heads)

        fused_dim = 6 * d_model  # 768

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(256),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.75),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        speech_emb: torch.Tensor,
        hw_emb: torch.Tensor,
        gait_emb: torch.Tensor,
    ) -> torch.Tensor:
        s_hw   = self.s_from_hw(speech_emb, hw_emb)
        s_g    = self.s_from_gait(speech_emb, gait_emb)
        hw_s   = self.hw_from_s(hw_emb, speech_emb)
        hw_g   = self.hw_from_g(hw_emb, gait_emb)
        g_s    = self.g_from_s(gait_emb, speech_emb)
        g_hw   = self.g_from_hw(gait_emb, hw_emb)

        fused = torch.cat([s_hw, s_g, hw_s, hw_g, g_s, g_hw], dim=1)
        return self.classifier(fused)


class MultimodalEmbeddingDataset(Dataset):
    def __init__(
        self,
        speech_embs: np.ndarray,
        hw_embs: np.ndarray,
        gait_embs: np.ndarray,
        labels: np.ndarray,
    ):
        self.s = torch.from_numpy(speech_embs).float()
        self.h = torch.from_numpy(hw_embs).float()
        self.g = torch.from_numpy(gait_embs).float()
        self.y = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.s[idx], self.h[idx], self.g[idx], self.y[idx]


def train_hybrid_fusion(
    speech_emb_model,
    hw_emb_model,
    gait_emb_model,
    train_data: Tuple,
    val_data: Tuple,
    save_dir: str,
    epochs: int = EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    smoke_test: bool = False,
) -> HybridFusionModel:
    """
    train_data / val_data: (speech_X, hw_X, gait_X, y)
    Extracts TF embeddings, then trains PyTorch HybridFusionModel.
    """
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Extract embeddings from Keras models (TF→numpy→PyTorch bridge)
    def get_embs(speech_X, hw_X, gait_X):
        s = speech_emb_model.predict(speech_X, verbose=0)
        h = hw_emb_model.predict(hw_X, verbose=0)
        g = gait_emb_model.predict(gait_X, verbose=0)
        return s, h, g

    print("\n[Hybrid Fusion] Extracting embeddings...")
    train_s, train_h, train_g = get_embs(*train_data[:3])
    val_s,   val_h,   val_g   = get_embs(*val_data[:3])

    # Cache embeddings for fast re-use
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "train_embs.npy"),
            np.concatenate([train_s, train_h, train_g], axis=1))
    np.save(os.path.join(save_dir, "val_embs.npy"),
            np.concatenate([val_s, val_h, val_g], axis=1))

    train_dataset = MultimodalEmbeddingDataset(train_s, train_h, train_g, train_data[3])
    val_dataset   = MultimodalEmbeddingDataset(val_s,   val_h,   val_g,   val_data[3])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Hybrid Fusion] Training on {device}")

    model = HybridFusionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=7
    )

    best_val_f1 = -1.0
    epochs_no_improve = 0
    run_epochs = 3 if smoke_test else epochs
    checkpoint_path = os.path.join(save_dir, "best_model.pt")

    for epoch in range(run_epochs):
        model.train()
        train_loss = 0.0
        for s_b, h_b, g_b, y_b in train_loader:
            s_b, h_b, g_b, y_b = s_b.to(device), h_b.to(device), g_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds = model(s_b, h_b, g_b).squeeze(1)
            loss = criterion(preds, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for s_b, h_b, g_b, y_b in val_loader:
                s_b, h_b, g_b = s_b.to(device), h_b.to(device), g_b.to(device)
                out = model(s_b, h_b, g_b).squeeze(1).cpu().numpy()
                val_preds.extend(out)
                val_labels.extend(y_b.numpy())

        val_preds_bin = (np.array(val_preds) >= 0.5).astype(int)
        from sklearn.metrics import f1_score
        val_f1 = f1_score(val_labels, val_preds_bin, zero_division=0)
        scheduler.step(val_f1)

        print(f"Epoch {epoch+1}/{run_epochs} | loss={train_loss/len(train_loader):.4f} | val_F1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[Hybrid Fusion] Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"\n[Hybrid Fusion] Best val F1={best_val_f1:.4f}. Model saved to {checkpoint_path}")
    return model


def load_hybrid_fusion_model(checkpoint_path: str) -> HybridFusionModel:
    model = HybridFusionModel()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


def predict_hybrid(
    model: HybridFusionModel,
    speech_embs: np.ndarray,
    hw_embs: np.ndarray,
    gait_embs: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (binary_predictions, probabilities)."""
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        s = torch.from_numpy(speech_embs).float().to(device)
        h = torch.from_numpy(hw_embs).float().to(device)
        g = torch.from_numpy(gait_embs).float().to(device)
        probs = model(s, h, g).squeeze(1).cpu().numpy()
    preds = (probs >= threshold).astype(np.int32)
    return preds, probs
