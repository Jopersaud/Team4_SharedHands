"""
train_transformer_gpu.py
------------------------
Trains a Transformer model on extracted hand landmark sequences (A-Z ASL).
Requires CUDA GPU. Uses mixed precision (bf16/fp16) for speed.

Input:  artifacts/video_landmarks/{LETTER}/*.npy  (shape: 30 x 63)
Output: artifacts/asl_transformer.pt       (best checkpoint)
        public/asl_transformer/model.onnx  (browser-ready)

Usage:
  python train_transformer_gpu.py
  python train_transformer_gpu.py --data_dir artifacts/video_landmarks --epochs 60
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast

LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}
NUM_CLASSES = 26
SEQ_LEN = 30
FEATURES = 63  # 21 landmarks × 3


# ─── Dataset ──────────────────────────────────────────────────────────────────

class LandmarkDataset(Dataset):
    def __init__(self, data_dir: str):
        self.samples = []  # (path, label_idx)
        root = Path(data_dir)

        for letter in LABELS:
            class_dir = root / letter
            if not class_dir.exists():
                continue
            for npy_file in class_dir.glob("*.npy"):
                self.samples.append((str(npy_file), LABEL_TO_IDX[letter]))

        if not self.samples:
            raise ValueError(f"No .npy files found in {data_dir}. Run extract_video_landmarks.py first.")

        counts = Counter(lbl for _, lbl in self.samples)
        print(f"Dataset: {len(self.samples)} sequences across {len(counts)} classes")
        for i, letter in enumerate(LABELS):
            if i in counts:
                print(f"  {letter}: {counts[i]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        seq = np.load(path)  # (30, 63)

        # Ensure correct shape
        if seq.shape != (SEQ_LEN, FEATURES):
            padded = np.zeros((SEQ_LEN, FEATURES), dtype=np.float32)
            rows = min(seq.shape[0], SEQ_LEN)
            cols = min(seq.shape[1], FEATURES)
            padded[:rows, :cols] = seq[:rows, :cols]
            seq = padded

        return torch.tensor(seq, dtype=torch.float32), label


# ─── Model ────────────────────────────────────────────────────────────────────

class ASLTransformer(nn.Module):
    """
    Transformer encoder for ASL gesture classification from landmark sequences.
    Input:  (batch, seq_len=30, features=63)
    Output: (batch, num_classes=26)
    """
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        seq_len: int = SEQ_LEN,
        input_dim: int = FEATURES,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.input_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x) + self.pos_embedding
        x = self.input_norm(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pool over time
        return self.head(x)


# ─── Training ─────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = correct = total = 0

    for seqs, labels in loader:
        seqs = seqs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(dtype=torch.bfloat16 if device.type == "cuda" else torch.float32):
            logits = model(seqs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * seqs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += seqs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0

    for seqs, labels in loader:
        seqs = seqs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(dtype=torch.bfloat16 if device.type == "cuda" else torch.float32):
            logits = model(seqs)
            loss = criterion(logits, labels)

        total_loss += loss.item() * seqs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += seqs.size(0)

    return total_loss / total, correct / total


def export_onnx(model, out_path: str, device):
    model.eval()
    dummy = torch.randn(1, SEQ_LEN, FEATURES).to(device)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model.cpu(),
        dummy.cpu(),
        str(out_path),
        input_names=["landmarks"],
        output_names=["logits"],
        dynamic_axes={"landmarks": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"ONNX model exported to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="artifacts/video_landmarks")
    parser.add_argument("--out_checkpoint", default="artifacts/asl_transformer.pt")
    parser.add_argument("--out_onnx", default="public/asl_transformer/model.onnx")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    # ── Device ──
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, training on CPU (will be slow)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Data ──
    dataset = LandmarkDataset(args.data_dir)
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )

    print(f"\nTrain: {train_size} | Val: {val_size}")

    # ── Model ──
    model = ASLTransformer().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
    )
    scaler = GradScaler()

    # ── Training loop ──
    best_val_acc = 0.0
    patience_counter = 0
    out_checkpoint = Path(args.out_checkpoint)
    out_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining for up to {args.epochs} epochs (early stopping patience={args.patience})\n")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc*100:.2f}% | "
            f"val loss {val_loss:.4f} acc {val_acc*100:.2f}% | "
            f"lr {lr:.2e}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "labels": LABELS,
                },
                str(out_checkpoint),
            )
            print(f"  ✓ Saved checkpoint (val acc {val_acc*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    print(f"\nBest val accuracy: {best_val_acc*100:.2f}%")

    # ── Export ONNX ──
    print("\nExporting to ONNX...")
    checkpoint = torch.load(str(out_checkpoint), map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    export_onnx(model, args.out_onnx, device)

    print("\nDone! Next step: update the React app to load the ONNX model.")
    print(f"  Checkpoint: {out_checkpoint}")
    print(f"  ONNX model: {args.out_onnx}")


if __name__ == "__main__":
    main()
