#!/usr/bin/env python3
"""
train_lstm.py — Train LSTM Autoencoder (Phase 5)

Chức năng:
  - Load dataset từ HDF5 files (train.h5, val.h5, test.h5)
  - Loss: F.mse_loss(recon, target) + beta * kl_divergence(z) với beta=1e-3
  - kl_divergence: 0.5 * sum(mu^2 + sigma^2 - log(sigma^2) - 1)
  - Optimizer: AdamW(lr=1e-3, weight_decay=1e-5)
  - Scheduler: CosineAnnealingWarmRestarts(T_0=10)
  - Gradient clipping: max_norm=1.0
  - EarlyStopping: patience=10, metric=val_loss
  - Log: tensorboard, loguru, console
  - Save best: models/lstm/lstm_ae_best.pt
  - Save last: models/lstm/lstm_ae_last.pt

Usage:
    python scripts/train_lstm.py --data-dir data/lstm_dataset --output-dir models/lstm

Author: AGENTIC-QUANT Team
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

try:
    import h5py
except ImportError:
    h5py = None  # type: ignore[assignment]

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[assignment]

from core.ai_engine.neural.hierarchical_lstm_ae import (
    HierarchicalLSTMAE,
    TICK_FEAT_DIM,
    BAR_FEAT_DIM,
    LATENT_DIM,
    NUM_TF,
)


# =============================================================================
# Constants
# =============================================================================

TICK_SEQ_LEN: int = 128
BAR_SEQ_LEN: int = 30
TF_NAMES: list[str] = ["M1", "M5", "M15", "H1", "H4", "D1"]

# Training hyperparameters
BETA_KL: float = 1e-3          # KL divergence weight
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-5
T_0: int = 10                  # CosineAnnealingWarmRestarts period
T_MULT: int = 2                # T_mult for scheduler
GRAD_MAX_NORM: float = 1.0     # Gradient clipping
PATIENCE: int = 10             # EarlyStopping patience
MIN_DELTA: float = 1e-6        # EarlyStopping min delta
NUM_EPOCHS: int = 200          # Max epochs

BEST_MODEL_NAME: str = "lstm_ae_best.pt"
LAST_MODEL_NAME: str = "lstm_ae_last.pt"


# =============================================================================
# Dataset Loader
# =============================================================================

class LSTMDataset(torch.utils.data.Dataset):
    """PyTorch Dataset cho LSTM Autoencoder training.

    Loads dữ liệu từ HDF5 files và trả về (tick_seq, bar_seqs_mtf) tuples.

    Args:
        h5_path: Đường dẫn đến file .h5
    """

    def __init__(self, h5_path: Path) -> None:
        if h5py is None:
            raise ImportError("h5py required. Run: pip install h5py")

        self.h5_path = h5_path
        with h5py.File(str(h5_path), "r") as f:
            self.n_sequences = f["meta"].attrs["n_sequences"]
            self.tick_shape = f["tick"].shape
            self.bar_shapes: dict[str, tuple] = {}
            for tf in TF_NAMES:
                if tf in f["bars"]:
                    self.bar_shapes[tf] = f["bars"][tf].shape

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
        with h5py.File(str(self.h5_path), "r") as f:
            tick = torch.from_numpy(f["tick"][idx]).float()  # [128, 8]

            bars: list[torch.Tensor] = []
            for tf in TF_NAMES:
                if tf in f["bars"]:
                    bar = torch.from_numpy(f["bars"][tf][idx]).float()  # [30, 12]
                else:
                    bar = torch.zeros(BAR_SEQ_LEN, BAR_FEAT_DIM, dtype=torch.float32)
                bars.append(bar)

        return tick, bars


def create_dataloaders(
    data_dir: Path,
    batch_size: int = 64,
    num_workers: int = 2,
) -> dict[str, torch.utils.data.DataLoader]:
    """Create DataLoaders cho train, val, test.

    Args:
        data_dir: Thư mục chứa HDF5 files
        batch_size: Batch size
        num_workers: Số worker threads cho data loading

    Returns:
        Dict với keys 'train', 'val', 'test'

    Raises:
        FileNotFoundError: Nếu thiếu file HDF5
    """
    splits = ["train", "val", "test"]
    dataloaders: dict[str, torch.utils.data.DataLoader] = {}

    for split_name in splits:
        h5_path = data_dir / f"{split_name}.h5"
        if not h5_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file {h5_path}")

        dataset = LSTMDataset(h5_path)
        shuffle = split_name == "train"
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split_name == "train"),
        )
        dataloaders[split_name] = dataloader
        logger.info(f"  {split_name}: {len(dataset)} batches -> {len(dataloader)}")

    return dataloaders


# =============================================================================
# Loss Functions
# =============================================================================

def kl_divergence(z: torch.Tensor) -> torch.Tensor:
    """Tính KL divergence cho latent vector z.

    Sử dụng VAE-style prior: z ~ N(0, I)
    Thay vì mu/sigma riêng, ta ước lượng mu = mean(z), sigma = std(z)
    và tính KL: 0.5 * sum(mu^2 + sigma^2 - log(sigma^2) - 1)

    Args:
        z: Latent vector [batch, latent_dim]

    Returns:
        KL divergence scalar (mean over batch)
    """
    mu = z.mean(dim=0)       # [latent_dim]
    sigma_sq = z.var(dim=0) + 1e-8  # [latent_dim], thêm epsilon tránh log(0)
    sigma_sq = sigma_sq.clamp(min=1e-8)

    kl = 0.5 * torch.sum(
        mu**2 + sigma_sq - torch.log(sigma_sq) - 1.0
    )  # scalar

    # Normalize by latent_dim để KL không phụ thuộc vào dimension
    kl = kl / z.size(-1)
    return kl


def compute_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    z: torch.Tensor,
    beta: float = BETA_KL,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tính tổng loss = MSE reconstruction + beta * KL divergence.

    Args:
        recon: Reconstructed M1 sequence [batch, seq_len, 8]
        target: Target M1 sequence (tick_seq[:, :, :8]) [batch, seq_len, 8]
        z: Latent vector [batch, 512]
        beta: KL divergence weight

    Returns:
        (total_loss, mse_loss, kl_loss): Tuple các loss values
    """
    mse_loss = F.mse_loss(recon, target)
    kl_loss = kl_divergence(z)
    total_loss = mse_loss + beta * kl_loss
    return total_loss, mse_loss, kl_loss


# =============================================================================
# Early Stopping
# =============================================================================

class EarlyStopping:
    """Early stopping để tránh overfitting.

    Dừng training nếu val_loss không cải thiện sau 'patience' epochs.

    Args:
        patience: Số epochs chờ trước khi dừng
        min_delta: Ngưỡng cải thiện tối thiểu
        verbose: In log khi early stopping kích hoạt
    """

    def __init__(
        self,
        patience: int = PATIENCE,
        min_delta: float = MIN_DELTA,
        verbose: bool = True,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter: int = 0
        self.best_score: float = float("inf")
        self.early_stop: bool = False

    def __call__(self, val_loss: float) -> bool:
        """Check if training should stop.

        Args:
            val_loss: Validation loss hiện tại

        Returns:
            True nếu nên dừng training
        """
        if val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"EarlyStopping counter: {self.counter}/{self.patience} "
                    f"(val_loss={val_loss:.6f}, best={self.best_score:.6f})"
                )
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.warning(f"Early stopping triggered sau {self.patience} epochs!")
        return self.early_stop


# =============================================================================
# Training Utilities
# =============================================================================

def collate_fn(batch: list[tuple[torch.Tensor, list[torch.Tensor]]]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Custom collate function cho DataLoader.

    Args:
        batch: List of (tick, bars_list) tuples

    Returns:
        (tick_batch, bars_batch_list)
            tick_batch: [batch, 128, 8]
            bars_batch_list: list of 6 tensors [batch, 30, 12]
    """
    tick_batch = torch.stack([item[0] for item in batch], dim=0)

    n_tf = len(batch[0][1])
    bars_batch_list: list[torch.Tensor] = []
    for tf_idx in range(n_tf):
        tf_stack = torch.stack([item[1][tf_idx] for item in batch], dim=0)
        bars_batch_list.append(tf_stack)

    return tick_batch, bars_batch_list


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    loss: float,
    path: Path,
) -> None:
    """Save training checkpoint.

    Args:
        model: Model instance
        optimizer: Optimizer instance
        scheduler: Scheduler instance
        epoch: Current epoch
        loss: Current loss value
        path: Đường dẫn lưu checkpoint
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, str(path))
    logger.info(f"Checkpoint saved: {path} (epoch={epoch}, loss={loss:.6f})")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    path: Path | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Load checkpoint.

    Args:
        model: Model instance để load state_dict
        optimizer: Optimizer instance (optional)
        scheduler: Scheduler instance (optional)
        path: Đường dẫn checkpoint
        device: Device để map tensors

    Returns:
        Dict với keys: epoch, loss
    """
    if path is None or not path.exists():
        return {"epoch": 0, "loss": float("inf")}

    if device is None:
        device = torch.device("cpu")

    checkpoint = torch.load(str(path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    logger.info(f"Checkpoint loaded: {path} (epoch={checkpoint.get('epoch', '?')})")
    return {"epoch": checkpoint.get("epoch", 0), "loss": checkpoint.get("loss", float("inf"))}


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta: float = BETA_KL,
    clip_grad: bool = True,
) -> dict[str, float]:
    """Train model trong 1 epoch.

    Args:
        model: LSTM Autoencoder model
        dataloader: Training DataLoader
        optimizer: Optimizer
        device: Device (cpu/cuda)
        beta: KL weight
        clip_grad: Gradient clipping

    Returns:
        Dict với average loss values: total_loss, mse_loss, kl_loss
    """
    model.train()
    total_loss_sum = 0.0
    mse_loss_sum = 0.0
    kl_loss_sum = 0.0
    n_batches = 0

    for tick_seq, bar_seqs_mtf in dataloader:
        tick_seq = tick_seq.to(device, non_blocking=True)
        bar_seqs_mtf = [b.to(device, non_blocking=True) for b in bar_seqs_mtf]

        optimizer.zero_grad()

        # Forward
        z, recon = model(tick_seq, bar_seqs_mtf)

        # Loss
        total_loss, mse_loss, kl_loss = compute_loss(
            recon, tick_seq[..., :8], z, beta=beta
        )

        # Backward
        total_loss.backward()

        # Gradient clipping
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_MAX_NORM)

        optimizer.step()

        total_loss_sum += total_loss.item()
        mse_loss_sum += mse_loss.item()
        kl_loss_sum += kl_loss.item()
        n_batches += 1

    return {
        "total_loss": total_loss_sum / max(n_batches, 1),
        "mse_loss": mse_loss_sum / max(n_batches, 1),
        "kl_loss": kl_loss_sum / max(n_batches, 1),
    }


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    beta: float = BETA_KL,
) -> dict[str, float]:
    """Validate model trong 1 epoch.

    Args:
        model: LSTM Autoencoder model
        dataloader: Validation DataLoader
        device: Device
        beta: KL weight

    Returns:
        Dict với average loss values
    """
    model.eval()
    total_loss_sum = 0.0
    mse_loss_sum = 0.0
    kl_loss_sum = 0.0
    n_batches = 0

    for tick_seq, bar_seqs_mtf in dataloader:
        tick_seq = tick_seq.to(device, non_blocking=True)
        bar_seqs_mtf = [b.to(device, non_blocking=True) for b in bar_seqs_mtf]

        z, recon = model(tick_seq, bar_seqs_mtf)
        total_loss, mse_loss, kl_loss = compute_loss(
            recon, tick_seq[..., :8], z, beta=beta
        )

        total_loss_sum += total_loss.item()
        mse_loss_sum += mse_loss.item()
        kl_loss_sum += kl_loss.item()
        n_batches += 1

    return {
        "total_loss": total_loss_sum / max(n_batches, 1),
        "mse_loss": mse_loss_sum / max(n_batches, 1),
        "kl_loss": kl_loss_sum / max(n_batches, 1),
    }


# =============================================================================
# Main Training Pipeline
# =============================================================================

def run_training(
    data_dir: Path,
    output_dir: Path,
    log_dir: Path | None = None,
    batch_size: int = 64,
    num_epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    beta_kl: float = BETA_KL,
    patience: int = PATIENCE,
    resume: bool = False,
    device_str: str = "auto",
) -> None:
    """Run the full LSTM Autoencoder training pipeline.

    Args:
        data_dir: Thư mục chứa HDF5 dataset
        output_dir: Thư mục lưu model checkpoints
        log_dir: Thư mục lưu TensorBoard logs
        batch_size: Batch size
        num_epochs: Số epochs tối đa
        learning_rate: Learning rate
        weight_decay: Weight decay
        beta_kl: KL divergence weight
        patience: Early stopping patience
        resume: Resume từ checkpoint
        device_str: Device string ('auto', 'cuda', 'cpu')
    """
    # --- Device ---
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info(f"Device: {device}")

    # --- Data ---
    logger.info("Đang load dataset...")
    dataloaders = create_dataloaders(data_dir, batch_size=batch_size, num_workers=2)
    logger.info(f"  Train batches: {len(dataloaders['train'])}")
    logger.info(f"  Val batches:   {len(dataloaders['val'])}")
    logger.info(f"  Test batches:  {len(dataloaders['test'])}")

    # --- Model ---
    model = HierarchicalLSTMAE()
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {n_params:,} parameters")

    # --- Optimizer & Scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=T_0,
        T_mult=T_MULT,
    )

    # --- Early Stopping ---
    early_stopping = EarlyStopping(patience=patience, min_delta=MIN_DELTA, verbose=True)

    # --- Resume ---
    start_epoch = 0
    best_val_loss = float("inf")
    if resume:
        last_path = output_dir / LAST_MODEL_NAME
        if last_path.exists():
            info = load_checkpoint(model, optimizer, scheduler, last_path, device)
            start_epoch = info.get("epoch", 0) + 1
            logger.info(f"Resumed from epoch {start_epoch}")

    # --- TensorBoard ---
    writer: SummaryWriter | None = None
    if SummaryWriter is not None:
        tb_log_dir = log_dir or (output_dir / "tb_logs")
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(tb_log_dir))
        logger.info(f"TensorBoard logs: {tb_log_dir}")
    else:
        logger.warning("TensorBoard not available (pip install tensorboard)")

    # --- Training Loop ---
    logger.info("=" * 60)
    logger.info("BẮT ĐẦU TRAINING")
    logger.info("=" * 60)
    logger.info(
        f"  epochs={num_epochs}, lr={learning_rate}, "
        f"wd={weight_decay}, beta_kl={beta_kl}, "
        f"batch_size={batch_size}, patience={patience}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_epoch: int = -1

    train_start = time.time()
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(model, dataloaders["train"], optimizer, device, beta=beta_kl)
        train_loss = train_metrics["total_loss"]

        # Validate
        val_metrics = validate_epoch(model, dataloaders["val"], device, beta=beta_kl)
        val_loss = val_metrics["total_loss"]

        # Scheduler step
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Log
        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch:3d}/{num_epochs - 1:3d} | "
            f"train={train_loss:.6f} (mse={train_metrics['mse_loss']:.6f}, "
            f"kl={train_metrics['kl_loss']:.6f}) | "
            f"val={val_loss:.6f} | "
            f"lr={current_lr:.2e} | "
            f"time={epoch_time:.1f}s"
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # TensorBoard
        if writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Loss/mse_train", train_metrics["mse_loss"], epoch)
            writer.add_scalar("Loss/kl_train", train_metrics["kl_loss"], epoch)
            writer.add_scalar("LR", current_lr, epoch)

        # Save last checkpoint
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_loss,
            output_dir / LAST_MODEL_NAME,
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                output_dir / BEST_MODEL_NAME,
            )
            logger.info(f"  🏆 New best model! val_loss={val_loss:.6f}")

        # Early stopping
        if early_stopping(val_loss):
            logger.warning(f"Early stopping at epoch {epoch}")
            break

    total_time = time.time() - train_start
    logger.info("=" * 60)
    logger.info("HOÀN THÀNH TRAINING")
    logger.info("=" * 60)
    logger.info(f"  Total time: {total_time:.1f}s ({total_time / 60:.1f}min)")
    logger.info(f"  Best epoch: {best_epoch} (val_loss={best_val_loss:.6f})")
    logger.info(f"  Final train_loss: {train_losses[-1] if train_losses else 'N/A':.6f}")
    logger.info(f"  Final val_loss:   {val_losses[-1] if val_losses else 'N/A':.6f}")

    # --- Final test evaluation ---
    if "test" in dataloaders and len(dataloaders["test"]) > 0:
        logger.info("\nEvaluating on test set...")
        model.load_state_dict(
            torch.load(
                str(output_dir / BEST_MODEL_NAME),
                map_location=device,
                weights_only=False,
            )["model_state_dict"]
        )
        test_metrics = validate_epoch(model, dataloaders["test"], device, beta=beta_kl)
        logger.info(f"  Test total_loss={test_metrics['total_loss']:.6f}")
        logger.info(f"  Test mse_loss={test_metrics['mse_loss']:.6f}")
        logger.info(f"  Test kl_loss={test_metrics['kl_loss']:.6f}")

        if writer is not None:
            writer.add_scalar("Loss/test_total", test_metrics["total_loss"])
            writer.add_scalar("Loss/test_mse", test_metrics["mse_loss"])
            writer.add_scalar("Loss/test_kl", test_metrics["kl_loss"])

    if writer is not None:
        writer.close()

    logger.success(f"Training complete! Models saved to {output_dir.resolve()}")


# =============================================================================
# CLI
# =============================================================================

def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Train LSTM Autoencoder (Phase 5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/lstm_dataset",
        help="Thư mục chứa HDF5 dataset (default: data/lstm_dataset)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/lstm",
        help="Thư mục lưu model checkpoints (default: models/lstm)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Thư mục TensorBoard logs (default: output_dir/tb_logs)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help=f"Max epochs (default: {NUM_EPOCHS})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE})",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=WEIGHT_DECAY,
        help=f"Weight decay (default: {WEIGHT_DECAY})",
    )
    parser.add_argument(
        "--beta-kl",
        type=float,
        default=BETA_KL,
        help=f"KL divergence weight (default: {BETA_KL})",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=PATIENCE,
        help=f"Early stopping patience (default: {PATIENCE})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training từ checkpoint cuối cùng",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device (default: auto)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )
    return parser.parse_args(args)


def main() -> None:
    """Entry point cho train_lstm script."""
    args = parse_args()

    # Cấu hình logging
    logger.remove()
    logger.add(sys.stderr, level=args.log_level, format="<level>{level: <8}</level> | {message}")

    base_dir = Path(__file__).resolve().parent.parent

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = base_dir / data_dir

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir

    log_dir: Path | None = None
    if args.log_dir:
        log_dir = Path(args.log_dir)
        if not log_dir.is_absolute():
            log_dir = base_dir / log_dir

    try:
        run_training(
            data_dir=data_dir,
            output_dir=output_dir,
            log_dir=log_dir,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            beta_kl=args.beta_kl,
            patience=args.patience,
            resume=args.resume,
            device_str=args.device,
        )
    except Exception as e:
        logger.error(f"Training thất bại: {e}")
        raise


if __name__ == "__main__":
    main()
