#!/usr/bin/env python3
"""
build_lstm_dataset.py — Xây dựng dataset cho LSTM Autoencoder (Phase 5)

Chức năng:
  - Đọc dữ liệu tick từ parquet files trong data/parquet/
  - Temporal split: train (2022-2023), val (2024-Q1), test (2024-Q2)
  - Build tick_seq[128x8]: [open, high, low, close, volume, spread, bid, ask]
  - Build bar_seqs[6 TF x 30 x 12]: M1(30), M5(30), M15(30), H1(30), H4(30), D1(30)
    Với 12 bar features: [open, high, low, close, volume, wick_upper, wick_lower,
    body, range, vwap, spread, tick_count]
  - Normalize: RobustScaler per feature (fit on train, transform all)
  - Save: HDF5 with h5py (train.h5, val.h5, test.h5)
  - Save scaler: models/lstm/feature_scaler.pkl (joblib)

Usage:
    python scripts/build_lstm_dataset.py --data-dir data/parquet --output-dir data/lstm_dataset

Author: AGENTIC-QUANT Team
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

try:
    import h5py
except ImportError:
    h5py = None  # type: ignore[assignment]

try:
    from sklearn.preprocessing import RobustScaler
    import joblib
except ImportError:
    RobustScaler = None  # type: ignore[assignment]
    joblib = None  # type: ignore[assignment]


# =============================================================================
# Constants
# =============================================================================

TICK_SEQ_LEN: int = 128       # Sequence length for tick data
BAR_SEQ_LEN: int = 30         # Sequence length for bar data (each TF)
TICK_FEAT_DIM: int = 8        # Số feature của tick: [open, high, low, close, volume, spread, bid, ask]
BAR_FEAT_DIM: int = 12        # Số feature của bar: [open, high, low, close, volume,
                               #   wick_upper, wick_lower, body, range, vwap, spread, tick_count]
NUM_TF: int = 6               # Số timeframe: M1, M5, M15, H1, H4, D1
TF_NAMES: list[str] = ["M1", "M5", "M15", "H1", "H4", "D1"]

# Timeframe lookback windows (number of bars cho mỗi TF sequence)
TF_BAR_COUNTS: dict[str, int] = {
    "M1": 30,
    "M5": 30,
    "M15": 30,
    "H1": 30,
    "H4": 30,
    "D1": 30,
}

# Temporal split boundaries (YYYY-MM-DD)
TRAIN_END: str = "2023-12-31"
VAL_END: str = "2024-03-31"
# Test: 2024-04-01 -> 2024-06-30

# Output filenames
TRAIN_H5: str = "train.h5"
VAL_H5: str = "val.h5"
TEST_H5: str = "test.h5"
SCALER_PKL: str = "feature_scaler.pkl"


# =============================================================================
# Data Loading
# =============================================================================

def discover_parquet_files(data_dir: Path) -> list[Path]:
    """Tìm tất cả file .parquet trong thư mục data_dir.

    Args:
        data_dir: Đường dẫn thư mục chứa parquet files.

    Returns:
        List các Path đến file .parquet.

    Raises:
        FileNotFoundError: Nếu không tìm thấy file nào.
    """
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"Không tìm thấy file .parquet nào trong {data_dir.resolve()}"
        )
    logger.info(f"Tìm thấy {len(parquet_files)} file parquet")
    return parquet_files


def load_and_concatenate_parquet(parquet_files: list[Path]) -> pd.DataFrame:
    """Đọc và gộp tất cả parquet files thành một DataFrame.

    Args:
        parquet_files: List các file parquet.

    Returns:
        DataFrame chứa toàn bộ dữ liệu, sắp xếp theo timestamp.
    """
    chunks: list[pd.DataFrame] = []
    for fpath in parquet_files:
        try:
            df = pd.read_parquet(fpath)
            if df.empty:
                logger.warning(f"File rỗng: {fpath}")
                continue
            chunks.append(df)
            logger.debug(f"Đã đọc {fpath.name}: {df.shape}")
        except Exception as e:
            logger.warning(f"Lỗi đọc {fpath}: {e}")

    if not chunks:
        raise ValueError("Không thể đọc được dữ liệu từ parquet files.")

    df = pd.concat(chunks, ignore_index=True)

    # Đảm bảo có datetime column để temporal split
    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    elif "datetime" in df.columns:
        df.rename(columns={"datetime": "timestamp"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        # Cố gắng tìm column datetime
        for col in df.columns:
            if "time" in col.lower() and "date" in col.lower():
                df.rename(columns={col: "timestamp"}, inplace=True)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                break
        else:
            raise ValueError(
                "Không tìm thấy timestamp/datetime column trong dữ liệu. "
                f"Các column hiện tại: {list(df.columns)}"
            )

    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info(
        f"Đã load {len(df)} rows từ {len(parquet_files)} files, "
        f"timestamp range: {df['timestamp'].min()} -> {df['timestamp'].max()}"
    )
    return df


# =============================================================================
# Feature Engineering
# =============================================================================

def build_tick_features(df: pd.DataFrame) -> np.ndarray:
    """Build tick_seq[seq_len x 8] từ tick data.

    8 features: [open, high, low, close, volume, spread, bid, ask]

    Args:
        df: DataFrame chứa dữ liệu tick với các columns cần thiết.

    Returns:
        np.ndarray shape (n_samples, TICK_SEQ_LEN, TICK_FEAT_DIM)
    """
    # Map columns: cố gắng tìm các column phù hợp
    ohlcv_cols = {"open": None, "high": None, "low": None, "close": None, "volume": None}
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ohlcv_cols:
            ohlcv_cols[col_lower] = col

    # Tìm spread, bid, ask
    spread_col = None
    bid_col = None
    ask_col = None
    for col in df.columns:
        cl = col.lower().strip()
        if cl == "spread":
            spread_col = col
        elif cl in ("bid", "bid_price"):
            bid_col = col
        elif cl in ("ask", "ask_price"):
            ask_col = col

    # Kiểm tra các column cần thiết
    missing = [k for k, v in ohlcv_cols.items() if v is None]
    if missing:
        raise ValueError(
            f"Thiếu columns OHLCV trong dữ liệu: {missing}. "
            f"Các columns hiện tại: {list(df.columns)}"
        )

    n_samples = len(df)
    n_sequences = n_samples // TICK_SEQ_LEN
    if n_sequences == 0:
        raise ValueError(
            f"Dữ liệu quá ít ({n_samples} rows) để tạo sequence "
            f"với độ dài {TICK_SEQ_LEN}"
        )

    n_usable = n_sequences * TICK_SEQ_LEN

    # Build feature array
    features = np.zeros((n_sequences, TICK_SEQ_LEN, TICK_FEAT_DIM), dtype=np.float64)

    for i in range(TICK_FEAT_DIM):
        if i < 5:  # OHLCV
            col_name = ohlcv_cols[["open", "high", "low", "close", "volume"][i]]
            values = df[col_name].values[:n_usable].astype(np.float64)
        elif i == 5:  # spread
            if spread_col:
                values = df[spread_col].values[:n_usable].astype(np.float64)
            else:
                values = np.zeros(n_usable, dtype=np.float64)
        elif i == 6:  # bid
            if bid_col:
                values = df[bid_col].values[:n_usable].astype(np.float64)
            else:
                values = np.zeros(n_usable, dtype=np.float64)
        elif i == 7:  # ask
            if ask_col:
                values = df[ask_col].values[:n_usable].astype(np.float64)
            else:
                values = np.zeros(n_usable, dtype=np.float64)
        else:
            values = np.zeros(n_usable, dtype=np.float64)

        features[:, :, i] = values.reshape(n_sequences, TICK_SEQ_LEN)

    return features


def compute_bar_features(df: pd.DataFrame, timeframe: str = "M1") -> np.ndarray:
    """Compute 12 bar features từ OHLCV data.

    12 features: [open, high, low, close, volume,
                  wick_upper, wick_lower, body, range, vwap, spread, tick_count]

    Args:
        df: DataFrame với OHLCV columns
        timeframe: Tên timeframe (cho logging)

    Returns:
        np.ndarray shape (n_sequences, BAR_SEQ_LEN, BAR_FEAT_DIM)
    """
    ohlcv_cols = {"open": None, "high": None, "low": None, "close": None, "volume": None}
    for col in df.columns:
        cl = col.lower().strip()
        if cl in ohlcv_cols:
            ohlcv_cols[cl] = col

    missing = [k for k, v in ohlcv_cols.items() if v is None]
    if missing:
        raise ValueError(
            f"Thiếu columns OHLCV cho {timeframe}: {missing}"
        )

    open_col = ohlcv_cols["open"]
    high_col = ohlcv_cols["high"]
    low_col = ohlcv_cols["low"]
    close_col = ohlcv_cols["close"]
    volume_col = ohlcv_cols["volume"]

    n_samples = len(df)
    # Tìm tick_count & spread
    tick_count_col = None
    spread_col = None
    for col in df.columns:
        cl = col.lower().strip()
        if cl == "tick_count" or cl == "ticks":
            tick_count_col = col
        elif cl == "spread" and spread_col is None:
            spread_col = col

    n_sequences = n_samples // BAR_SEQ_LEN
    if n_sequences == 0:
        return np.zeros((0, BAR_SEQ_LEN, BAR_FEAT_DIM), dtype=np.float64)

    n_usable = n_sequences * BAR_SEQ_LEN

    features = np.zeros((n_sequences, BAR_SEQ_LEN, BAR_FEAT_DIM), dtype=np.float64)

    for i in range(n_sequences):
        start = i * BAR_SEQ_LEN
        end = start + BAR_SEQ_LEN
        slice_df = df.iloc[start:end]

        # 0: open
        features[i, :, 0] = slice_df[open_col].values.astype(np.float64)
        # 1: high
        features[i, :, 1] = slice_df[high_col].values.astype(np.float64)
        # 2: low
        features[i, :, 2] = slice_df[low_col].values.astype(np.float64)
        # 3: close
        features[i, :, 3] = slice_df[close_col].values.astype(np.float64)
        # 4: volume
        features[i, :, 4] = slice_df[volume_col].values.astype(np.float64)

        opens = features[i, :, 0]
        highs = features[i, :, 1]
        lows = features[i, :, 2]
        closes = features[i, :, 3]

        # 5: wick_upper = high - max(open, close)
        features[i, :, 5] = highs - np.maximum(opens, closes)
        # 6: wick_lower = min(open, close) - low
        features[i, :, 6] = np.minimum(opens, closes) - lows
        # 7: body = abs(close - open)
        features[i, :, 7] = np.abs(closes - opens)
        # 8: range = high - low
        features[i, :, 8] = highs - lows
        # 9: vwap = (high + low + close) / 3
        features[i, :, 9] = (highs + lows + closes) / 3.0

        # 10: spread
        if spread_col:
            features[i, :, 10] = slice_df[spread_col].values.astype(np.float64)
        else:
            features[i, :, 10] = 0.0

        # 11: tick_count
        if tick_count_col:
            features[i, :, 11] = slice_df[tick_count_col].values.astype(np.float64)
        else:
            features[i, :, 11] = 0.0

    return features


def build_mtf_bar_seqs(
    df: pd.DataFrame,
    tf_bar_counts: dict[str, int] = TF_BAR_COUNTS,
) -> dict[str, np.ndarray]:
    """Build bar_seqs cho tất cả timeframes.

    Args:
        df: DataFrame với OHLCV data (đã có timeframe info)
        tf_bar_counts: Dict {tf_name: số bars mỗi sequence}

    Returns:
        Dict {tf_name: np.ndarray [n_sequences, bar_seq_len, BAR_FEAT_DIM]}
    """
    bar_seqs: dict[str, np.ndarray] = {}

    for tf in TF_NAMES:
        # Filter cho timeframe nếu có column timeframe
        if "timeframe" in df.columns:
            tf_df = df[df["timeframe"].str.upper().str.strip() == tf].copy()
            if tf_df.empty:
                logger.warning(f"Không có dữ liệu cho timeframe {tf}, bỏ qua.")
                bar_seqs[tf] = np.zeros((0, BAR_SEQ_LEN, BAR_FEAT_DIM), dtype=np.float64)
                continue
        else:
            tf_df = df

        features = compute_bar_features(tf_df, timeframe=tf)
        bar_seqs[tf] = features
        logger.debug(f"  {tf}: {features.shape}")

    return bar_seqs


# =============================================================================
# Temporal Split
# =============================================================================

def temporal_split(
    tick_data: np.ndarray,
    bar_seqs: dict[str, np.ndarray],
    timestamps: pd.DatetimeIndex,
) -> dict[str, dict[str, Any]]:
    """Chia dữ liệu theo temporal split.

    Train: 2022 -> 2023
    Val:   2024-Q1
    Test:  2024-Q2

    Args:
        tick_data: tick_seq array (n_sequences, TICK_SEQ_LEN, TICK_FEAT_DIM)
        bar_seqs: dict {tf: np.ndarray (n_sequences, BAR_SEQ_LEN, BAR_FEAT_DIM)}
        timestamps: DatetimeIndex cho mỗi sequence (dùng timestamp của bar cuối)

    Returns:
        Dict với keys 'train', 'val', 'test',
        mỗi key chứa {'tick': ..., 'bars': {tf: ..., ...}, 'indices': ...}
    """
    train_mask = timestamps <= TRAIN_END
    val_mask = (timestamps > TRAIN_END) & (timestamps <= VAL_END)
    test_mask = timestamps > VAL_END

    n_total = len(timestamps)
    splits: dict[str, dict[str, Any]] = {}

    for split_name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        indices = np.where(mask.values)[0]
        if len(indices) == 0:
            logger.warning(f"Split '{split_name}' không có dữ liệu!")
            splits[split_name] = {"tick": np.zeros((0, *tick_data.shape[1:]), dtype=np.float64),
                                  "bars": {}, "indices": indices}
            continue

        split_tick = tick_data[indices]
        split_bars: dict[str, np.ndarray] = {}
        for tf in TF_NAMES:
            tf_data = bar_seqs.get(tf)
            if tf_data is not None and len(tf_data) > 0:
                split_bars[tf] = tf_data[indices]
            else:
                split_bars[tf] = np.zeros((0, BAR_SEQ_LEN, BAR_FEAT_DIM), dtype=np.float64)

        splits[split_name] = {"tick": split_tick, "bars": split_bars, "indices": indices}
        logger.info(
            f"  {split_name}: {len(indices)} sequences "
            f"({len(indices) / n_total * 100:.1f}%)"
        )

    return splits


# =============================================================================
# Normalization
# =============================================================================

def compute_robust_scalers(
    splits: dict[str, dict[str, Any]],
    tick_feat_dim: int = TICK_FEAT_DIM,
    bar_feat_dim: int = BAR_FEAT_DIM,
) -> dict[str, RobustScaler]:
    """Fit RobustScaler trên tập train cho mỗi feature group.

    Args:
        splits: Dict chứa dữ liệu các split
        tick_feat_dim: Số feature của tick
        bar_feat_dim: Số feature của bar

    Returns:
        Dict với keys: 'tick_scaler', cùng với 'bar_scaler_{tf}' cho mỗi TF
    """
    if RobustScaler is None:
        raise ImportError("scikit-learn required for RobustScaler. Run: pip install scikit-learn")

    scalers: dict[str, RobustScaler] = {}

    train_tick = splits["train"]["tick"]
    if train_tick.size > 0:
        # Flatten: (n, seq_len, feat_dim) -> (n * seq_len, feat_dim)
        train_tick_2d = train_tick.reshape(-1, tick_feat_dim)
        tick_scaler = RobustScaler(quantile_range=(5.0, 95.0))
        tick_scaler.fit(train_tick_2d)
        scalers["tick_scaler"] = tick_scaler
        logger.info(f"Tick scaler fitted: center.shape={tick_scaler.center_.shape}")
    else:
        raise ValueError("Không có dữ liệu train để fit scaler.")

    # Bar scalers: fit per TF
    for tf in TF_NAMES:
        train_bars = splits["train"]["bars"].get(tf)
        if train_bars is not None and train_bars.size > 0:
            train_bar_2d = train_bars.reshape(-1, bar_feat_dim)
            bar_scaler = RobustScaler(quantile_range=(5.0, 95.0))
            bar_scaler.fit(train_bar_2d)
            scalers[f"bar_scaler_{tf}"] = bar_scaler
            logger.debug(f"  Bar scaler {tf} fitted: {bar_scaler.center_.shape}")
        else:
            logger.warning(f"  Bar scaler {tf}: không có dữ liệu train, skip.")

    return scalers


def transform_with_scalers(
    splits: dict[str, dict[str, Any]],
    scalers: dict[str, RobustScaler],
    tick_feat_dim: int = TICK_FEAT_DIM,
    bar_feat_dim: int = BAR_FEAT_DIM,
) -> dict[str, dict[str, Any]]:
    """Apply normalization (RobustScaler) cho tất cả splits.

    Args:
        splits: Dict dữ liệu các split
        scalers: Dict các scalers đã fit
        tick_feat_dim: Số feature tick
        bar_feat_dim: Số feature bar

    Returns:
        Dict splits với dữ liệu đã normalized
    """
    normalized: dict[str, dict[str, Any]] = {}

    for split_name, split_data in splits.items():
        tick = split_data["tick"]
        if tick.size > 0:
            orig_shape = tick.shape
            tick_2d = tick.reshape(-1, tick_feat_dim)
            tick_norm = scalers["tick_scaler"].transform(tick_2d)
            tick_norm = tick_norm.reshape(orig_shape).astype(np.float32)
        else:
            tick_norm = tick.astype(np.float32)

        bars_norm: dict[str, np.ndarray] = {}
        for tf in TF_NAMES:
            tf_data = split_data["bars"].get(tf)
            if tf_data is not None and tf_data.size > 0 and f"bar_scaler_{tf}" in scalers:
                orig_shape = tf_data.shape
                bar_2d = tf_data.reshape(-1, bar_feat_dim)
                bar_norm = scalers[f"bar_scaler_{tf}"].transform(bar_2d)
                bar_norm = bar_norm.reshape(orig_shape).astype(np.float32)
                bars_norm[tf] = bar_norm
            elif tf_data is not None:
                bars_norm[tf] = tf_data.astype(np.float32)
            else:
                bars_norm[tf] = np.zeros((0, BAR_SEQ_LEN, bar_feat_dim), dtype=np.float32)

        normalized[split_name] = {"tick": tick_norm, "bars": bars_norm}

    return normalized


# =============================================================================
# Save to HDF5
# =============================================================================

def save_hdf5_dataset(
    data: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    """Save dataset to HDF5 files (train.h5, val.h5, test.h5).

    Args:
        data: Dict with keys 'train', 'val', 'test',
              mỗi key chứa {'tick': ..., 'bars': {tf: ..., ...}}
        output_dir: Thư mục output
    """
    if h5py is None:
        raise ImportError("h5py required. Run: pip install h5py")

    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        split_data = data.get(split_name)
        if split_data is None or split_data["tick"].size == 0:
            logger.warning(f"Split '{split_name}' rỗng, bỏ qua save.")
            continue

        h5_path = output_dir / f"{split_name}.h5"
        logger.info(f"Đang lưu {split_name} -> {h5_path} ...")

        with h5py.File(str(h5_path), "w") as f:
            # Save tick data
            f.create_dataset(
                "tick",
                data=split_data["tick"],
                compression="gzip",
                compression_opts=6,
                chunks=True,
            )

            # Save bar data per TF
            bars_group = f.create_group("bars")
            for tf in TF_NAMES:
                tf_data = split_data["bars"].get(tf)
                if tf_data is not None and tf_data.size > 0:
                    bars_group.create_dataset(
                        tf,
                        data=tf_data,
                        compression="gzip",
                        compression_opts=6,
                        chunks=True,
                    )

            # Ghi metadata
            meta_group = f.create_group("meta")
            meta_group.attrs["tick_seq_len"] = TICK_SEQ_LEN
            meta_group.attrs["bar_seq_len"] = BAR_SEQ_LEN
            meta_group.attrs["tick_feat_dim"] = TICK_FEAT_DIM
            meta_group.attrs["bar_feat_dim"] = BAR_FEAT_DIM
            meta_group.attrs["num_timeframes"] = NUM_TF
            meta_group.attrs["timeframes"] = [tf.encode("utf-8") for tf in TF_NAMES]
            meta_group.attrs["n_sequences"] = split_data["tick"].shape[0]

            logger.info(
                f"  {split_name}: {split_data['tick'].shape[0]} sequences, "
                f"tick={split_data['tick'].shape}, "
                f"bars={{tf: shape}}"
            )

    logger.success(f"Dataset HDF5 đã lưu vào {output_dir.resolve()}")


def save_feature_scaler(
    scalers: dict[str, RobustScaler],
    output_path: Path,
) -> None:
    """Save feature scalers to disk.

    Args:
        scalers: Dict các RobustScaler
        output_path: Đường dẫn file .pkl
    """
    if joblib is None:
        raise ImportError("joblib required. Run: pip install scikit-learn")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scalers, str(output_path))
    logger.info(f"Feature scalers saved to {output_path}")


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline(
    data_dir: Path,
    output_dir: Path,
    model_dir: Path,
) -> None:
    """Run the full dataset building pipeline.

    Args:
        data_dir: Thư mục chứa parquet files
        output_dir: Thư mục output cho HDF5 files
        model_dir: Thư mục lưu scaler
    """
    logger.info("=" * 60)
    logger.info("BẮT ĐẦU BUILD LSTM DATASET")
    logger.info("=" * 60)

    # Step 1: Discover and load parquet files
    logger.info("[1/5] Đang đọc parquet files...")
    parquet_files = discover_parquet_files(data_dir)
    df = load_and_concatenate_parquet(parquet_files)
    logger.info(f"  Tổng số rows: {len(df)}, columns: {list(df.columns)}")

    # Step 2: Build tick features
    logger.info("[2/5] Đang build tick_seq[128x8]...")
    tick_data = build_tick_features(df)
    logger.info(f"  tick_seq shape: {tick_data.shape}")

    # Step 3: Build MTF bar sequences
    logger.info("[3/5] Đang build bar_seqs[6 TF x 30 x 12]...")
    bar_seqs = build_mtf_bar_seqs(df)
    for tf, arr in bar_seqs.items():
        logger.info(f"  {tf}: {arr.shape}")

    # Đồng bộ hóa số sequences giữa tick và bars
    n_tick_seqs = tick_data.shape[0]
    min_seqs = n_tick_seqs
    for tf, arr in bar_seqs.items():
        if arr.shape[0] > 0:
            min_seqs = min(min_seqs, arr.shape[0])

    if min_seqs < n_tick_seqs:
        logger.info(f"  Đồng bộ hóa sequences: {n_tick_seqs} -> {min_seqs}")
        tick_data = tick_data[:min_seqs]
        for tf in TF_NAMES:
            if bar_seqs[tf].shape[0] > 0:
                bar_seqs[tf] = bar_seqs[tf][:min_seqs]

    # Tạo timestamps cho temporal split từ last timestamp của mỗi window
    # (dùng timestamp của bar cuối cùng trong mỗi sequence)
    seq_timestamps_list: list[pd.Timestamp] = []
    n_seqs = tick_data.shape[0]
    for i in range(n_seqs):
        # Lấy timestamp ở vị trí (i+1)*TICK_SEQ_LEN - 1
        idx = min((i + 1) * TICK_SEQ_LEN - 1, len(df) - 1)
        seq_timestamps_list.append(df["timestamp"].iloc[idx])
    seq_timestamps = pd.DatetimeIndex(seq_timestamps_list)

    # Step 4: Temporal split and normalize
    logger.info("[4/5] Đang temporal split và normalize...")
    splits = temporal_split(tick_data, bar_seqs, seq_timestamps)

    # Fit scalers
    scalers = compute_robust_scalers(splits)

    # Transform all splits
    normalized_splits = transform_with_scalers(splits, scalers)

    # Step 5: Save outputs
    logger.info("[5/5] Đang lưu dataset...")
    save_hdf5_dataset(normalized_splits, output_dir)

    scaler_path = model_dir / SCALER_PKL
    save_feature_scaler(scalers, scaler_path)

    # Summary
    logger.info("=" * 60)
    logger.info("HOÀN THÀNH BUILD LSTM DATASET")
    logger.info("=" * 60)
    for split_name in ["train", "val", "test"]:
        split_data = normalized_splits.get(split_name)
        if split_data is not None:
            n = split_data["tick"].shape[0]
            logger.info(f"  {split_name}: {n} sequences -> {output_dir / f'{split_name}.h5'}")
    logger.info(f"  Scaler: {scaler_path}")


# =============================================================================
# CLI
# =============================================================================

def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Build LSTM Autoencoder dataset from parquet files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/parquet",
        help="Thư mục chứa parquet files (default: data/parquet)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/lstm_dataset",
        help="Thư mục output cho HDF5 files (default: data/lstm_dataset)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/lstm",
        help="Thư mục lưu scaler (default: models/lstm)",
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
    """Entry point cho build_lstm_dataset script."""
    args = parse_args()

    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level=args.log_level, format="<level>{level: <8}</level> | {message}")

    base_dir = Path(__file__).resolve().parent.parent
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = base_dir / data_dir

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir

    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = base_dir / model_dir

    try:
        run_pipeline(data_dir, output_dir, model_dir)
    except Exception as e:
        logger.error(f"Pipeline thất bại: {e}")
        raise


if __name__ == "__main__":
    main()
