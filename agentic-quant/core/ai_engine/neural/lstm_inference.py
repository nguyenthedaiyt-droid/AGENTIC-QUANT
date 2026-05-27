# =============================================================================
# AGENTIC-QUANT — LSTM Inference Engine
# Phase 5: LSTM Autoencoder Inference
#
# LSTMInferenceEngine:
#   - __init__(weights_path): load_state_dict, model.eval(), warmup
#   - @torch.no_grad() encode(tick_seq[8], bar_seqs_mtf[6][12]) -> z[512]
#   - Redis cache via set_latent_vector/get_latent_vector
#     key: latent:{symbol}:{bar_close_ts} TTL=300s
#   - Latency target < 20ms tren CPU
#   - Fallback: neu ko co Redis, tra numpy array truc tiep
# =============================================================================

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

from core.ai_engine.neural.hierarchical_lstm_ae import (
    HierarchicalLSTMAE,
    TICK_FEAT_DIM,
    BAR_FEAT_DIM,
    LATENT_DIM,
    NUM_TF,
)

# Redis cache manager (optional dependency)
_HAS_REDIS: bool = True
try:
    from core.memory.short_term.redis_cache_manager import (
        get_redis_cache_manager as _get_redis_cache_manager,
    )
except ImportError:
    _HAS_REDIS = False

# =============================================================================
# Constants
# =============================================================================

REDIS_LATENT_TTL: int = 300        # 300s = 5 phut
LATENCY_TARGET_MS: float = 20.0    # <20ms target
DEFAULT_NUM_THREADS: int = 4


# =============================================================================
# LSTMInferenceEngine
# =============================================================================

class LSTMInferenceEngine:
    """Inference engine cho HierarchicalLSTMAE.

    Chay tren CPU, <20ms latency.
    Tu dong cache latent vector trong Redis (TTL=300s).
    Fallback ve numpy array truc tiep neu Redis khong co.

    Usage:
        engine = LSTMInferenceEngine(weights_path="models/lstm_ae.pt")
        z = engine.encode(tick_seq, bar_seqs_mtf)
        # z: np.ndarray [512] float32
    """

    def __init__(
        self,
        weights_path: str | Path,
        device: str = "cpu",
        num_threads: int = DEFAULT_NUM_THREADS,
        use_redis: bool = True,
    ) -> None:
        """Khoi tao inference engine.

        Args:
            weights_path: Path den file .pt chua state_dict
            device: 'cpu' hoac 'cuda' mac dinh
            num_threads: So CPU threads cho PyTorch
            use_redis: Co dung Redis cache hay khong
        """
        self.weights_path = Path(weights_path)
        self.device = torch.device(device)
        self.use_redis = use_redis

        # Cau hinh CPU threads
        if self.device.type == "cpu":
            torch.set_num_threads(num_threads)

        # Model
        self.model: HierarchicalLSTMAE | None = None

        # Redis (lazy init)
        self._redis = None
        self._redis_available: bool = False

        # Stats
        self._inference_count: int = 0
        self._total_latency_ms: float = 0.0

        # --- Load weights ---
        self._load_weights()
        self._warmup()

    # =========================================================================
    # Model Loading
    # =========================================================================

    def _load_weights(self) -> None:
        """Load weights tu checkpoint vao model.

        Raises:
            FileNotFoundError: Neu checkpoint ko ton tai
            RuntimeError: Neu load state_dict that bai
        """
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Khong tim thay model checkpoint tai: {self.weights_path.resolve()}"
            )

        logger.info(f"Dang load LSTM Autoencoder tu: {self.weights_path}")

        # Tao model architecture
        self.model = HierarchicalLSTMAE()

        try:
            checkpoint = torch.load(
                self.weights_path,
                map_location=self.device,
                weights_only=True,
            )

            # Xu ly ca raw state_dict va dict chua 'model_state_dict'
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                logger.debug(
                    f"Checkpoint epoch={checkpoint.get('epoch', 'N/A')}, "
                    f"loss={checkpoint.get('loss', 'N/A')}"
                )
            else:
                state_dict = checkpoint

            self.model.load_state_dict(state_dict, strict=False)

        except Exception as e:
            raise RuntimeError(
                f"Loi load model checkpoint tu {self.weights_path}: {e}"
            ) from e

        self.model.to(self.device)
        self.model.eval()

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"LSTM Autoencoder loaded: {n_params:,} params")

    # =========================================================================
    # Warmup
    # =========================================================================

    def _warmup(self) -> None:
        """Chay 1 forward pass voi dummy data de warmup CPU/PyTorch kernels."""
        if self.model is None:
            return

        logger.debug("Warmup LSTM Autoencoder...")
        batch_size = 1

        dummy_tick = torch.randn(batch_size, 128, TICK_FEAT_DIM, device=self.device)
        dummy_bars = [
            torch.randn(batch_size, 30, BAR_FEAT_DIM, device=self.device)
            for _ in range(NUM_TF)
        ]

        with torch.no_grad():
            _ = self.model.encode(dummy_tick, dummy_bars)

        logger.debug("Warmup hoan tat")

    # =========================================================================
    # Redis Cache Integration
    # =========================================================================

    def _init_redis(self) -> None:
        """Khoi tao Redis connection (lazy, chi goi 1 lan)."""
        if self._redis is not None or not self.use_redis:
            return

        if not _HAS_REDIS:
            self._redis_available = False
            logger.debug("Redis module khong available, bo qua cache")
            return

        try:
            self._redis = _get_redis_cache_manager()
            self._redis_available = True
            logger.debug("Redis cache available cho LSTM inference")
        except Exception as e:
            logger.warning(f"Redis unavailable: {e}")
            self._redis_available = False

    def _build_cache_key(self, symbol: str, bar_close_ts: int) -> str:
        """Xay dung Redis key: latent:{symbol}:{bar_close_ts}."""
        return f"latent:{symbol}:{bar_close_ts}"

    def _get_cached_z(
        self, symbol: str, bar_close_ts: int
    ) -> np.ndarray | None:
        """Doc latent vector tu Redis cache (blocking call).

        Args:
            symbol: Ma symbol (vi du 'XAUUSD')
            bar_close_ts: Timestamp dong bar (unix ms)

        Returns:
            np.ndarray [512] hoac None neu cache miss
        """
        if not self._redis_available or self._redis is None:
            return None

        try:
            # get_latent_vector la async, can chay trong event loop
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                cached = loop.run_until_complete(
                    self._redis.get_latent_vector(symbol, bar_close_ts)
                )
            finally:
                loop.close()

            if cached is not None:
                logger.debug(f"Redis cache HIT: latent:{symbol}:{bar_close_ts}")
                return np.array(cached, dtype=np.float32)

        except Exception as e:
            logger.debug(f"Redis get that bai (fallback): {e}")

        return None

    def _set_cached_z(
        self, symbol: str, bar_close_ts: int, z: np.ndarray
    ) -> None:
        """Luu latent vector vao Redis cache (fire-and-forget).

        Args:
            symbol: Ma symbol
            bar_close_ts: Timestamp dong bar
            z: np.ndarray [512]
        """
        if not self._redis_available or self._redis is None:
            return

        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    self._redis.set_latent_vector(
                        symbol, bar_close_ts, z.tolist()
                    )
                )
                # Set TTL=300s (override default 3600)
                key = self._build_cache_key(symbol, bar_close_ts)
                loop.run_until_complete(
                    self._redis.client.expire(key, REDIS_LATENT_TTL)
                )
            finally:
                loop.close()
        except Exception as e:
            logger.debug(f"Redis set that bai: {e}")

    # =========================================================================
    # Core Inference
    # =========================================================================

    @torch.no_grad()
    def encode(
        self,
        tick_seq: np.ndarray | torch.Tensor,
        bar_seqs_mtf: list[np.ndarray | torch.Tensor],
        symbol: str = "",
        bar_close_ts: int = 0,
    ) -> np.ndarray:
        """Encode inputs -> latent vector z.

        @torch.no_grad() dam bao khong tinh gradient.

        Neu symbol + bar_close_ts duoc cung cap, tu dong kiem tra
        Redis cache truoc khi inference.

        Args:
            tick_seq: [batch, seq_len, 8] hoac [seq_len, 8]
            bar_seqs_mtf: list cua 6 tensors [batch, seq_len_tf, 12]
                          hoac [seq_len_tf, 12]
            symbol: Symbol name (cho Redis cache key)
            bar_close_ts: Bar close timestamp (cho Redis cache key)

        Returns:
            z: np.ndarray [batch, 512] hoac [512] float32
        """
        # ---- Redis cache check ----
        if symbol and bar_close_ts > 0:
            self._init_redis()
            cached = self._get_cached_z(symbol, bar_close_ts)
            if cached is not None:
                return cached

        # ---- Inference ----
        if self.model is None:
            raise RuntimeError("Model chua duoc load")

        start_time = time.perf_counter()

        # Convert numpy -> torch
        tick = self._to_tensor(tick_seq)
        bars = [self._to_tensor(b) for b in bar_seqs_mtf]

        # Dam bao batch dimension
        single_input = tick.dim() == 2
        if single_input:
            tick = tick.unsqueeze(0)
            bars = [b.unsqueeze(0) for b in bars]

        # Move to device
        tick = tick.to(self.device)
        bars = [b.to(self.device) for b in bars]

        # Encode
        z = self.model.encode(tick, bars)  # [batch, 512]

        # Convert to numpy
        z_np = z.cpu().numpy().astype(np.float32)

        # Squeeze batch neu single input
        if single_input:
            z_np = z_np.squeeze(0)  # [512]

        # Latency tracking
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self._inference_count += 1
        self._total_latency_ms += elapsed_ms

        if elapsed_ms > LATENCY_TARGET_MS:
            logger.warning(
                f"Inference latency: {elapsed_ms:.2f}ms "
                f"(target: <{LATENCY_TARGET_MS}ms)"
            )

        # ---- Redis cache set (async fire-and-forget) ----
        if symbol and bar_close_ts > 0 and self._redis_available:
            self._set_cached_z(symbol, bar_close_ts, z_np)

        return z_np

    # =========================================================================
    # Utility
    # =========================================================================

    @staticmethod
    def _to_tensor(data: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert numpy array sang torch tensor float32."""
        if isinstance(data, np.ndarray):
            return torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))
        if isinstance(data, torch.Tensor):
            return data.float().contiguous()
        raise TypeError(f"Khong ho tro kieu du lieu: {type(data)}")

    def get_stats(self) -> dict[str, Any]:
        """Tra ve thong ke inference."""
        avg_ms = (
            self._total_latency_ms / max(self._inference_count, 1)
        )
        return {
            "inference_count": self._inference_count,
            "avg_latency_ms": round(avg_ms, 2),
            "total_latency_ms": round(self._total_latency_ms, 2),
            "redis_available": self._redis_available,
            "device": str(self.device),
        }

    def reset_stats(self) -> None:
        """Reset thong ke."""
        self._inference_count = 0
        self._total_latency_ms = 0.0


# =============================================================================
# Helper: synchronous encode khong can tao engine
# =============================================================================

def encode_sync(
    engine: LSTMInferenceEngine,
    tick_seq: np.ndarray,
    bar_seqs_mtf: list[np.ndarray],
    symbol: str = "",
    bar_close_ts: int = 0,
) -> np.ndarray:
    """Sync wrapper cho engine.encode().

    Args:
        engine: LSTMInferenceEngine instance (da init)
        tick_seq: [128, 8] hoac [batch, 128, 8]
        bar_seqs_mtf: list 6 x [seq_len, 12] hoac [batch, seq_len, 12]
        symbol: Symbol name (optional)
        bar_close_ts: Bar timestamp (optional)

    Returns:
        z: [512] hoac [batch, 512] float32
    """
    return engine.encode(tick_seq, bar_seqs_mtf, symbol, bar_close_ts)


# =============================================================================
# Unit-test
# =============================================================================

if __name__ == "__main__":
    print("Testing LSTMInferenceEngine (without checkpoint)...")

    # Tao engine voi checkpoint ko ton tai -> se fallback ve init truc tiep
    model = HierarchicalLSTMAE()
    model.eval()

    # Fake engine
    class _FakeEngine:
        def __init__(self) -> None:
            self.model = model
            self.device = torch.device("cpu")
            self._inference_count = 0
            self._total_latency_ms = 0.0
            self._redis = None
            self._redis_available = False

        @torch.no_grad()
        def encode(
            self,
            tick_seq: np.ndarray | torch.Tensor,
            bar_seqs_mtf: list[np.ndarray | torch.Tensor],
            symbol: str = "",
            bar_close_ts: int = 0,
        ) -> np.ndarray:
            start = time.perf_counter()

            tick = LSTMInferenceEngine._to_tensor(tick_seq)
            bars = [LSTMInferenceEngine._to_tensor(b) for b in bar_seqs_mtf]

            single = tick.dim() == 2
            if single:
                tick = tick.unsqueeze(0)
                bars = [b.unsqueeze(0) for b in bars]

            z = self.model.encode(tick, bars)
            z_np = z.cpu().numpy().astype(np.float32)
            if single:
                z_np = z_np.squeeze(0)

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._inference_count += 1
            self._total_latency_ms += elapsed_ms
            print(f"  encode() latency: {elapsed_ms:.2f}ms")
            return z_np

    engine = _FakeEngine()

    tick_seq = np.random.randn(128, TICK_FEAT_DIM).astype(np.float32)
    bar_seqs_mtf = [
        np.random.randn(30, BAR_FEAT_DIM).astype(np.float32)
        for _ in range(NUM_TF)
    ]

    # Encode
    z = engine.encode(tick_seq, bar_seqs_mtf)
    print(f"z shape: {z.shape}")      # (512,)
    print(f"z dtype: {z.dtype}")      # float32

    # Batch test
    tick_batch = np.random.randn(2, 128, TICK_FEAT_DIM).astype(np.float32)
    bar_batch = [
        np.random.randn(2, 30, BAR_FEAT_DIM).astype(np.float32)
        for _ in range(NUM_TF)
    ]
    z_batch = engine.encode(tick_batch, bar_batch)
    print(f"z batch shape: {z_batch.shape}")  # (2, 512)

    print("\n✅ LSTMInferenceEngine smoke test passed!")
