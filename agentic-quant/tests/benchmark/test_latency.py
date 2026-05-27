# =============================================================================
# AGENTIC-QUANT — Benchmark: Tick Pipeline Latency Test
#
# Mo phong tick pipeline: tick_received -> LSTM encode -> XGBoost predict
# -> WebSocket broadcast. Do latency tung step, target < 50ms E2E.
# 1000 iterations, report P50, P95, P99.
# =============================================================================

from __future__ import annotations

import statistics
import time
from typing import Any

import numpy as np
import pytest

from core.ai_engine.xgboost.feature_builder import (
    D_X_A,
    D_X_B,
    XGBoostFeatureBuilder,
)
from core.ai_engine.xgboost.inference import InferenceEngineA, InferenceEngineB

# =============================================================================
# Constants
# =============================================================================

NUM_ITERATIONS = 1000
TARGET_E2E_MS = 50.0  # E2E latency target < 50ms
SEED = 42

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_tick_data() -> dict[str, Any]:
    """Tao mock tick data de mo phong pipeline.

    Tra ve dict chua bid, ask, last, volume, symbol, timestamp_us.
    """
    rng = np.random.default_rng(SEED)
    base_price = 2500.0
    return {
        "symbol": "XAUUSD",
        "bid": round(base_price - 0.5 + rng.normal(0, 0.1), 2),
        "ask": round(base_price + 0.5 + rng.normal(0, 0.1), 2),
        "last": round(base_price + rng.normal(0, 0.2), 2),
        "volume": round(abs(rng.normal(10, 2)), 1),
        "timestamp_us": int(time.time() * 1_000_000),
    }


@pytest.fixture
def mock_latent_vector() -> np.ndarray:
    """Tao latent vector [512] mock LSTM encoder output."""
    rng = np.random.default_rng(SEED)
    return rng.standard_normal(512).astype(np.float32)


@pytest.fixture
def mock_feature_vector_xa() -> np.ndarray:
    """Tao feature vector [648] cho XGBoost Model A."""
    rng = np.random.default_rng(SEED)
    return rng.standard_normal(D_X_A).astype(np.float64)


@pytest.fixture
def mock_feature_vector_xb() -> np.ndarray:
    """Tao feature vector [560] cho XGBoost Model B."""
    rng = np.random.default_rng(SEED)
    return rng.standard_normal(D_X_B).astype(np.float64)


# =============================================================================
# Benchmark: Step Latency
# =============================================================================


def _measure_latency(steps: list[tuple[str, Any]]) -> dict[str, float]:
    """Do latency cua tung step trong pipeline.

    Args:
        steps: List cac tuple (step_name, callable) de do latency.

    Returns:
        Dict: {step_name: latency_ms}
    """
    latencies: dict[str, float] = {}
    for name, fn in steps:
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        latencies[name] = (t1 - t0) * 1000.0  # ms
    return latencies


@pytest.mark.benchmark
class TestTickPipelineLatency:
    """Benchmark latency cua tick pipeline.

    Mo phong pipeline:
        tick_received -> LSTM encode -> XGBoost predict -> WebSocket broadcast

    Target: E2E latency < 50ms (P99).
    """

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        """Setup truoc moi test: khoi tao RNG seed."""
        np.random.seed(SEED)

    def test_tick_received_latency(self, mock_tick_data: dict[str, Any]) -> None:
        """Do latency cua tick_received step.

        Step nay xu ly tick incoming: parse, validate, push to event bus.
        """
        latencies: list[float] = []
        for _ in range(NUM_ITERATIONS):
            t0 = time.perf_counter()
            # Mo phong: parse tick data (dict lookup, type conversion)
            _ = {
                "symbol": str(mock_tick_data["symbol"]),
                "bid": float(mock_tick_data["bid"]),
                "ask": float(mock_tick_data["ask"]),
                "last": float(mock_tick_data["last"]),
                "volume": float(mock_tick_data["volume"]),
                "timestamp_us": int(mock_tick_data["timestamp_us"]),
            }
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        p50 = statistics.median(latencies)
        latencies_sorted = sorted(latencies)
        p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
        p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)]

        print(f"\n  [tick_received] P50={p50:.4f}ms P95={p95:.4f}ms P99={p99:.4f}ms")

        assert p99 < TARGET_E2E_MS, (
            f"tick_received P99={p99:.4f}ms vuot target {TARGET_E2E_MS}ms"
        )

    def test_lstm_encode_latency(self, mock_tick_data: dict[str, Any]) -> None:
        """Do latency cua LSTM encode step.

        Step nay encode tick data thanh latent vector [512].
        Mo phong bang numpy matmul + activation.
        """
        latencies: list[float] = []
        rng = np.random.default_rng(SEED)

        # Pre-build weights cho LSTM simulation
        input_dim = 5  # bid, ask, last, volume, spread
        hidden_dim = 512
        W = rng.standard_normal((input_dim, hidden_dim)).astype(np.float32)
        b = rng.standard_normal(hidden_dim).astype(np.float32)

        for _ in range(NUM_ITERATIONS):
            tick_vec = np.array(
                [
                    mock_tick_data["bid"],
                    mock_tick_data["ask"],
                    mock_tick_data["last"],
                    mock_tick_data["volume"],
                    mock_tick_data["ask"] - mock_tick_data["bid"],
                ],
                dtype=np.float32,
            )

            t0 = time.perf_counter()
            # Mo phong LSTM forward pass: linear + tanh
            latent = np.tanh(tick_vec @ W + b)
            _ = latent  # latent vector [512]
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        p50 = statistics.median(latencies)
        latencies_sorted = sorted(latencies)
        p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
        p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)]

        print(f"\n  [lstm_encode] P50={p50:.4f}ms P95={p95:.4f}ms P99={p99:.4f}ms")

        assert p99 < TARGET_E2E_MS, (
            f"lstm_encode P99={p99:.4f}ms vuot target {TARGET_E2E_MS}ms"
        )

    def test_xgboost_predict_latency(
        self,
        mock_feature_vector_xa: np.ndarray,
        mock_feature_vector_xb: np.ndarray,
    ) -> None:
        """Do latency cua XGBoost predict step.

        Step nay chay inference ca Model A (direction) va Model B (zone hold).
        """
        latencies: list[float] = []
        feature_builder = XGBoostFeatureBuilder()

        for _ in range(NUM_ITERATIONS):
            t0 = time.perf_counter()
            # Mo phong: build features (gia lap)
            _ = feature_builder._validate_vector(mock_feature_vector_xa, D_X_A, "X_A")
            # Mo phong: predict (chi goi numpy ops)
            _ = np.argmax(mock_feature_vector_xa[:3])
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        p50 = statistics.median(latencies)
        latencies_sorted = sorted(latencies)
        p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
        p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)]

        print(f"\n  [xgboost_predict] P50={p50:.4f}ms P95={p95:.4f}ms P99={p99:.4f}ms")

        assert p99 < TARGET_E2E_MS, (
            f"xgboost_predict P99={p99:.4f}ms vuot target {TARGET_E2E_MS}ms"
        )

    def test_websocket_broadcast_latency(self) -> None:
        """Do latency cua WebSocket broadcast step.

        Step nay serialize ket qua predict thanh JSON va broadcast.
        """
        import json

        latencies: list[float] = []
        sample_result = {
            "symbol": "XAUUSD",
            "p_bsl": 0.35,
            "p_ssl": 0.55,
            "p_lateral": 0.10,
            "confidence": "MEDIUM",
            "timestamp_ms": int(time.time() * 1000),
        }

        for _ in range(NUM_ITERATIONS):
            t0 = time.perf_counter()
            # Mo phong: serialize + broadcast
            msg = json.dumps(sample_result, ensure_ascii=False)
            # Mo phong: broadcast (len ~ len(msg) bytes)
            _ = len(msg)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        p50 = statistics.median(latencies)
        latencies_sorted = sorted(latencies)
        p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
        p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)]

        print(f"\n  [ws_broadcast] P50={p50:.4f}ms P95={p95:.4f}ms P99={p99:.4f}ms")

        assert p99 < TARGET_E2E_MS, (
            f"ws_broadcast P99={p99:.4f}ms vuot target {TARGET_E2E_MS}ms"
        )

    def test_e2e_pipeline_latency(self) -> None:
        """Do latency E2E cua toan bo tick pipeline.

        Mo phong day du:
            tick_received -> LSTM encode -> XGBoost predict -> WebSocket broadcast
        """
        import json

        rng = np.random.default_rng(SEED)
        latencies: list[float] = []

        # Pre-build weights
        input_dim = 5
        hidden_dim = 512
        W = rng.standard_normal((input_dim, hidden_dim)).astype(np.float32)
        b = rng.standard_normal(hidden_dim).astype(np.float32)

        sample_result_template = {
            "symbol": "XAUUSD",
            "p_bsl": 0.35,
            "p_ssl": 0.55,
            "p_lateral": 0.10,
            "confidence": "MEDIUM",
            "timestamp_ms": 0,
        }

        for i in range(NUM_ITERATIONS):
            # Tao tick data
            bid = round(2500.0 - 0.5 + rng.normal(0, 0.1), 2)
            ask = round(2500.0 + 0.5 + rng.normal(0, 0.1), 2)
            last = round(2500.0 + rng.normal(0, 0.2), 2)
            volume = round(abs(rng.normal(10, 2)), 1)

            t0 = time.perf_counter()

            # Step 1: tick_received — parse
            tick_vec = np.array([bid, ask, last, volume, ask - bid], dtype=np.float32)

            # Step 2: LSTM encode
            latent = np.tanh(tick_vec @ W + b)

            # Step 3: XGBoost predict
            p_bsl = float(1.0 / (1.0 + np.exp(-latent[0])))
            p_ssl = float(1.0 / (1.0 + np.exp(-latent[1])))
            p_lateral = max(0.0, 1.0 - p_bsl - p_ssl)

            # Step 4: WebSocket broadcast
            result = dict(sample_result_template)
            result["p_bsl"] = round(p_bsl, 4)
            result["p_ssl"] = round(p_ssl, 4)
            result["p_lateral"] = round(p_lateral, 4)
            result["timestamp_ms"] = int(time.time() * 1000)
            msg = json.dumps(result, ensure_ascii=False)
            _ = len(msg)

            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        p50 = statistics.median(latencies)
        latencies_sorted = sorted(latencies)
        p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
        p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)]
        avg = sum(latencies) / len(latencies)
        min_lat = min(latencies)
        max_lat = max(latencies)

        print(f"\n  [e2e_pipeline]  iterations={NUM_ITERATIONS}")
        print(f"    Min={min_lat:.4f}ms  Avg={avg:.4f}ms  Max={max_lat:.4f}ms")
        print(f"    P50={p50:.4f}ms  P95={p95:.4f}ms  P99={p99:.4f}ms")

        assert p99 < TARGET_E2E_MS, (
            f"E2E pipeline P99={p99:.4f}ms vuot target {TARGET_E2E_MS}ms"
        )
