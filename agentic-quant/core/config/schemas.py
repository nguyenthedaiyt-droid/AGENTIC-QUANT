# =============================================================================
# AGENTIC-QUANT — Configuration Schemas
# Dinh nghia Pydantic models validate config khi load
# =============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# System Config
# =============================================================================

class PortConfig(BaseModel):
    zmq_pull: int = Field(default=5556, ge=1024, le=65535)
    zmq_pub: int = Field(default=5557, ge=1024, le=65535)
    websocket: int = Field(default=47290, ge=1024, le=65535)
    websocket_fallback_start: int = Field(default=47291, ge=1024, le=65535)
    websocket_fallback_end: int = Field(default=47299, ge=1024, le=65535)
    tv_webhook: int = Field(default=8080, ge=1024, le=65535)
    prometheus: int = Field(default=9090, ge=1024, le=65535)

    @field_validator("websocket")
    @classmethod
    def check_websocket_not_in_fallback_range(cls, v: int) -> int:
        if 47291 <= v <= 47299:
            raise ValueError(f"WebSocket port {v} nam trong fallback range 47291-47299")
        return v


class RedisConfig(BaseModel):
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=6379, ge=1024, le=65535)
    db: int = Field(default=0, ge=0, le=15)
    password: str | None = Field(default=None)
    maxmemory_mb: int = Field(default=512, ge=64)
    ssl: bool = Field(default=False)


class VectorDBConfig(BaseModel):
    provider: Literal["qdrant", "chromadb"] = Field(default="qdrant")
    url: str = Field(default="http://127.0.0.1:6333")
    collection_debate: str = Field(default="debate_archive")
    collection_zones: str = Field(default="zone_embeddings")


class FeatureThresholds(BaseModel):
    tick_abnormal_spread_pips: float = Field(default=0.5, ge=0)
    atr_period: int = Field(default=14, ge=1)
    atr_tf_m1: int = Field(default=14)
    atr_tf_m5: int = Field(default=14)
    atr_tf_m15: int = Field(default=14)
    atr_tf_h1: int = Field(default=14)
    atr_tf_h4: int = Field(default=14)
    atr_tf_d1: int = Field(default=14)


class LeakyQueueConfig(BaseModel):
    max_size: int = Field(default=10000, ge=100)
    sample_threshold: int = Field(default=8000, ge=100)
    k_dynamic_enabled: bool = Field(default=True)


class SystemConfig(BaseModel):
    name: str = Field(default="AGENTIC-QUANT")
    version: str = Field(default="0.1.0")
    environment: Literal["development", "staging", "production"] = Field(default="development")
    symbol: str = Field(default="XAUUSD")
    ports: PortConfig = Field(default_factory=PortConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    vectordb: VectorDBConfig = Field(default_factory=VectorDBConfig)
    tick_thresholds: FeatureThresholds = Field(default_factory=FeatureThresholds)
    leaky_queue: LeakyQueueConfig = Field(default_factory=LeakyQueueConfig)
    backtest_mode: bool = Field(default=False)
    forward_lock_enabled: bool = Field(default=True)


# =============================================================================
# Killzone Config
# =============================================================================

class SessionWeight(BaseModel):
    ltf: float = Field(default=1.0, ge=0)
    htf: float = Field(default=1.0, ge=0)


class SessionConfig(BaseModel):
    id: str
    name: str
    start_hour_utc: int = Field(ge=0, le=23)
    start_minute_utc: int = Field(ge=0, le=59)
    end_hour_utc: int = Field(ge=0, le=23)
    end_minute_utc: int = Field(ge=0, le=59)
    color: str
    session_weight: SessionWeight = Field(default_factory=SessionWeight)


class KillzoneConfig(BaseModel):
    sessions: list[SessionConfig] = Field(default_factory=list)

    def get_session_by_id(self, session_id: str) -> SessionConfig | None:
        for s in self.sessions:
            if s.id == session_id:
                return s
        return None


# =============================================================================
# Model Params Config
# =============================================================================

class LSTMEncoderConfig(BaseModel):
    hidden_dim: int = Field(ge=16)
    num_layers: int = Field(ge=1, le=8)
    bidirectional: bool = Field(default=False)
    input_dim: int = Field(ge=1)
    seq_length: int = Field(ge=16)


class LSTMConfig(BaseModel):
    tick_encoder: LSTMEncoderConfig = Field(default_factory=lambda: LSTMEncoderConfig(hidden_dim=128, num_layers=2, input_dim=8, seq_length=512))
    bar_encoder: LSTMEncoderConfig = Field(default_factory=lambda: LSTMEncoderConfig(hidden_dim=256, num_layers=3, input_dim=12))
    cross_attention_heads: int = Field(default=8, ge=1)
    d_model: int = Field(default=512, ge=64)
    latent_dim: int = Field(default=512, ge=64)
    dropout: float = Field(default=0.1, ge=0, le=0.5)
    training_lr: float = Field(default=0.001, ge=1e-6, le=1.0)
    weight_decay: float = Field(default=1e-5, ge=0)
    beta_kl: float = Field(default=1e-4, ge=0)
    patience: int = Field(default=10, ge=1)


class XGBoostModelAConfig(BaseModel):
    feature_dim: int = Field(default=648, ge=1)
    n_estimators: int = Field(default=500, ge=1)
    max_depth: int = Field(default=6, ge=1, le=20)
    learning_rate: float = Field(default=0.05, ge=1e-4, le=1.0)
    subsample: float = Field(default=0.8, ge=0.1, le=1.0)
    colsample_bytree: float = Field(default=0.7, ge=0.1, le=1.0)
    min_child_weight: int = Field(default=5, ge=1)
    reg_alpha: float = Field(default=0.1, ge=0)
    reg_lambda: float = Field(default=1.5, ge=0)
    theta_oc: float = Field(default=0.85, ge=0, le=1.0)
    lambda_oc: float = Field(default=0.5, ge=0)
    lambda_nr: float = Field(default=0.3, ge=0)
    inference_timeout_ms: int = Field(default=5, ge=1)


class XGBoostModelBConfig(BaseModel):
    feature_dim: int = Field(default=560, ge=1)
    n_estimators: int = Field(default=600, ge=1)
    max_depth: int = Field(default=5, ge=1, le=20)
    learning_rate: float = Field(default=0.03, ge=1e-4, le=1.0)
    subsample: float = Field(default=0.75, ge=0.1, le=1.0)
    colsample_bytree: float = Field(default=0.65, ge=0.1, le=1.0)
    min_child_weight: int = Field(default=8, ge=1)
    reg_alpha: float = Field(default=0.2, ge=0)
    reg_lambda: float = Field(default=2.0, ge=0)
    fp_cost: float = Field(default=2.5, ge=0)
    fn_cost: float = Field(default=1.0, ge=0)
    optimal_threshold: float = Field(default=0.71, ge=0, le=1.0)
    inference_timeout_ms: int = Field(default=5, ge=1)


class ModelParamsConfig(BaseModel):
    lstm: LSTMConfig = Field(default_factory=LSTMConfig)
    model_a: XGBoostModelAConfig = Field(default_factory=XGBoostModelAConfig)
    model_b: XGBoostModelBConfig = Field(default_factory=XGBoostModelBConfig)
    vector_projection_dim: int = Field(default=256, ge=64)


# =============================================================================
# News Config
# =============================================================================

class NewsWeightsConfig(BaseModel):
    alpha: float = Field(default=0.4, ge=0, le=2.0)
    impact_low: float = Field(default=0.2, ge=0, le=1.0)
    impact_medium: float = Field(default=0.5, ge=0, le=1.0)
    impact_high: float = Field(default=1.0, ge=0, le=1.0)
    pre_news_window_seconds: int = Field(default=900, ge=60)
    news_window_seconds: int = Field(default=300, ge=30)
    post_news_lookback_seconds: int = Field(default=600, ge=60)
    surprise_z_threshold: float = Field(default=2.0, ge=0)
    pre_news_dampening: float = Field(default=0.3, ge=0, le=1.0)
    news_window_dampening: float = Field(default=0.1, ge=0, le=1.0)
    gamma_0: float = Field(default=0.3, ge=0, le=1.0)


# =============================================================================
# System Thresholds
# =============================================================================

class ThresholdPair(BaseModel):
    good: float
    warning: float
    critical: float


class SystemThresholdsConfig(BaseModel):
    ic: ThresholdPair = Field(default_factory=lambda: ThresholdPair(good=0.10, warning=0.05, critical=0.05))
    brier_score: ThresholdPair = Field(default_factory=lambda: ThresholdPair(good=0.20, warning=0.28, critical=0.28))
    ece: ThresholdPair = Field(default_factory=lambda: ThresholdPair(good=0.05, warning=0.08, critical=0.08))
    f1_hold: ThresholdPair = Field(default_factory=lambda: ThresholdPair(good=0.65, warning=0.50, critical=0.50))
    feature_drift_score: ThresholdPair = Field(default_factory=lambda: ThresholdPair(good=0.20, warning=0.20, critical=0.40))
    ipc_latency_ms: ThresholdPair = Field(default_factory=lambda: ThresholdPair(good=25, warning=50, critical=50))
    redis_memory_pct: ThresholdPair = Field(default_factory=lambda: ThresholdPair(good=60, warning=80, critical=80))


# =============================================================================
# Master Config Loader
# =============================================================================

class MasterConfig(BaseModel):
    system: SystemConfig = Field(default_factory=SystemConfig)
    killzones: KillzoneConfig = Field(default_factory=KillzoneConfig)
    model_params: ModelParamsConfig = Field(default_factory=ModelParamsConfig)
    news_weights: NewsWeightsConfig = Field(default_factory=NewsWeightsConfig)
    thresholds: SystemThresholdsConfig = Field(default_factory=SystemThresholdsConfig)

    @classmethod
    def from_yaml_files(
        cls,
        config_dir: str | Path = "config",
    ) -> MasterConfig:
        """Doc tat ca config tu YAML files va merge thanh MasterConfig."""
        import yaml

        config_dir = Path(config_dir)
        data: dict = {}

        # Doc tung file
        file_map = {
            "system": ["system.yaml", "killzones.yaml", "news_weights.yaml"],
            "model_params": ["model_params.yaml"],
            "thresholds": ["system.yaml"],
        }

        # system.yaml
        sys_path = config_dir / "system.yaml"
        if sys_path.exists():
            with open(sys_path) as f:
                sys_data = yaml.safe_load(f) or {}
                data["system"] = sys_data.get("system", sys_data)
                data["killzones"] = sys_data.get("killzones", {})
                data["news_weights"] = sys_data.get("news_weights", {})
                data["thresholds"] = sys_data.get("system_thresholds", {})

        # model_params.yaml
        mp_path = config_dir / "model_params.yaml"
        if mp_path.exists():
            with open(mp_path) as f:
                mp_data = yaml.safe_load(f) or {}
                data["model_params"] = mp_data

        return cls.model_validate(data)
