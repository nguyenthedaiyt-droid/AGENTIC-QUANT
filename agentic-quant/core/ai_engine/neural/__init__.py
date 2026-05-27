# =============================================================================
# AGENTIC-QUANT — Neural Engine (Phase 5: LSTM Autoencoder)
#
# Module nay implement LSTM Autoencoder cho hierarchical feature extraction.
# Gom 4 thanh phan chinh:
#   1. HierarchicalLSTMAE  - LSTM Autoencoder (encode -> z -> decode)
#   2. LSTMInferenceEngine  - Inference engine (<20ms CPU, Redis cache)
#   3. USVProjector         - Chieu z[512] + f_smc[64] + f_macro[12] -> e_USV[256]
#   4. EmbeddingConfig      - Dataclass config cho kich thuoc embedding
#
# Flow:
#   tick_seq (seq_len x 8 feats)
#     -> TickEncoder (BiLSTM 128x2)         -> h_tick[256]
#     -> BarEncoder (LSTM 256x3)             -> h_bar[768]
#     -> MTFEncoders (6 x LSTM 256x3)        -> h_mtf[6 x 512]
#     -> CrossTFAttention (8 heads, 512)     -> h_agg[512]
#     -> Projection (512+256 -> 512)          -> z[512]
#     -> Decoder (LSTM 512->256, 2 layers)   -> recon[seq_len x 8]
#
#   e_USV = LayerNorm(tanh(W_proj @ concat(z[512], f_smc[64], f_macro[12])))
#   e_USV = L2-normalized -> 256 dims
# =============================================================================

from __future__ import annotations

from core.ai_engine.neural.hierarchical_lstm_ae import (
    HierarchicalLSTMAE,
    AEOutput,
    EncoderOutput,
    TickEncoder,
    BarEncoder,
    MTFEncoders,
    CrossTFAttention,
    ProjectionLayer,
    DecoderNetwork,
)
from core.ai_engine.neural.lstm_inference import (
    LSTMInferenceEngine,
    encode_sync,
)
from core.ai_engine.neural.usv_projector import (
    USVProjector,
    TripletLoss,
)
from core.ai_engine.neural.config import (
    EmbeddingConfig,
)

__all__ = [
    # Model
    "HierarchicalLSTMAE",
    "AEOutput",
    "EncoderOutput",
    "TickEncoder",
    "BarEncoder",
    "MTFEncoders",
    "CrossTFAttention",
    "ProjectionLayer",
    "DecoderNetwork",
    # Inference
    "LSTMInferenceEngine",
    "encode_sync",
    # USV Projector
    "USVProjector",
    "TripletLoss",
    # Config
    "EmbeddingConfig",
]
