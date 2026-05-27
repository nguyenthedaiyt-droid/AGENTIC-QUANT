# =============================================================================
# AGENTIC-QUANT — Embedding Config cho Phase 5: LSTM Autoencoder
#
# Cau hinh kich thuoc cac embedding trong pipeline:
#   - lstm_hidden_tick: 128  (hidden dim cua TickEncoder BiLSTM)
#   - lstm_hidden_bar:  256  (hidden dim cua BarEncoder LSTM)
#   - lstm_latent_dim:  512  (kich thuoc z latent)
#   - usv_dim:          256  (kich thuoc e_USV)
#   - triplet_margin:    1.0 (margin cho TripletLoss)
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingConfig:
    """Cau hinh kich thuoc embedding cho toan bo neural pipeline.

    Tat ca cac gia tri duoc freeze (immutable) de dam bao consistency
    giua cac module (encoder, decoder, projector, loss function).

    Attributes:
        lstm_hidden_tick: Hidden dimension cua TickEncoder (BiLSTM, moi chieu 128)
        lstm_hidden_bar:  Hidden dimension cua BarEncoder (LSTM, 3 layers)
        lstm_latent_dim:  Kich thuoc latent vector z (output cua encoder)
        usv_dim:          Kich thuoc e_USV sau projection (output cua USVProjector)
        triplet_margin:   Margin cho TripletLoss (online mining)
    """

    lstm_hidden_tick: int = 128
    lstm_hidden_bar: int = 256
    lstm_latent_dim: int = 512
    usv_dim: int = 256
    triplet_margin: float = 1.0
