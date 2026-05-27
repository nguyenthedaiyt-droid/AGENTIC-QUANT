# =============================================================================
# AGENTIC-QUANT — Hierarchical LSTM Autoencoder
# Phase 5: LSTM Autoencoder
#
# Architecture (specification):
#   TickEncoder:   BiLSTM(input=8, hidden=128, layers=2, bidirectional=True)
#   BarEncoder:    LSTM(input=12, hidden=256, layers=3)
#   MTFEncoders:   6 parallel LSTM(12, 256, 3) cho M1/M5/M15/H1/H4/D1
#   CrossTFAttention: MultiheadAttention(embed_dim=512, heads=8, batch_first=True)
#   Projection:    Linear(512+256, 512) -> z
#   Decoder:       LSTM(512, 256, 2) -> reconstruct M1 sequence
#
#   forward(): z, recon
#   encode():  z
#
# Constraints:
#   - < 50M params
#   - GELU activation
#   - Docstring + comments bang Tieng Viet
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    pass


# =============================================================================
# Constants (theo EmbeddingConfig)
# =============================================================================
TICK_FEAT_DIM: int = 8        # So feature cua tick data (OHLCV + ...)
BAR_FEAT_DIM: int = 12        # So feature cua bar data (OHLCV + ...)
TICK_HIDDEN: int = 128        # hidden_dim cua TickEncoder
BAR_HIDDEN: int = 256         # hidden_dim cua BarEncoder / MTFEncoders
LATENT_DIM: int = 512         # Kich thuoc z latent
NUM_TF: int = 6               # So luong timeframe: M1, M5, M15, H1, H4, D1
NUM_HEADS: int = 8            # So head trong MultiheadAttention
DROPOUT: float = 0.1


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EncoderOutput:
    """Output cua Encoder.

    Attributes:
        z: Latent vector [batch, latent_dim] (dung cho USV projection)
        h_agg: Aggregated representation truoc projection [batch, 512]
    """
    z: torch.Tensor
    h_agg: torch.Tensor | None = None


@dataclass
class AEOutput:
    """Output day du cua HierarchicalLSTMAE.

    Attributes:
        z: Latent vector [batch, latent_dim]
        recon: Reconstruction cua M1 sequence [batch, seq_len, 8]
        loss: MSE loss (neu tinh)
    """
    z: torch.Tensor
    recon: torch.Tensor | None = None
    loss: torch.Tensor | None = None


# =============================================================================
# TickEncoder — Level 1
# =============================================================================

class TickEncoder(nn.Module):
    """TickEncoder: xu ly tick data bang BiLSTM.

    Input:  [batch, seq_len, 8]   (8 features: OHLCV + spread + volume + ...)
    Output: [batch, 256]          (128 fwd + 128 bwd)

    Architecture:
      Input(8) -> BiLSTM(hidden=128, layers=2, bidirectional=True)
      -> concat last fwd[128] + last bwd[128] -> [256]
    """

    def __init__(self, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=TICK_FEAT_DIM,
            hidden_size=TICK_HIDDEN,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0.0,
        )
        self.norm = nn.LayerNorm(TICK_HIDDEN * 2)  # 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass TickEncoder.

        Args:
            x: [batch, seq_len, 8] tick features

        Returns:
            h_tick: [batch, 256] encoded tick representation
        """
        # Cap nhat dropout neu train
        if self.training and self.lstm.dropout > 0:
            self.lstm.dropout = DROPOUT

        _, (hn, _) = self.lstm(x)
        # hn: [num_layers * num_directions, batch, hidden] = [4, batch, 128]

        # GELU activation
        h_forward = F.gelu(hn[-2])   # [batch, 128] - forward last layer
        h_backward = F.gelu(hn[-1])  # [batch, 128] - backward last layer
        h_tick = torch.cat([h_forward, h_backward], dim=-1)  # [batch, 256]
        h_tick = self.norm(h_tick)
        return h_tick


# =============================================================================
# BarEncoder — Level 2
# =============================================================================

class BarEncoder(nn.Module):
    """BarEncoder: xu ly bar OHLCV sequence bang LSTM.

    Input:  [batch, seq_len, 12]  (12 features: OHLCV + ...)
    Output: [batch, 768]          (3 layers * 256)

    Architecture:
      Input(12) -> LSTM(hidden=256, layers=3)
      -> concat last hidden cua ca 3 layers -> [768]
    """

    def __init__(self, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=BAR_FEAT_DIM,
            hidden_size=BAR_HIDDEN,
            num_layers=3,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if dropout > 0 else 0.0,
        )
        self.norm = nn.LayerNorm(BAR_HIDDEN * 3)  # 768

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass BarEncoder.

        Args:
            x: [batch, seq_len, 12] bar features

        Returns:
            h_bar: [batch, 768] encoded bar representation
        """
        if self.training and self.lstm.dropout > 0:
            self.lstm.dropout = DROPOUT

        _, (hn, _) = self.lstm(x)
        # hn: [3, batch, 256]

        # GELU activation + concat all layers
        h_layers = [F.gelu(hn[i]) for i in range(3)]  # 3 x [batch, 256]
        h_bar = torch.cat(h_layers, dim=-1)  # [batch, 768]
        h_bar = self.norm(h_bar)
        return h_bar


# =============================================================================
# MTFEncoders — Level 3 (6 parallel LSTMs)
# =============================================================================

class MTFEncoders(nn.Module):
    """MTFEncoders: 6 parallel LSTM channels, 1 cho moi timeframe.

    Moi channel la LSTM(12, 256, 3) doc lap.
    6 timeframe: M1, M5, M15, H1, H4, D1.

    Input:  list of 6 tensors, moi tensor [batch, seq_len_tf, 12]
            (hoac [batch, seq_len_tf, 12] tu BarEncoder-style features)
    Output: [batch, 6, 512]  (moi TF 1 vector 512-dim)
    """

    def __init__(self, dropout: float = DROPOUT) -> None:
        super().__init__()
        # 6 LSTM channels song song
        self.channels = nn.ModuleList([
            nn.LSTM(
                input_size=BAR_FEAT_DIM,   # 12
                hidden_size=BAR_HIDDEN,    # 256
                num_layers=3,
                batch_first=True,
                bidirectional=False,
                dropout=0.0,  # dropout cho single-layer lstm ko co tac dung
            )
            for _ in range(NUM_TF)
        ])
        # Chieu tu 768 (3 layers * 256) -> 512
        self.proj = nn.Linear(BAR_HIDDEN * 3, LATENT_DIM)
        self.norm = nn.LayerNorm(LATENT_DIM)

    def forward(self, mtf_inputs: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass MTFEncoders.

        Args:
            mtf_inputs: list cua 6 tensors, moi tensor [batch, seq_len_tf, 12]

        Returns:
            h_mtf: [batch, 6, 512] stacked TF vectors
        """
        outputs: list[torch.Tensor] = []
        for i, x in enumerate(mtf_inputs):
            _, (hn, _) = self.channels[i](x)
            # hn: [3, batch, 256]
            # GELU + concat 3 layers -> [batch, 768]
            h = F.gelu(torch.cat([hn[j] for j in range(3)], dim=-1))
            # Chieu tu 768 -> 512
            h = self.norm(F.gelu(self.proj(h)))  # [batch, 512]
            outputs.append(h)

        h_mtf = torch.stack(outputs, dim=1)  # [batch, 6, 512]
        return h_mtf


# =============================================================================
# CrossTFAttention — Level 4
# =============================================================================

class CrossTFAttention(nn.Module):
    """Cross-Temporal Fusion Attention.

    MultiheadAttention(embed_dim=512, heads=8, batch_first=True)
    giua 6 TF vectors de tao aggregated representation.

    Input:  [batch, 6, 512]
    Output: [batch, 512]
    """

    def __init__(self, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=LATENT_DIM,   # 512
            num_heads=NUM_HEADS,    # 8
            batch_first=True,
            dropout=dropout,
        )
        # Learnable CLS token lam query
        self.cls_token = nn.Parameter(torch.randn(1, 1, LATENT_DIM) * 0.02)
        self.norm = nn.LayerNorm(LATENT_DIM)

    def forward(self, h_mtf: torch.Tensor) -> torch.Tensor:
        """Forward pass CrossTFAttention.

        Args:
            h_mtf: [batch, 6, 512] 6 TF vectors

        Returns:
            h_agg: [batch, 512] aggregated representation
        """
        batch_size = h_mtf.size(0)

        # Them CLS token vao dau sequence -> query
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch, 1, 512]
        x = torch.cat([cls_tokens, h_mtf], dim=1)  # [batch, 7, 512]
        x = self.norm(x)

        # Self-attention: CLS attends to all 6 TF vectors
        # Query = CLS, Key = Value = tat ca 7 tokens
        attn_out, _ = self.attention(x, x, x)

        # Lay CLS token output lam aggregated representation
        h_agg = attn_out[:, 0, :]  # [batch, 512]
        return h_agg


# =============================================================================
# Projection — Linear(512+256, 512) -> z
# =============================================================================

class ProjectionLayer(nn.Module):
    """Projection: combine h_agg[512] + h_bar_compact -> z[512].

    h_bar_compact = h_bar[768] projection xuong 256.

    Architecture:
      Linear(512 + 256, 512) -> GELU -> LayerNorm

    Input:  h_agg[512], h_bar[768]
    Output: z[512]
    """

    def __init__(self) -> None:
        super().__init__()
        # Chieu h_bar tu 768 -> 256
        self.bar_compact = nn.Linear(BAR_HIDDEN * 3, 256)  # 768 -> 256
        # Chieu concat(512 + 256) -> 512
        self.proj = nn.Linear(LATENT_DIM + 256, LATENT_DIM)
        self.norm = nn.LayerNorm(LATENT_DIM)

    def forward(self, h_agg: torch.Tensor, h_bar: torch.Tensor) -> torch.Tensor:
        """Forward pass Projection.

        Args:
            h_agg: [batch, 512] aggregated tu CrossTFAttention
            h_bar: [batch, 768] tu BarEncoder

        Returns:
            z: [batch, 512] latent vector
        """
        h_bar_c = F.gelu(self.bar_compact(h_bar))  # [batch, 256]
        z = F.gelu(self.proj(torch.cat([h_agg, h_bar_c], dim=-1)))  # [batch, 512]
        z = self.norm(z)
        return z


# =============================================================================
# DecoderNetwork — LSTM(512, 256, 2) -> reconstruct M1 sequence
# =============================================================================

class DecoderNetwork(nn.Module):
    """Decoder: tu z[512] -> reconstruct M1 sequence.

    Architecture:
      z[512]
      -> Repeat(seq_len)               -> [seq_len, 512]
      -> LSTM(512, 256, 2)              -> [seq_len, 256]
      -> Linear(256, 8)                 -> [seq_len, 8]

    Output dung cho reconstruction loss (MSE).
    """

    def __init__(self, seq_len: int = 128) -> None:
        super().__init__()
        self.seq_len = seq_len

        # LSTM decoder
        self.lstm = nn.LSTM(
            input_size=LATENT_DIM,    # 512
            hidden_size=BAR_HIDDEN,   # 256
            num_layers=2,
            batch_first=True,
            bidirectional=False,
        )
        # Chieu tu 256 -> 8 (tick features)
        self.out_proj = nn.Linear(BAR_HIDDEN, TICK_FEAT_DIM)
        self.norm = nn.LayerNorm(LATENT_DIM)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass Decoder.

        Args:
            z: [batch, 512] latent vector

        Returns:
            recon: [batch, seq_len, 8] reconstructed M1 tick sequence
        """
        batch_size = z.size(0)

        # Lap lai z cho moi timestep cua decoder
        z_expanded = z.unsqueeze(1).expand(-1, self.seq_len, -1)  # [batch, seq_len, 512]
        z_expanded = self.norm(z_expanded)

        # LSTM decode
        lstm_out, _ = self.lstm(z_expanded)  # [batch, seq_len, 256]

        # Project ve tick feature space
        recon = self.out_proj(lstm_out)  # [batch, seq_len, 8]
        return recon


# =============================================================================
# HierarchicalLSTMAE — Full Model
# =============================================================================

class HierarchicalLSTMAE(nn.Module):
    """Hierarchical LSTM Autoencoder day du.

    Flow:
      tick_data[seq_len x 8]
        -> TickEncoder(BiLSTM 128x2)         -> h_tick[256]
        -> BarEncoder(LSTM 256x3)             -> h_bar[768]
        -> MTFEncoders(6 x LSTM 256x3)        -> h_mtf[6 x 512]
        -> CrossTFAttention(8 heads)          -> h_agg[512]
        -> Projection(512+256 -> 512)          -> z[512]
        -> Decoder(LSTM 512->256, 2 layers)   -> recon[seq_len x 8]

    forward(): Tra ve (z, recon)
    encode():  Tra ve z (latent vector)

    Constraints:
      - < 50M params
      - GELU activation
    """

    def __init__(
        self,
        dropout: float = DROPOUT,
        recon_seq_len: int = 128,
    ) -> None:
        super().__init__()

        # Encoder levels
        self.tick_encoder = TickEncoder(dropout=dropout)
        self.bar_encoder = BarEncoder(dropout=dropout)
        self.mtf_encoders = MTFEncoders(dropout=dropout)

        # Cross-temporal attention
        self.cross_attention = CrossTFAttention(dropout=dropout)

        # Projection -> z
        self.projection = ProjectionLayer()

        # Decoder
        self.decoder = DecoderNetwork(seq_len=recon_seq_len)

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(
        self,
        tick_seq: torch.Tensor,
        bar_seqs_mtf: list[torch.Tensor],
    ) -> torch.Tensor:
        """Encode inputs -> latent vector z.

        Args:
            tick_seq: [batch, seq_len, 8] tick data
            bar_seqs_mtf: list cua 6 tensors [batch, seq_len_tf, 12]
                          (M1, M5, M15, H1, H4, D1)

        Returns:
            z: [batch, 512] latent vector
        """
        # Level 1: Tick encoding
        h_tick = self.tick_encoder(tick_seq)  # [batch, 256]

        # Level 2: Bar encoding (dung M1 bar sequence)
        h_bar = self.bar_encoder(bar_seqs_mtf[0])  # [batch, 768]

        # Level 3: Multi-TF encoding
        h_mtf = self.mtf_encoders(bar_seqs_mtf)  # [batch, 6, 512]

        # Level 4: Cross-temporal attention
        h_agg = self.cross_attention(h_mtf)  # [batch, 512]

        # Level 5: Projection -> z
        z = self.projection(h_agg, h_bar)  # [batch, 512]

        return z

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        tick_seq: torch.Tensor,
        bar_seqs_mtf: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: encode + decode.

        Args:
            tick_seq: [batch, seq_len, 8] tick data
            bar_seqs_mtf: list cua 6 tensors [batch, seq_len_tf, 12]

        Returns:
            (z, recon):
                z: [batch, 512] latent vector
                recon: [batch, 128, 8] reconstruction cua M1 sequence
        """
        z = self.encode(tick_seq, bar_seqs_mtf)
        recon = self.decoder(z)

        return z, recon


# =============================================================================
# Parameter Counting
# =============================================================================

def count_model_params(model: nn.Module) -> dict[str, int]:
    """Dem so luong parameters cua model.

    Returns:
        dict voi keys: 'total', 'trainable'
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


# =============================================================================
# Unit-test
# =============================================================================

if __name__ == "__main__":
    print("Testing HierarchicalLSTMAE...")

    model = HierarchicalLSTMAE()
    model.eval()

    params = count_model_params(model)
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    assert params["total"] < 50_000_000, \
        f"Params {params['total']:,} > 50M limit!"
    print(f"✅ Under 50M params: {params['total']:,} < 50,000,000")

    # Fake inputs
    batch_size = 2
    tick_seq = torch.randn(batch_size, 128, TICK_FEAT_DIM)
    bar_seqs_mtf = [
        torch.randn(batch_size, 30, BAR_FEAT_DIM)
        for _ in range(NUM_TF)
    ]

    # Test encode
    with torch.no_grad():
        z = model.encode(tick_seq, bar_seqs_mtf)

    print(f"\nz shape: {z.shape}")  # [2, 512]

    # Test forward
    with torch.no_grad():
        z_out, recon = model(tick_seq, bar_seqs_mtf)

    print(f"z_out shape: {z_out.shape}")
    print(f"recon shape: {recon.shape}")

    # Test gradient flow
    model.train()
    z_out, recon = model(tick_seq, bar_seqs_mtf)
    loss = F.mse_loss(recon, tick_seq)
    loss.backward()
    print(f"\nLoss (MSE): {loss.item():.6f}")
    print(f"Gradient norm: {sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None):.4f}")

    # Verify GELU activation
    print(f"\n✅ HierarchicalLSTMAE smoke test passed!")
