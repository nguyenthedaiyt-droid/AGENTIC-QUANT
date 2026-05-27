# =============================================================================
# AGENTIC-QUANT — USV Projector
# Phase 5: LSTM Autoencoder -> USV Embedding (e_USV)
#
# Chieu latent vector z[512] + SMC features f_smc[64] + macro features
# f_macro[12] thanh embedded USV (e_USV) voi 256 dims.
#
# Formula:
#   e_USV = LayerNorm(tanh(W_proj @ concat(z[512], f_smc[64], f_macro[12])))
#   e_USV = L2_normalize(e_USV)
#
#   W_proj: [256 x 588] projection matrix
#   Total input: 512 + 64 + 12 = 588
#
# Learnable via TripletLoss (margin=1.0, online mining).
# Output: np.ndarray[256] L2-normalized
# =============================================================================

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.ai_engine.neural.config import EmbeddingConfig


# =============================================================================
# USVProjector
# =============================================================================

class USVProjector(nn.Module):
    """USV Projector: chieu z[512] + f_smc[64] + f_macro[12] -> e_USV[256].

    Day la projection network don gian, co the learn via TripletLoss.

    Architecture:
      concat(z[512], f_smc[64], f_macro[12])  -> [588]
      -> Linear(588, 256)                       -> W_proj [256 x 588]
      -> tanh
      -> LayerNorm(256)
      -> L2 normalize                           -> e_USV[256]

    Output:
      e_USV ∈ ℝ^256, L2-normalized, eps=1e-8
    """

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        """Khoi tao USVProjector.

        Args:
            config: EmbeddingConfig. Neu None, dung defaults.
        """
        super().__init__()

        cfg = config or EmbeddingConfig()

        # Kich thuoc cac component
        self.z_dim: int = cfg.lstm_latent_dim       # 512
        self.f_smc_dim: int = 64                     # SMC features
        self.f_macro_dim: int = 12                   # Macro features
        self.usv_dim: int = cfg.usv_dim              # 256
        self.total_input_dim: int = self.z_dim + self.f_smc_dim + self.f_macro_dim  # 588

        # Projection matrix W_proj: [256 x 588]
        self.w_proj = nn.Linear(self.total_input_dim, self.usv_dim, bias=False)

        # LayerNorm sau tanh, truoc L2 normalize
        self.layer_norm = nn.LayerNorm(self.usv_dim)

        # Epsilon cho numerical stability
        self.eps: float = 1e-8

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Khoi tao W_proj voi Xavier uniform."""
        nn.init.xavier_uniform_(self.w_proj.weight, gain=1.0)

    # =========================================================================
    # Forward
    # =========================================================================

    def forward(
        self,
        z: torch.Tensor,
        f_smc: torch.Tensor | None = None,
        f_macro: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass: project concatenated features -> e_USV.

        Args:
            z: [batch, 512] latent vector tu LSTM Autoencoder
            f_smc: [batch, 64] SMC features (structure + FVG + liquidity).
                   Neu None, dung zeros.
            f_macro: [batch, 12] macro features (news, rates, etc.).
                     Neu None, dung zeros.

        Returns:
            e_usv: [batch, 256] L2-normalized embedding
        """
        batch_size = z.size(0)

        # Default zero features neu khong duoc cung cap
        if f_smc is None:
            f_smc = torch.zeros(
                batch_size, self.f_smc_dim, device=z.device, dtype=z.dtype
            )
        if f_macro is None:
            f_macro = torch.zeros(
                batch_size, self.f_macro_dim, device=z.device, dtype=z.dtype
            )

        # Concatenate: [512 + 64 + 12] = [588]
        x = torch.cat([z, f_smc, f_macro], dim=-1)  # [batch, 588]

        # Projection: W_proj @ x -> [batch, 256]
        x = self.w_proj(x)  # [batch, 256]

        # tanh activation
        x = torch.tanh(x)  # [batch, 256]

        # LayerNorm
        x = self.layer_norm(x)  # [batch, 256]

        # L2 normalize
        e_usv = F.normalize(x, p=2, dim=-1, eps=self.eps)  # [batch, 256]

        return e_usv

    # =========================================================================
    # Inference helper: tra ve numpy array
    # =========================================================================

    @torch.no_grad()
    def to_e_usv(
        self,
        z: torch.Tensor | list[float] | "np.ndarray",  # noqa: F821
        f_smc: torch.Tensor | list[float] | "np.ndarray | None" = None,  # noqa: F821
        f_macro: torch.Tensor | list[float] | "np.ndarray | None" = None,  # noqa: F821
    ) -> "np.ndarray":  # noqa: F821
        """Inference-only: chieu inputs -> e_USV numpy array.

        Args:
            z: [512] latent vector (tensor, list, hoac numpy)
            f_smc: [64] SMC features (optional)
            f_macro: [12] macro features (optional)

        Returns:
            e_usv: np.ndarray [256] float32, L2-normalized
        """
        import numpy as np

        # Chuyen ve tensor
        if not isinstance(z, torch.Tensor):
            z = torch.tensor(np.ascontiguousarray(z, dtype=np.float32))
        if f_smc is not None and not isinstance(f_smc, torch.Tensor):
            f_smc = torch.tensor(np.ascontiguousarray(f_smc, dtype=np.float32))
        if f_macro is not None and not isinstance(f_macro, torch.Tensor):
            f_macro = torch.tensor(np.ascontiguousarray(f_macro, dtype=np.float32))

        # Dam bao batch dim
        if z.dim() == 1:
            z = z.unsqueeze(0)         # [1, 512]
            if f_smc is not None:
                f_smc = f_smc.unsqueeze(0)
            if f_macro is not None:
                f_macro = f_macro.unsqueeze(0)

        self.eval()
        e_usv = self.forward(z, f_smc, f_macro)  # [1, 256]

        return e_usv.squeeze(0).cpu().numpy().astype(np.float32)  # [256]


# =============================================================================
# TripletLoss (Online Mining)
# =============================================================================

class TripletLoss(nn.Module):
    """TripletLoss voi online mining (margin=1.0).

    Tu dong chon hard/triplet samples trong batch.

    Args:
        margin: Khoang cach margin (default: 1.0)
    """

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """Tinh TripletLoss.

        Args:
            anchor: [batch, 256] embeddings
            positive: [batch, 256] embeddings (cung class voi anchor)
            negative: [batch, 256] embeddings (khac class voi anchor)

        Returns:
            loss: Scalar tensor
        """
        # Euclidean distance
        pos_dist = F.pairwise_distance(anchor, positive, p=2)   # [batch]
        neg_dist = F.pairwise_distance(anchor, negative, p=2)   # [batch]

        # Triplet loss: max(0, d_pos - d_neg + margin)
        losses = F.relu(pos_dist - neg_dist + self.margin)  # [batch]

        return losses.mean()


# =============================================================================
# Parameter Counting
# =============================================================================

def count_projector_params(projector: USVProjector) -> dict[str, int]:
    """Dem parameters cua USVProjector."""
    total = sum(p.numel() for p in projector.parameters())
    trainable = sum(p.numel() for p in projector.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


# =============================================================================
# Unit-test
# =============================================================================

if __name__ == "__main__":
    import numpy as np

    print("Testing USVProjector...")

    # Khoi tao
    projector = USVProjector()
    projector.eval()

    params = count_projector_params(projector)
    print(f"Projector params: {params['total']:,}")

    # Test forward
    batch_size = 2
    z = torch.randn(batch_size, 512)
    f_smc = torch.randn(batch_size, 64)
    f_macro = torch.randn(batch_size, 12)

    with torch.no_grad():
        e_usv = projector(z, f_smc, f_macro)

    print(f"e_usv shape: {e_usv.shape}")              # [2, 256]
    norms = e_usv.norm(p=2, dim=-1)
    print(f"L2 norms: {norms}")                        # [1.0, 1.0]
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6), \
        "e_USV chua L2-normalized!"
    print("✅ L2 normalized OK")

    # Test to_e_usv (numpy output)
    z_np = np.random.randn(512).astype(np.float32)
    f_smc_np = np.random.randn(64).astype(np.float32)
    f_macro_np = np.random.randn(12).astype(np.float32)

    e_usv_np = projector.to_e_usv(z_np, f_smc_np, f_macro_np)
    print(f"e_usv_np shape: {e_usv_np.shape}")        # (256,)
    print(f"e_usv_np L2 norm: {np.linalg.norm(e_usv_np):.6f}")  # ~1.0

    # Test TripletLoss
    triplet_loss = TripletLoss(margin=1.0)
    anchor = torch.randn(4, 256)
    positive = anchor + 0.1 * torch.randn(4, 256)  # gan anchor
    negative = torch.randn(4, 256)                   # xa anchor
    loss = triplet_loss(anchor, positive, negative)
    print(f"TripletLoss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss phai > 0"

    print("\n✅ USVProjector smoke test passed!")
