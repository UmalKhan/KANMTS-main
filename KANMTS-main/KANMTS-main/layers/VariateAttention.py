import torch
import torch.nn as nn


class VariateAttention(nn.Module):
    """
    Multi-head self-attention applied over the variate (channel) dimension.

    Each variate is treated as a "token" in the sequence dimension of the
    attention module — matching the iTransformer design (Liu et al., 2024):
    https://arxiv.org/abs/2310.06625

    This captures cross-variate dependencies that purely feed-forward
    KAN/MLP mixers cannot model.

    Input/Output shape: (B, N, D)
      B = batch size
      N = number of variates (channels)
      D = model dimension
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        """
        Args:
            d_model: Feature dimension per variate.
            n_heads: Number of attention heads.
            dropout: Dropout probability on attention weights.
        """
        super(VariateAttention, self).__init__()
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,   # expects (B, N, D)
        )
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, N, D).
        Returns:
            Tensor of shape (B, N, D) with variate-level attention applied.
        """
        attn_out, _ = self.attn(x, x, x)           # self-attention over N variates
        return self.norm(x + self.drop(attn_out))   # residual + layer norm
