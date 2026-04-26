import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN).
    Normalizes per-channel statistics and reverses them after prediction.
    Reference: Kim et al. (2022) - https://openreview.net/forum?id=cGDAkQo1C0p
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        """
        Args:
            num_features: Number of channels/variates (N).
            eps: Stability epsilon.
            affine: If True, learns per-channel scale and shift after normalization.
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, N).
            mode: 'norm' to normalize; 'denorm' to reverse normalization.
        Returns:
            Normalized or de-normalized tensor of same shape.
        """
        if mode == 'norm':
            self._compute_stats(x)
            x = (x - self.mean) / (self.stdev + self.eps)
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self.stdev + self.mean
        else:
            raise ValueError(f"RevIN mode must be 'norm' or 'denorm', got '{mode}'")
        return x

    def _compute_stats(self, x: torch.Tensor):
        """Compute and cache per-channel mean and std from input."""
        # x: (B, T, N) — reduce over the time dimension
        self.mean = x.mean(dim=1, keepdim=True).detach()   # (B, 1, N)
        self.stdev = torch.sqrt(
            x.var(dim=1, keepdim=True, unbiased=False) + self.eps
        ).detach()  # (B, 1, N)
