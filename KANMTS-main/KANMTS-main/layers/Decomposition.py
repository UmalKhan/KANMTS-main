import torch
import torch.nn as nn


class SeriesDecomposition(nn.Module):
    """
    Series Decomposition block that splits a time series into:
      - Trend component  (moving average)
      - Seasonal component (residual = original - trend)

    Inspired by Autoformer (Wu et al., 2021): https://arxiv.org/abs/2106.13008
    """

    def __init__(self, kernel_size: int = 25):
        """
        Args:
            kernel_size: Window size for the moving average (should be odd for symmetry).
        """
        super(SeriesDecomposition, self).__init__()
        # Use padding so output has the same length as input
        padding = kernel_size // 2
        self.avg_pool = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            count_include_pad=False,
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor of shape (B, T, N).
        Returns:
            seasonal: Seasonal component, shape (B, T, N).
            trend:    Trend component,    shape (B, T, N).
        """
        # avg_pool1d expects (B, C, L) — treat N as "channels", T as length
        x_t = x.permute(0, 2, 1)           # (B, N, T)
        trend = self.avg_pool(x_t)          # (B, N, T)

        # Trim or pad to ensure exactly the same length as input
        T = x.size(1)
        trend = trend[:, :, :T]

        trend = trend.permute(0, 2, 1)      # (B, T, N)
        seasonal = x - trend
        return seasonal, trend
