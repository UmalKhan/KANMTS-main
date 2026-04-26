import torch
import torch.nn as nn


class DropPath(nn.Module):
    """
    Stochastic Depth (Drop Path) regularization.
    Randomly drops entire sample paths during training to prevent overfitting.

    Reference: Huang et al. (2016) - Deep Networks with Stochastic Depth
    """

    def __init__(self, drop_prob: float = 0.0):
        """
        Args:
            drop_prob: Probability of dropping a path (0.0 = disabled).
        """
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x

        keep_prob = 1.0 - self.drop_prob
        # Create a random binary mask with shape (B, 1, 1, ...) for broadcasting
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)  # binarize
        # Scale to maintain expected value
        output = x / keep_prob * random_tensor
        return output

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.4f}"
