import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NewGELU(nn.Module):
    """
    GELU activation matching Google BERT / OpenAI GPT.
    Reference: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )


class KANLinear(nn.Module):
    """
    Kolmogorov-Arnold Network linear layer using B-spline basis functions.

    Improvements over original:
    ✅ Learnable per-input grid scaling (adaptive_scale) and shift (adaptive_bias)
       so the model can learn the optimal input range for each feature, rather
       than being hard-coded to [-1, 1].
    ✅ All original functionality (base_weight, spline_weight, scaler, update_grid,
       curve2coeff, regularization_loss) is preserved.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation=nn.SiLU,
        grid_eps: float = 0.02,
        grid_range=None,
    ):
        super(KANLinear, self).__init__()

        if grid_range is None:
            grid_range = [-1, 1]

        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # ── Fixed reference grid (used for spline basis initialisation) ───────
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]
        ).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        # ── Learnable weights ─────────────────────────────────────────────────
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))

        # ── NEW: Learnable adaptive input range per feature ───────────────────
        # Allows the model to rescale each input feature before computing splines,
        # effectively learning the optimal grid range automatically.
        self.adaptive_scale = nn.Parameter(torch.ones(in_features))
        self.adaptive_bias = nn.Parameter(torch.zeros(in_features))

        # ── Scalars ───────────────────────────────────────────────────────────
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    # ─────────────────────────────────────────────────────────────────────────

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5)
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order], noise
                )
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(
                    self.spline_scaler, a=math.sqrt(5) * self.scale_spline
                )

    # ─────────────────────────────────────────────────────────────────────────

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """Compute B-spline basis for input x of shape (batch, in_features)."""
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:]
            )
        assert bases.size() == (x.size(0), self.in_features, self.grid_size + self.spline_order)
        return bases.contiguous()

    # ─────────────────────────────────────────────────────────────────────────

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0
        )

    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (B, L, in_features) or (B, in_features).
        Returns:
            Output tensor with last dim = out_features.
        """
        original_shape = x.shape
        if x.dim() == 3:
            x = x.reshape(-1, x.size(-1))

        assert x.dim() == 2 and x.size(1) == self.in_features

        # ── Apply adaptive input rescaling (NEW) ──────────────────────────────
        x_scaled = x * self.adaptive_scale.unsqueeze(0) + self.adaptive_bias.unsqueeze(0)

        # ── Base (SiLU) linear path ───────────────────────────────────────────
        self.acts = self.base_activation(x_scaled)
        base_output = F.linear(self.base_activation(x_scaled), self.base_weight)

        # ── Spline path ───────────────────────────────────────────────────────
        spline_output = F.linear(
            self.b_splines(x_scaled).view(x_scaled.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )

        output = base_output + spline_output

        if len(original_shape) == 3:
            output = output.reshape(original_shape[0], original_shape[1], -1)
        return output

    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin: float = 0.01):
        """Adaptively update the spline grid based on actual data distribution."""
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x).permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight.permute(1, 2, 0)
        unreduced_spline_output = torch.bmm(splines, orig_coeff).permute(1, 0, 2)

        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device)
        ]
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(self.grid_size + 1, dtype=torch.float32, device=x.device).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.cat(
            [
                grid[:1] - uniform_step * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:] + uniform_step * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )
        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    # ─────────────────────────────────────────────────────────────────────────

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        assert result.size() == (self.out_features, self.in_features, self.grid_size + self.spline_order)
        return result.contiguous()

    # ─────────────────────────────────────────────────────────────────────────

    def regularization_loss(
        self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0
    ) -> torch.Tensor:
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / (regularization_loss_activation + 1e-8)
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )
