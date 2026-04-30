import math

from layers.Embed import DataEmbedding_inverted
from layers.KANLinear import KANLinear
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.VariateAttention import VariateAttention
from layers.Decomposition import SeriesDecomposition
from layers.RevIN import RevIN

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Activation
# ─────────────────────────────────────────────────────────────────────────────

class NewGELU(nn.Module):
    """
    GELU activation matching the Google BERT / OpenAI GPT implementation.
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


# ─────────────────────────────────────────────────────────────────────────────
# KAN Mixing Blocks  (BUG-FIXED: LayerNorm axis + residual placement)
# ─────────────────────────────────────────────────────────────────────────────

class TokenMixingKAN(nn.Module):
    """
    KAN-based mixing along the TIME (token) dimension.

    FIX applied vs. original:
      1. LayerNorm is now applied over the last dim (channel) only — the
         original used LayerNorm([n_tokens, n_channel]) on a permuted tensor
         whose last two dims were swapped, causing incorrect normalisation.
      2. The residual `X + z` is only done HERE (inside the block).
         The caller (Mixer2dTriUKAN) must NOT add another residual on top,
         which was the double-residual bug.
    """
    def __init__(self, n_tokens: int, n_channel: int, n_hidden: int, dropout: float = 0.1):
        super().__init__()
        # n_tokens = d_core (128), n_channel = variates (11)
        # Normalize over the d_core feature dimension (last dim)
        self.layer_norm = nn.LayerNorm(n_tokens)
        self.kan1 = KANLinear(n_tokens, n_hidden)
        self.kan2 = KANLinear(n_hidden, n_tokens)
        self.drop = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, n_channel, n_tokens) = (B, 11, 128)
        z = self.layer_norm(X)              # (B, 11, 128) — correct axis
        z = self.kan1(z)                    # (B, 11, n_hidden)
        z = self.drop(z)
        z = self.kan2(z)                    # (B, 11, 128)
        return X + z                        # single residual — caller does NOT add again


class ChannelMixingKAN(nn.Module):
    """
    KAN-based mixing along the VARIATE (channel) dimension.
    """
    def __init__(self, n_tokens: int, n_channel: int, n_hidden: int, dropout: float = 0.1):
        super().__init__()
        # Normalize over the d_core feature dimension (last dim)
        self.layer_norm = nn.LayerNorm(n_tokens)
        self.kan1 = KANLinear(n_channel, n_hidden)
        self.kan2 = KANLinear(n_hidden, n_channel)
        self.drop = nn.Dropout(dropout)

    def forward(self, U: torch.Tensor) -> torch.Tensor:
        # U: (B, n_channel, n_tokens) = (B, 11, 128)
        z = self.layer_norm(U)              # (B, 11, 128)
        z = z.permute(0, 2, 1)             # (B, 128, 11) — mix across variates
        z = self.kan1(z)                    # (B, 128, n_hidden)
        z = self.drop(z)
        z = self.kan2(z)                    # (B, 128, 11)
        z = z.permute(0, 2, 1)             # (B, 11, 128)
        return U + z                        # single residual


# ─────────────────────────────────────────────────────────────────────────────
# MLP Mixer Fallback Blocks
# ─────────────────────────────────────────────────────────────────────────────

class MixerBlock1(nn.Module):
    """MLP token-mixing block (operates along the time/token dimension)."""

    def __init__(self, time_steps: int, channels: int, hidden_dim: int, dropout: float = 0.0):
        super(MixerBlock1, self).__init__()
        self.LN_1 = nn.LayerNorm(time_steps)      # normalize over feature dim
        self.dense_1 = nn.Linear(time_steps, hidden_dim)
        self.act = nn.GELU()
        self.dense_2 = nn.Linear(hidden_dim, time_steps)
        self.drop = nn.Dropout(dropout)

    def forward(self, U: torch.Tensor) -> torch.Tensor:
        # U: (B, channels, time_steps)
        x = self.LN_1(U)                    # (B, channels, time_steps)
        x = self.dense_1(x)                 # (B, channels, hidden_dim)
        x = self.act(x)
        x = self.drop(x)
        x = self.dense_2(x)                 # (B, channels, time_steps)
        x = self.drop(x)
        return U + x


class MixerBlock2(nn.Module):
    """MLP variate-mixing block (operates along the variate/channel dimension)."""

    def __init__(self, time_steps: int, channels: int, hidden_dim: int, dropout: float = 0.0):
        super(MixerBlock2, self).__init__()
        self.LN_2 = nn.LayerNorm(time_steps)
        self.dense_1 = nn.Linear(channels, hidden_dim)
        self.act = nn.GELU()
        self.dense_2 = nn.Linear(hidden_dim, channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, K: torch.Tensor) -> torch.Tensor:
        # K: (B, channels, time_steps)
        x = self.LN_2(K)                    # (B, channels, time_steps)
        x = x.permute(0, 2, 1)             # (B, time_steps, channels)
        x = self.dense_1(x)                 # (B, time_steps, hidden_dim)
        x = self.act(x)
        x = self.drop(x)
        x = self.dense_2(x)                 # (B, time_steps, channels)
        x = self.drop(x)
        x = x.permute(0, 2, 1)             # (B, channels, time_steps)
        return K + x


# ─────────────────────────────────────────────────────────────────────────────
# Core KAN Mixer  (BUG-FIXED: removed double residual)
# ─────────────────────────────────────────────────────────────────────────────

class STARKAN(nn.Module):
    """
    STar Aggregate-Redistribute Module using KAN.
    Aggregates global channel context and redistributes it back to individual channels.
    """
    def __init__(self, d_series, d_core, grid_size=5):
        super(STARKAN, self).__init__()
        self.gen1 = KANLinear(d_series, d_series, grid_size=grid_size)
        self.gen2 = KANLinear(d_series, d_core, grid_size=grid_size)
        self.gen3 = KANLinear(d_series + d_core, d_series, grid_size=grid_size)
        self.gen4 = KANLinear(d_series, d_series, grid_size=grid_size)

    def forward(self, x, *args, **kwargs):
        # x: (B, N, D)
        B, N, D = x.shape
        
        # ── Aggregate ────────────────────────────────────────────────────────
        combined = self.gen1(x)
        combined = self.gen2(combined) # (B, N, D_core)
        
        # Global context pooling (stochastic in training, weighted in eval)
        if self.training:
            ratio = F.softmax(combined, dim=1) # (B, N, D_core)
            # Simple mean pooling is often more stable for KAN than stochastic multinomial
            global_context = torch.mean(combined, dim=1, keepdim=True) # (B, 1, D_core)
        else:
            weight = F.softmax(combined, dim=1)
            global_context = torch.sum(combined * weight, dim=1, keepdim=True) # (B, 1, D_core)
            
        global_context = global_context.repeat(1, N, 1) # (B, N, D_core)
        
        # ── Redistribute ─────────────────────────────────────────────────────
        cat_feat = torch.cat([x, global_context], dim=-1) # (B, N, D + D_core)
        out = self.gen3(cat_feat)
        out = self.gen4(out)
        return x + out # Residual connection


class Mixer2dTriUKAN(nn.Module):
    """
    2-D TriU KAN Mixer block — v4 (STAR-KAN).
    """

    def __init__(
        self,
        time_steps: int,
        channels: int,
        d_core: int,
        grid_size: int,
        hidden_dim: int,
        dropout: float = 0.1,
        n_heads: int = 4,
    ):
        super(Mixer2dTriUKAN, self).__init__()

        self.proj_in = nn.Linear(time_steps, time_steps, bias=False)
        self.kan_bottleneck = KANLinear(time_steps, d_core, grid_size=grid_size)
        self.act = NewGELU()
        self.drop_in = nn.Dropout(dropout)

        self.TokenMixingKAN = TokenMixingKAN(d_core, channels, hidden_dim, dropout=dropout)
        self.ChannelMixingKAN = ChannelMixingKAN(d_core, channels, hidden_dim, dropout=dropout)


        # ── Replace VariateAttention with STARKAN ─────────────────────────────
        self.starkan = STARKAN(d_series=d_core, d_core=d_core // 2, grid_size=grid_size)

        self.proj_out = nn.Linear(d_core, time_steps, bias=True)
        self.drop_out = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, *args, **kwargs):
        z = self.proj_in(inputs)
        z = self.act(z)
        z = self.drop_in(z)
        z = self.kan_bottleneck(z)

        # PARALLELIZE: Execute Token and Channel mixings simultaneously
        x = self.TokenMixingKAN(z)
        y = self.ChannelMixingKAN(z)
        
        # Combine parallel pathways
        combined = x + y
        
        # STARKAN Redistribution
        y = self.starkan(combined)

        out = self.proj_out(y)
        out = self.drop_out(out)
        return out, None


# ─────────────────────────────────────────────────────────────────────────────
# MLP Mixer variant (unchanged structure, bug-fixed LayerNorm)
# ─────────────────────────────────────────────────────────────────────────────

class Mixer2dTriUMLP(nn.Module):
    """MLP-Mixer variant (reference / ablation)."""

    def __init__(
        self,
        time_steps: int,
        channels: int,
        d_core: int,
        gri_size: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super(Mixer2dTriUMLP, self).__init__()

        self.gen1 = nn.Sequential(
            nn.Linear(time_steps, time_steps),
            NewGELU(),
            nn.Linear(time_steps, d_core),
        )
        self.gen2 = nn.Sequential(
            nn.Linear(d_core, time_steps),
            NewGELU(),
            nn.Linear(time_steps, time_steps),
        )

        self.timeMixer = MixerBlock1(d_core, channels + 4, hidden_dim, dropout=dropout)
        self.channelMixer = MixerBlock2(d_core, channels + 4, hidden_dim, dropout=dropout)

    def forward(self, inputs: torch.Tensor, *args, **kwargs):
        combined_mean = self.gen1(inputs)
        x = self.timeMixer(combined_mean)
        x = x + combined_mean                        # outer residual only for MLP variant
        y = self.channelMixer(x)
        y = y + x
        z = self.gen2(y)
        return z, None


# ─────────────────────────────────────────────────────────────────────────────
# Main Model
# ─────────────────────────────────────────────────────────────────────────────

class Model(nn.Module):
    """
    KANMTS v2 — Simplified Expander + Gated Fusion.

    Improvements over v1:
    ✅ Mixer2dTriUKAN v2: Linear→KAN bottleneck + single Linear expander
       (replaces the double-KAN expander that was the main cause of overfitting)
    ✅ Gated seasonal/trend fusion: learnable per-variate sigmoid gate
       instead of fixed additive combination — model decides per-task weight
    ✅ Multi-scale trend: additional Conv1d trend branch captures short-range
       periodicity that the moving-average trend misses
    ✅ All v1 improvements retained (RevIN, decomp, VariateAttention, dropout)
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.batch_size = configs.batch_size
        N = configs.enc_in

        # ── Normalization ────────────────────────────────────────────────────
        self.revin = RevIN(num_features=N, affine=True)
        self.use_norm = configs.use_norm

        # ── Series Decomposition ─────────────────────────────────────────────
        self.decomp = SeriesDecomposition(kernel_size=configs.moving_avg)

        # ── Seasonal Branch Embedding ─────────────────────────────────────────
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.grid_size
        )

        # ── Seasonal Branch Encoder ───────────────────────────────────────────
        dropout = getattr(configs, 'dropout', 0.1)
        n_heads = getattr(configs, 'n_heads', 4)

        # Calculate actual channel count after DataEmbedding_inverted adds time features
        # (MonthDay, DayOfWeek, DayOfMonth, etc. based on frequency)
        if configs.embed == 'timeF':
            # Mapping based on utils/timefeatures.py
            freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
            # Handle aliases (e.g. 'min', 'minutely')
            f_key = configs.freq[0].lower() if configs.freq else 'h'
            num_time_marks = freq_map.get(f_key, 4)
        else:
            num_time_marks = 0
            
        total_channels = N + num_time_marks

        self.encoder = Encoder(
            [
                EncoderLayer(
                    Mixer2dTriUKAN(
                        configs.d_model,
                        total_channels,
                        configs.d_core,
                        configs.grid_size,
                        configs.hidden_dim,
                        dropout=dropout,
                        n_heads=n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
        )


        # ── Seasonal Branch Projection ────────────────────────────────────────
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        # ── Primary Trend Branch (DLinear-style per-variate linear) ───────────
        self.trend_projection = nn.Linear(configs.seq_len, configs.pred_len, bias=True)

        # ── Multi-scale Trend Branch (Inception-style Conv1d kernels) ────────
        # Uses multiple receptive fields (3, 5, 7) to capture different trend scales.
        self.ms_trend_3 = nn.Conv1d(N, N, kernel_size=3, padding=1, groups=N)
        self.ms_trend_5 = nn.Conv1d(N, N, kernel_size=5, padding=2, groups=N)
        self.ms_trend_7 = nn.Conv1d(N, N, kernel_size=7, padding=3, groups=N)
        
        self.ms_trend_proj = nn.Linear(configs.seq_len, configs.pred_len, bias=True)

        # ── KAN Gated Fusion ──────────────────────────────────────────────────
        # Fuses 3 branches: Seasonal, Linear Trend, Multi-scale Trend
        # Input size = pred_len * 3. Output size = pred_len.
        # We process this per-variate for efficiency and to reduce overfitting.
        self.fusion_kan = KANLinear(configs.pred_len * 3, configs.pred_len, grid_size=configs.grid_size)


    # ─────────────────────────────────────────────────────────────────────────

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, T, N = x_enc.shape

        # ── Step 1: Normalization ─────────────────────────────────────────────
        if self.use_norm:
            x_enc = self.revin(x_enc, mode='norm')

            # ── Step 2: Decomposition ─────────────────────────────────────────────
        seasonal, trend = self.decomp(x_enc)

        # ── Step 3: Trend Branches ────────────────────────────────────────────
        # Branch A: DLinear (Per-variate Linear)
        trend_a = self.trend_projection(trend.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Branch B: Multi-scale Conv (Local structure)
        x_p = x_enc.permute(0, 2, 1)
        ms3 = self.ms_trend_3(x_p)
        ms5 = self.ms_trend_5(x_p)
        ms7 = self.ms_trend_7(x_p)
        ms_trend = self.ms_trend_proj(ms3 + ms5 + ms7).permute(0, 2, 1)

        # ── Step 4: Seasonal Branch ───────────────────────────────────────────
        enc_out = self.enc_embedding(seasonal, x_mark_enc)
        enc_out, _ = self.encoder(enc_out)
        seasonal_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        # ── Step 5: KAN Fusion ────────────────────────────────────────────────
        # Concatenate components for fusion
        # Each component has shape (B, pred_len, N)
        combined_comp = torch.cat([seasonal_out, trend_a, ms_trend], dim=1) # (B, pred_len * 3, N)
        
        # Apply KAN Fusion per-variate
        # fusion expects (B, L_in, N) -> permute to (B, N, L_in)
        dec_out = self.fusion_kan(combined_comp.permute(0, 2, 1)).permute(0, 2, 1) # (B, pred_len, N)


        # ── Step 6: De-normalization ──────────────────────────────────────────
        if self.use_norm:
            dec_out = self.revin(dec_out, mode='denorm')

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]   # (B, pred_len, N)
