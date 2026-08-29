"""One-dimensional PoolFormer blocks used by FDPN."""

from __future__ import annotations

import torch
from torch import nn


class Pooling(nn.Module):
    """PoolFormer token mixer for temporal feature sequences."""

    def __init__(self, pool_size: int = 3) -> None:
        super().__init__()
        self.pool = nn.AvgPool1d(
            pool_size,
            stride=1,
            padding=pool_size // 2,
            count_include_pad=False,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.pool(inputs) - inputs


class GroupNorm(nn.GroupNorm):
    """Channel-wise group normalization with one group."""

    def __init__(self, num_channels: int, **kwargs: object) -> None:
        super().__init__(1, num_channels, **kwargs)


class Mlp(nn.Module):
    """The pointwise temporal MLP used by a PoolFormer block."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        activation: type[nn.Module] = nn.GELU,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Conv1d(in_features, hidden_features, kernel_size=1)
        self.act = activation()
        self.fc2 = nn.Conv1d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.fc1(inputs)
        outputs = self.act(outputs)
        outputs = self.drop(outputs)
        outputs = self.fc2(outputs)
        return self.drop(outputs)


class PatchEmbed(nn.Module):
    """A temporal convolution used to project PoolFormer features."""

    def __init__(
        self,
        patch_size: int = 5,
        stride: int = 1,
        padding: int = 2,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv1d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(inputs))


class PoolFormerBlock(nn.Module):
    """A PoolFormer block operating on ``[batch, channels, frames]``."""

    def __init__(
        self,
        dim: int,
        pool_size: int = 3,
        mlp_ratio: float = 4.0,
        activation: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = GroupNorm,
        dropout: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            activation=activation,
            dropout=dropout,
        )
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs
        if self.use_layer_scale:
            scale_1 = self.layer_scale_1.unsqueeze(0).unsqueeze(-1)
            scale_2 = self.layer_scale_2.unsqueeze(0).unsqueeze(-1)
            outputs = outputs + scale_1 * self.token_mixer(self.norm1(outputs))
            outputs = outputs + scale_2 * self.mlp(self.norm2(outputs))
            return outputs

        outputs = outputs + self.token_mixer(self.norm1(outputs))
        return outputs + self.mlp(self.norm2(outputs))
