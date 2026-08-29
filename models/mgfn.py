"""MGFN baseline used in the VIEW360 comparison."""

from __future__ import annotations

import torch
from einops import rearrange
from torch import einsum, nn


class LayerNorm(nn.Module):
    def __init__(self, dim: int, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        variance = torch.var(inputs, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(inputs, dim=1, keepdim=True)
        return (inputs - mean) / (variance.sqrt() + self.epsilon) * self.g + self.b


def feed_forward(dim: int, expansion: int = 4, dropout: float = 0.0) -> nn.Module:
    return nn.Sequential(
        LayerNorm(dim),
        nn.Conv1d(dim, dim * expansion, kernel_size=1),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Conv1d(dim * expansion, dim, kernel_size=1),
    )


class Focus(nn.Module):
    """Local multi-head relation aggregator from MGFN."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int = 64,
        local_kernel: int = 5,
    ) -> None:
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = nn.BatchNorm1d(dim)
        self.to_v = nn.Conv1d(dim, inner_dim, kernel_size=1, bias=False)
        self.rel_pos = nn.Conv1d(
            heads,
            heads,
            local_kernel,
            padding=local_kernel // 2,
            groups=heads,
        )
        self.to_out = nn.Conv1d(inner_dim, dim, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(inputs)
        batch_size = normalized.shape[0]
        values = self.to_v(normalized)
        values = rearrange(values, "b (c h) t -> (b c) h t", h=self.heads)
        outputs = self.rel_pos(values)
        outputs = rearrange(outputs, "(b c) h t -> b (c h) t", b=batch_size)
        return self.to_out(outputs)


class Glance(nn.Module):
    """Global multi-head relation aggregator from MGFN."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        del dropout
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * heads
        self.norm = LayerNorm(dim)
        self.to_qkv = nn.Conv1d(dim, inner_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv1d(inner_dim, dim, kernel_size=1)
        self.attn: torch.Tensor | int = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        shape = inputs.shape
        normalized = self.norm(inputs)
        queries, keys, values = self.to_qkv(normalized).chunk(3, dim=1)
        queries, keys, values = (
            rearrange(tensor, "b (h d) n -> b h n d", h=self.heads)
            for tensor in (queries, keys, values)
        )
        similarity = einsum("b h i d, b h j d -> b h i j", queries * self.scale, keys)
        self.attn = similarity.softmax(dim=-1)
        outputs = einsum("b h i j, b h j d -> b h i d", self.attn, values)
        outputs = rearrange(outputs, "b h n d -> b (h d) n")
        return self.to_out(outputs).reshape(*shape)


class Backbone(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        relation_type: str = "gb",
        kernel: int = 5,
        dim_head: int = 64,
        feed_forward_expansion: int = 4,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            if relation_type == "fb":
                relation = Focus(dim, heads, dim_head, kernel)
            elif relation_type == "gb":
                relation = Glance(dim, heads, dim_head, attention_dropout)
            else:
                raise ValueError(f"Unknown MGFN relation type: {relation_type}")
            self.layers.append(
                nn.ModuleList(
                    (
                        nn.Conv1d(dim, dim, kernel_size=3, padding=1),
                        relation,
                        feed_forward(dim, feed_forward_expansion, dropout),
                    )
                )
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs
        for convolution, relation, feed_forward_layer in self.layers:
            outputs = convolution(outputs) + outputs
            outputs = relation(outputs) + outputs
            outputs = feed_forward_layer(outputs) + outputs
        return outputs


class MGFN(nn.Module):
    """Magnitude-contrastive Glance-and-Focus Network."""

    def __init__(
        self,
        feature_size: int = 2048,
        dims: tuple[int, ...] = (64, 128, 1024),
        depths: tuple[int, ...] = (3, 3, 2),
        relation_types: tuple[str, ...] = ("gb", "fb", "fb"),
        dim_head: int = 64,
        magnitude_ratio: float = 0.1,
        selection_dropout: float = 0.7,
    ) -> None:
        super().__init__()
        if not (len(dims) == len(depths) == len(relation_types)):
            raise ValueError("dims, depths and relation_types must have equal lengths")
        self.feature_size = feature_size
        self.magnitude_ratio = magnitude_ratio
        initial_dim = dims[0]
        last_dim = dims[-1]

        self.to_tokens = nn.Conv1d(feature_size, initial_dim, kernel_size=3, padding=1)
        self.stages = nn.ModuleList()
        for index, (depth, relation_type) in enumerate(zip(depths, relation_types)):
            stage_dim = dims[index]
            is_last = index == len(dims) - 1
            self.stages.append(
                nn.ModuleList(
                    (
                        Backbone(
                            dim=stage_dim,
                            depth=depth,
                            heads=stage_dim // dim_head,
                            relation_type=relation_type.lower(),
                        ),
                        None
                        if is_last
                        else nn.Sequential(
                            LayerNorm(stage_dim),
                            nn.Conv1d(stage_dim, dims[index + 1], kernel_size=1),
                        ),
                    )
                )
            )

        self.to_logits = nn.Sequential(nn.LayerNorm(last_dim))
        self.fc = nn.Linear(last_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.drop_out = nn.Dropout(selection_dropout)
        self.to_mag = nn.Conv1d(1, initial_dim, kernel_size=3, padding=1)

    def forward_features(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 4:
            raise ValueError("MGFN input must have shape [B, crops, T, C]")
        batch_size, crops, snippets, channels = inputs.shape
        if channels != self.feature_size:
            raise ValueError(
                f"Expected {self.feature_size} feature channels, received {channels}"
            )

        flattened = inputs.reshape(batch_size * crops, snippets, channels)
        flattened = flattened.permute(0, 2, 1)
        magnitudes = torch.linalg.vector_norm(flattened, ord=2, dim=1, keepdim=True)
        features = self.to_tokens(flattened)
        features = features + self.magnitude_ratio * self.to_mag(magnitudes)
        for backbone, projection in self.stages:
            features = backbone(features)
            if projection is not None:
                features = projection(features)
        features = features.permute(0, 2, 1)
        crop_scores = self.sigmoid(self.fc(self.to_logits(features)))
        return features, crop_scores

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, crop_scores = self.forward_features(inputs)
        batch_size, crops = inputs.shape[:2]
        return crop_scores.reshape(batch_size, crops, -1).mean(dim=1).unsqueeze(-1)
