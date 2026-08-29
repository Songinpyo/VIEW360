"""Frame and Direction Prediction Network (FDPN).

The VIEW360 checkpoint released with the paper uses RTFM as its snippet
network and 2,048-dimensional ResNet features from saliency-masked frames.
The public checkpoint names are preserved (``Aggregate.*``, ``fc*`` and
``FPS.*``) so that the original weights load strictly.
"""

from __future__ import annotations

from typing import TypedDict

import torch
from torch import nn

from .poolformer import PatchEmbed, PoolFormerBlock


class FDPNOutput(TypedDict, total=False):
    """Outputs returned by :class:`FDPN`."""

    frame_scores: torch.Tensor
    snippet_scores: torch.Tensor
    snippet_features: torch.Tensor
    feature_magnitudes: torch.Tensor
    direction_scores: torch.Tensor


def _initialize_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv1d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class _NonLocalBlock1D(nn.Module):
    """The non-local block used by the RTFM feature aggregator."""

    def __init__(
        self,
        in_channels: int,
        inter_channels: int | None = None,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels or max(in_channels // 2, 1)

        self.g = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1)
        self.theta = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1)

        if batch_norm:
            self.W = nn.Sequential(
                nn.Conv1d(self.inter_channels, in_channels, kernel_size=1),
                nn.BatchNorm1d(in_channels),
            )
            nn.init.zeros_(self.W[1].weight)
            nn.init.zeros_(self.W[1].bias)
        else:
            self.W = nn.Conv1d(self.inter_channels, in_channels, kernel_size=1)
            nn.init.zeros_(self.W.weight)
            nn.init.zeros_(self.W.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        values = self.g(inputs).reshape(batch_size, self.inter_channels, -1)
        values = values.permute(0, 2, 1)
        queries = self.theta(inputs).reshape(batch_size, self.inter_channels, -1)
        queries = queries.permute(0, 2, 1)
        keys = self.phi(inputs).reshape(batch_size, self.inter_channels, -1)

        affinities = torch.matmul(queries, keys) / keys.shape[-1]
        outputs = torch.matmul(affinities, values)
        outputs = outputs.permute(0, 2, 1).contiguous()
        outputs = outputs.reshape(batch_size, self.inter_channels, *inputs.shape[2:])
        return self.W(outputs) + inputs


class Aggregate(nn.Module):
    """Multi-scale temporal feature aggregation from RTFM."""

    def __init__(self, len_feature: int = 2048) -> None:
        super().__init__()
        self.len_feature = len_feature
        self.conv_1 = nn.Sequential(
            nn.Conv1d(len_feature, 512, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(len_feature, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.BatchNorm1d(512),
        )
        self.conv_3 = nn.Sequential(
            nn.Conv1d(len_feature, 512, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
            nn.BatchNorm1d(512),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv1d(len_feature, 512, kernel_size=1, bias=False),
            nn.ReLU(),
        )
        self.conv_5 = nn.Sequential(
            nn.Conv1d(len_feature, len_feature, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(len_feature),
        )
        self.non_local = _NonLocalBlock1D(512, batch_norm=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = inputs.permute(0, 2, 1)
        residual = features
        dilated = torch.cat(
            (self.conv_1(features), self.conv_2(features), self.conv_3(features)),
            dim=1,
        )
        non_local = self.non_local(self.conv_4(features))
        features = self.conv_5(torch.cat((dilated, non_local), dim=1))
        return (features + residual).permute(0, 2, 1)


class FramePredictionSubnet(nn.Module):
    """PoolFormer-based frame prediction subnetwork (FPS)."""

    def __init__(
        self,
        input_size: int = 4096,
        hidden_size: int = 1024,
        output_size: int = 1,
    ) -> None:
        super().__init__()
        self.poolformer1 = PoolFormerBlock(dim=input_size)
        self.patch_emb1 = PatchEmbed(in_chans=input_size, embed_dim=hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, hidden_size // 8),
            nn.ReLU(),
            nn.Linear(hidden_size // 8, hidden_size // 16),
            nn.ReLU(),
            nn.Linear(hidden_size // 16, output_size),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.poolformer1(inputs)
        features = self.patch_emb1(features).permute(0, 2, 1)
        return torch.sigmoid(self.fc(features))


class DirectionPredictionSubnet(nn.Module):
    """PoolFormer-based direction prediction subnetwork (DPS)."""

    def __init__(
        self,
        input_size: int = 4096,
        hidden_size: int = 1024,
        output_size: int = 3,
    ) -> None:
        super().__init__()
        self.poolformer1 = PoolFormerBlock(dim=input_size)
        self.patch_emb1 = PatchEmbed(in_chans=input_size, embed_dim=hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, hidden_size // 8),
            nn.ReLU(),
            nn.Linear(hidden_size // 8, hidden_size // 16),
            nn.ReLU(),
            nn.Linear(hidden_size // 16, output_size),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.poolformer1(inputs)
        features = self.patch_emb1(features).permute(0, 2, 1)
        return self.fc(features)


class RTFM(nn.Module):
    """RTFM snippet network used as FDPN's coarse prediction stage."""

    def __init__(self, feature_size: int = 2048) -> None:
        super().__init__()
        if feature_size != 2048:
            raise ValueError("The released VIEW360 FDPN expects 2,048-D features")
        self.feature_size = feature_size
        self.Aggregate = Aggregate(len_feature=feature_size)
        self.fc1 = nn.Linear(feature_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.drop_out = nn.Dropout(0.7)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.apply(_initialize_weights)

    def encode_snippets(
        self, snippet_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode ten-crop I3D features and predict snippet anomaly scores."""

        if snippet_features.ndim != 4:
            raise ValueError("snippet_features must have shape [B, crops, T, C]")
        batch_size, crops, snippets, channels = snippet_features.shape
        if channels != self.feature_size:
            raise ValueError(
                f"Expected {self.feature_size} I3D channels, received {channels}"
            )

        features = snippet_features.reshape(-1, snippets, channels)
        features = self.drop_out(self.Aggregate(features))
        scores = self.relu(self.fc1(features))
        scores = self.drop_out(scores)
        scores = self.relu(self.fc2(scores))
        scores = self.drop_out(scores)
        scores = self.sigmoid(self.fc3(scores))
        scores = scores.reshape(batch_size, crops, snippets).mean(dim=1)
        scores = scores.unsqueeze(-1)
        magnitudes = torch.linalg.vector_norm(features, ord=2, dim=2)
        magnitudes = magnitudes.reshape(batch_size, crops, snippets).mean(dim=1)
        return features, scores, magnitudes

    def forward(self, snippet_features: torch.Tensor) -> torch.Tensor:
        return self.encode_snippets(snippet_features)[1]


class FDPN(RTFM):
    """RTFM-based FDPN used for the VIEW360 result in WACV 2025."""

    def __init__(
        self,
        feature_size: int = 2048,
        frames_per_snippet: int = 16,
        with_direction: bool = False,
        direction_network_weight: float = 0.01,
    ) -> None:
        super().__init__(feature_size=feature_size)
        self.frames_per_snippet = frames_per_snippet
        self.direction_network_weight = direction_network_weight
        self.FPS = FramePredictionSubnet(input_size=feature_size * 2)
        self.DPS = (
            DirectionPredictionSubnet(input_size=feature_size * 2)
            if with_direction
            else None
        )

    def freeze_snippet_network(self) -> None:
        """Freeze RTFM and leave the paper's frame/direction subnets trainable."""

        for name, parameter in self.named_parameters():
            parameter.requires_grad = name.startswith(("FPS.", "DPS."))

    def _combine_frame_features(
        self,
        encoded_snippets: torch.Tensor,
        frame_features: torch.Tensor,
        batch_size: int,
        crops: int,
        snippets: int,
    ) -> torch.Tensor:
        if frame_features.ndim == 5:
            frame_features = frame_features[:, 0]
        if frame_features.ndim != 4:
            raise ValueError("frame_features must have shape [B, T, N, C]")
        expected = (batch_size, snippets, self.frames_per_snippet, self.feature_size)
        if tuple(frame_features.shape) != expected:
            raise ValueError(
                f"Expected frame features with shape {expected}, "
                f"received {tuple(frame_features.shape)}"
            )

        snippets_by_video = encoded_snippets.reshape(
            batch_size, crops, snippets, self.feature_size
        ).mean(dim=1)
        snippets_by_frame = snippets_by_video.repeat_interleave(
            self.frames_per_snippet, dim=1
        ).permute(0, 2, 1)
        frames = frame_features.reshape(
            batch_size, snippets * self.frames_per_snippet, self.feature_size
        ).permute(0, 2, 1)
        return torch.cat((snippets_by_frame, frames), dim=1)

    @staticmethod
    def _saliency_direction_probabilities(saliency_maps: torch.Tensor) -> torch.Tensor:
        if saliency_maps.ndim == 5:
            saliency_maps = saliency_maps[:, 0]
        if saliency_maps.ndim != 4 or saliency_maps.shape[-1] != 9:
            raise ValueError("saliency_maps must have shape [B, T, N, 9]")
        maps = saliency_maps.reshape(saliency_maps.shape[0], -1, 9)
        direction_evidence = torch.stack(
            (
                maps[:, :, 0:3].sum(dim=2),
                maps[:, :, 3] + maps[:, :, 5],
                maps[:, :, 6:9].sum(dim=2),
            ),
            dim=2,
        )
        return torch.softmax(direction_evidence, dim=-1)

    def forward(
        self,
        snippet_features: torch.Tensor,
        frame_features: torch.Tensor,
        saliency_maps: torch.Tensor | None = None,
    ) -> FDPNOutput:
        batch_size, crops, snippets, _ = snippet_features.shape
        encoded, snippet_scores, magnitudes = self.encode_snippets(snippet_features)
        combined = self._combine_frame_features(
            encoded, frame_features, batch_size, crops, snippets
        )
        output: FDPNOutput = {
            "frame_scores": self.FPS(combined),
            "snippet_scores": snippet_scores,
            "snippet_features": encoded,
            "feature_magnitudes": magnitudes,
        }

        if self.DPS is not None:
            direction_logits = self.DPS(combined)
            if saliency_maps is None:
                output["direction_scores"] = torch.softmax(direction_logits, dim=-1)
            else:
                saliency_probabilities = self._saliency_direction_probabilities(
                    saliency_maps
                )
                network_weight = self.direction_network_weight
                output["direction_scores"] = torch.softmax(
                    network_weight * direction_logits
                    + (1.0 - network_weight) * saliency_probabilities,
                    dim=-1,
                )
        return output
