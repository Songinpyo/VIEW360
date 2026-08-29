"""Train the RTFM snippet stage or the FDPN frame stage on VIEW360."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader

from datasets.view360 import VIEW360Dataset
from metrics import anomaly_metrics
from models import FDPN, RTFM


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must contain a state dictionary: {path}")
    for key in ("state_dict", "model_state_dict"):
        nested = checkpoint.get(key)
        if isinstance(nested, dict):
            checkpoint = nested
            break
    return {
        name.removeprefix("module."): value
        for name, value in checkpoint.items()
        if torch.is_tensor(value)
    }


def load_compatible_weights(
    model: nn.Module, checkpoint: Path
) -> tuple[int, list[str]]:
    state = load_checkpoint(checkpoint)
    expected = model.state_dict()
    incompatible_shapes = [
        name
        for name, value in state.items()
        if name in expected and expected[name].shape != value.shape
    ]
    if incompatible_shapes:
        raise ValueError(
            "Checkpoint contains incompatible tensors: "
            + ", ".join(incompatible_shapes)
        )
    compatible = {name: value for name, value in state.items() if name in expected}
    if not compatible:
        raise ValueError(f"No compatible FDPN/RTFM weights found in {checkpoint}")
    result = model.load_state_dict(compatible, strict=False)
    return len(compatible), list(result.missing_keys)


def _topk_rtfm_outputs(
    encoded_features: torch.Tensor,
    snippet_scores: torch.Tensor,
    pair_batch_size: int,
    crops: int,
    dropout: nn.Module,
    top_k: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    snippets = snippet_scores.shape[1]
    channels = encoded_features.shape[-1]
    features = encoded_features.reshape(pair_batch_size * 2, crops, snippets, channels)
    magnitudes = torch.linalg.vector_norm(features, ord=2, dim=3).mean(dim=1)
    normal_magnitudes = magnitudes[:pair_batch_size]
    abnormal_magnitudes = magnitudes[pair_batch_size:]
    top_k = min(top_k, snippets)

    normal_indices = torch.topk(
        normal_magnitudes * dropout(torch.ones_like(normal_magnitudes)),
        top_k,
        dim=1,
    ).indices
    abnormal_indices = torch.topk(
        abnormal_magnitudes * dropout(torch.ones_like(abnormal_magnitudes)),
        top_k,
        dim=1,
    ).indices

    def select_features(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        expanded = indices[:, None, :, None].expand(
            pair_batch_size, crops, top_k, channels
        )
        selected = torch.gather(values, dim=2, index=expanded)
        return selected.reshape(pair_batch_size * crops, top_k, channels)

    normal_features = select_features(features[:pair_batch_size], normal_indices)
    abnormal_features = select_features(features[pair_batch_size:], abnormal_indices)
    normal_scores = torch.gather(
        snippet_scores[:pair_batch_size], 1, normal_indices.unsqueeze(-1)
    ).mean(dim=1)
    abnormal_scores = torch.gather(
        snippet_scores[pair_batch_size:], 1, abnormal_indices.unsqueeze(-1)
    ).mean(dim=1)
    return normal_scores, abnormal_scores, normal_features, abnormal_features


def rtfm_loss(
    encoded_features: torch.Tensor,
    snippet_scores: torch.Tensor,
    pair_batch_size: int,
    crops: int,
    dropout: nn.Module,
    alpha: float = 1e-4,
    margin: float = 100.0,
) -> torch.Tensor:
    normal_scores, abnormal_scores, normal_features, abnormal_features = (
        _topk_rtfm_outputs(
            encoded_features,
            snippet_scores,
            pair_batch_size,
            crops,
            dropout,
        )
    )
    scores = torch.cat((normal_scores, abnormal_scores)).squeeze(-1)
    labels = torch.cat(
        (
            torch.zeros(pair_batch_size, device=scores.device),
            torch.ones(pair_batch_size, device=scores.device),
        )
    )
    classification = functional.binary_cross_entropy(scores, labels)

    abnormal_magnitude = torch.linalg.vector_norm(
        abnormal_features.mean(dim=1), ord=2, dim=1
    )
    normal_magnitude = torch.linalg.vector_norm(
        normal_features.mean(dim=1), ord=2, dim=1
    )
    magnitude_contrast = torch.mean(
        (torch.abs(margin - abnormal_magnitude) + normal_magnitude) ** 2
    )

    abnormal_sequence = snippet_scores[pair_batch_size:].squeeze(-1)
    sparsity = 8e-3 * abnormal_sequence.mean()
    smoothness = 8e-4 * torch.mean(
        (abnormal_sequence[:, 1:] - abnormal_sequence[:, :-1]) ** 2
    )
    return classification + alpha * magnitude_contrast + sparsity + smoothness


def binary_focal_loss(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    probabilities = probabilities.clamp(1e-6, 1.0 - 1e-6)
    positive = -targets * (1.0 - probabilities) ** gamma * probabilities.log()
    negative = -(1.0 - targets) * probabilities**gamma * (1.0 - probabilities).log()
    return (positive + negative).mean()


def fdpn_frame_loss(
    frame_scores: torch.Tensor,
    snippet_scores: torch.Tensor,
    pair_batch_size: int,
    frames_per_snippet: int = 16,
    ranking_frames: int = 48,
    ranking_weight: float = 1.0,
    smoothness_weight: float = 1.6e-3,
) -> torch.Tensor:
    frame_scores = frame_scores.squeeze(-1)
    with torch.no_grad():
        pseudo_labels = torch.zeros_like(snippet_scores.squeeze(-1))
        pseudo_labels[pair_batch_size:] = (
            snippet_scores[pair_batch_size:].squeeze(-1) >= 0.5
        ).float()
        pseudo_labels = pseudo_labels.repeat_interleave(frames_per_snippet, dim=1)

    focal = binary_focal_loss(frame_scores, pseudo_labels)
    ranking_frames = min(ranking_frames, frame_scores.shape[1])
    normal_top = torch.topk(
        frame_scores[:pair_batch_size], ranking_frames, dim=1
    ).values
    abnormal_top = torch.topk(
        frame_scores[pair_batch_size:], ranking_frames, dim=1
    ).values
    ranking = torch.mean(1.0 - abnormal_top + normal_top)
    abnormal = frame_scores[pair_batch_size:]
    smoothness = torch.mean((abnormal[:, 1:] - abnormal[:, :-1]) ** 2)
    return focal + ranking_weight * ranking + smoothness_weight * smoothness


def paired_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    dataset_options = {
        "data_root": args.data_root,
        "manifest": args.train_manifest,
        "split": "train",
        "frame_feature_directory": (
            args.frame_feature_directory if args.stage == "frame" else None
        ),
    }
    normal = VIEW360Dataset(**dataset_options, normal=True)
    abnormal = VIEW360Dataset(**dataset_options, normal=False)
    common = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.workers,
        "pin_memory": args.pin_memory,
        "drop_last": True,
    }
    return DataLoader(normal, **common), DataLoader(abnormal, **common)


def train_epoch(
    model: FDPN | RTFM,
    normal_loader: DataLoader,
    abnormal_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    stage: str,
) -> float:
    model.train()
    losses: list[float] = []
    for normal, abnormal in zip(normal_loader, abnormal_loader):
        snippets = torch.cat(
            (normal["snippet_features"], abnormal["snippet_features"])
        ).to(device)
        pair_batch_size = normal["snippet_features"].shape[0]
        optimizer.zero_grad(set_to_none=True)

        if stage == "snippet":
            encoded, snippet_scores, _ = model.encode_snippets(snippets)
            loss = rtfm_loss(
                encoded,
                snippet_scores,
                pair_batch_size,
                snippets.shape[1],
                model.drop_out,
            )
        else:
            frames = torch.cat(
                (normal["frame_features"], abnormal["frame_features"])
            ).to(device)
            outputs = model(snippets, frames)
            loss = fdpn_frame_loss(
                outputs["frame_scores"], outputs["snippet_scores"], pair_batch_size
            )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    if not losses:
        raise RuntimeError("No full paired batches were available for training")
    return float(np.mean(losses))


def evaluate(
    model: FDPN | RTFM,
    dataset: VIEW360Dataset,
    ground_truth: np.ndarray,
    device: torch.device,
    stage: str,
) -> dict[str, float]:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for sample in dataset:
            snippets = (
                torch.from_numpy(sample["snippet_features"]).unsqueeze(0).to(device)
            )
            if stage == "frame":
                frames = (
                    torch.from_numpy(sample["frame_features"]).unsqueeze(0).to(device)
                )
                scores = model(snippets, frames)["frame_scores"].flatten()
            else:
                _, scores, _ = model.encode_snippets(snippets)
                scores = scores.flatten().repeat_interleave(16)
            predictions.append(scores.cpu().numpy())
    return anomaly_metrics(ground_truth, np.concatenate(predictions)).as_dict()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("snippet", "frame"), required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("VIEW360"),
    )
    parser.add_argument(
        "--train-manifest", type=Path, default=Path("data/view360_train_i3d.txt")
    )
    parser.add_argument(
        "--test-manifest", type=Path, default=Path("data/view360_test_i3d.txt")
    )
    parser.add_argument(
        "--ground-truth", type=Path, default=Path("data/view360_frame_ground_truth.npy")
    )
    parser.add_argument("--frame-feature-directory", default="view360_masked_resnet")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-directory", type=Path, default=Path("runs"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--pin-memory", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--evaluate-every", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stage == "frame" and args.checkpoint is None:
        raise SystemExit("--checkpoint is required when training the FDPN frame stage")
    seed_everything(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model: FDPN | RTFM = FDPN() if args.stage == "frame" else RTFM()
    if args.checkpoint is not None:
        loaded, missing = load_compatible_weights(model, args.checkpoint)
        print(f"Loaded {loaded} tensors from {args.checkpoint}")
        if missing:
            print(f"Initialized {len(missing)} missing tensors from scratch")

    if args.stage == "frame":
        assert isinstance(model, FDPN)
        model.freeze_snippet_network()
    model = model.to(device)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    normal_loader, abnormal_loader = paired_loaders(args)
    test_dataset = VIEW360Dataset(
        args.data_root,
        args.test_manifest,
        "test",
        frame_feature_directory=(
            args.frame_feature_directory if args.stage == "frame" else None
        ),
    )
    ground_truth = np.load(args.ground_truth, allow_pickle=False)

    args.output_directory.mkdir(parents=True, exist_ok=True)
    configuration: dict[str, Any] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    (args.output_directory / "config.json").write_text(
        json.dumps(configuration, indent=2, sort_keys=True) + "\n"
    )
    best_roc_auc = -1.0
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(
            model,
            normal_loader,
            abnormal_loader,
            optimizer,
            device,
            args.stage,
        )
        print(f"epoch={epoch} loss={loss:.6f}")
        should_evaluate = args.evaluate_every > 0 and (
            epoch % args.evaluate_every == 0 or epoch == args.epochs
        )
        if should_evaluate:
            metrics = evaluate(model, test_dataset, ground_truth, device, args.stage)
            print(json.dumps({"epoch": epoch, **metrics}, sort_keys=True))
            if metrics["roc_auc"] > best_roc_auc:
                best_roc_auc = metrics["roc_auc"]
                torch.save(model.state_dict(), args.output_directory / "best_model.pth")
    torch.save(model.state_dict(), args.output_directory / "last_model.pth")


if __name__ == "__main__":
    main()
