"""Evaluate released artifacts or run VIEW360 feature inference."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from metrics import anomaly_metrics, direction_accuracy

DEFAULT_DATA_ROOT = Path("VIEW360")


def _load_array(path: str | Path) -> np.ndarray:
    array_path = Path(path)
    if not array_path.is_file():
        raise FileNotFoundError(f"Missing evaluation artifact: {array_path}")
    return np.load(array_path, allow_pickle=False)


def evaluate_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    frame_ground_truth = _load_array(args.ground_truth)
    results: dict[str, Any] = {}

    if args.mgfn_scores is not None:
        results["MGFN"] = anomaly_metrics(
            frame_ground_truth, _load_array(args.mgfn_scores)
        ).as_dict()
    if args.fdpn_scores is not None:
        results["FDPN"] = anomaly_metrics(
            frame_ground_truth, _load_array(args.fdpn_scores)
        ).as_dict()

    if args.direction_predictions is not None:
        direction_ground_truth = _load_array(args.direction_ground_truth)
        if not np.array_equal(
            direction_ground_truth != args.direction_background,
            frame_ground_truth.astype(bool),
        ):
            raise ValueError(
                "Direction and anomaly ground truth disagree on evaluated frames"
            )
        results["FDPN"]["direction_accuracy"] = direction_accuracy(
            direction_ground_truth,
            _load_array(args.direction_predictions),
            background_label=args.direction_background,
        )
    return results


def _load_state_dict(path: str | Path) -> dict[str, object]:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must contain a state dictionary: {path}")
    for key in ("state_dict", "model_state_dict"):
        nested = checkpoint.get(key)
        if isinstance(nested, dict):
            checkpoint = nested
            break
    return {name.removeprefix("module."): value for name, value in checkpoint.items()}


def run_inference(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from datasets.view360 import VIEW360Dataset
    from models import FDPN, MGFN

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    is_fdpn = args.model == "fdpn"
    dataset = VIEW360Dataset(
        data_root=args.data_root,
        manifest=args.manifest,
        split="test",
        frame_feature_directory=args.frame_feature_directory if is_fdpn else None,
    )
    model = FDPN() if is_fdpn else MGFN()
    model.load_state_dict(_load_state_dict(args.checkpoint), strict=True)
    model = model.to(device).eval()

    limit = (
        len(dataset) if args.max_videos is None else min(args.max_videos, len(dataset))
    )
    predictions: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for index in range(limit):
            sample = dataset[index]
            snippets = torch.from_numpy(sample["snippet_features"]).unsqueeze(0)
            snippets = snippets.to(device)
            if is_fdpn:
                frames = torch.from_numpy(sample["frame_features"]).unsqueeze(0)
                frames = frames.to(device)
                scores = model(snippets, frames)["frame_scores"].flatten()
            else:
                scores = model(snippets).flatten().repeat_interleave(16)
            predictions.append(scores.cpu().numpy())
            if args.progress:
                print(f"[{index + 1}/{limit}] {sample['name']}", flush=True)

    scores = np.concatenate(predictions)
    elapsed = time.perf_counter() - started
    results: dict[str, Any] = {
        "model": args.model.upper(),
        "videos": limit,
        "frames": len(scores),
        "elapsed_seconds": elapsed,
        "frames_per_second": len(scores) / elapsed,
    }
    if args.output is not None:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, scores)
        results["output"] = str(output)

    if args.ground_truth is not None:
        ground_truth = _load_array(args.ground_truth)
        if limit != len(dataset):
            ground_truth = ground_truth[: len(scores)]
        results.update(anomaly_metrics(ground_truth, scores).as_dict())
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    artifacts = subparsers.add_parser(
        "artifacts", help="evaluate prediction arrays without model inference"
    )
    artifacts.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("data/view360_frame_ground_truth.npy"),
    )
    artifacts.add_argument(
        "--direction-ground-truth",
        type=Path,
        default=Path("data/view360_direction_ground_truth.npy"),
    )
    artifacts.add_argument(
        "--mgfn-scores",
        type=Path,
        default=Path("weights/mgfn_view360_scores.npy"),
    )
    artifacts.add_argument(
        "--fdpn-scores",
        type=Path,
        default=Path("weights/fdpn_view360_scores.npy"),
    )
    artifacts.add_argument(
        "--direction-predictions",
        type=Path,
        default=Path("weights/fdpn_view360_direction_predictions.npy"),
    )
    artifacts.add_argument("--direction-background", type=int, default=3)
    artifacts.set_defaults(handler=evaluate_artifacts)

    inference = subparsers.add_parser(
        "inference", help="run a checkpoint on pre-extracted VIEW360 features"
    )
    inference.add_argument("--model", choices=("fdpn", "mgfn"), required=True)
    inference.add_argument("--checkpoint", type=Path, required=True)
    inference.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    inference.add_argument(
        "--manifest", type=Path, default=Path("data/view360_test_i3d.txt")
    )
    inference.add_argument(
        "--frame-feature-directory",
        default="view360_masked_resnet",
        help="masked ResNet feature directory under VIEW360 (FDPN only)",
    )
    inference.add_argument(
        "--ground-truth", type=Path, default=Path("data/view360_frame_ground_truth.npy")
    )
    inference.add_argument("--device", default="auto")
    inference.add_argument("--max-videos", type=int)
    inference.add_argument("--output", type=Path)
    inference.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=True
    )
    inference.set_defaults(handler=run_inference)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = args.handler(args)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
