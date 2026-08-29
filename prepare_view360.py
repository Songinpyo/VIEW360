"""Generate VIEW360 split manifests and frame-level ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from datasets.view360 import (
    build_frame_ground_truth,
    find_snippet_directory,
    make_split_manifest,
    read_temporal_annotations,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("VIEW360"),
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/view360_temporal_annotations.txt"),
    )
    parser.add_argument("--output-directory", type=Path, default=Path("data"))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_entries = make_split_manifest(find_snippet_directory(args.data_root, "train"))
    test_entries = make_split_manifest(find_snippet_directory(args.data_root, "test"))
    annotations = read_temporal_annotations(args.annotations)
    frame_ground_truth, direction_ground_truth = build_frame_ground_truth(
        test_entries, annotations
    )

    summary = {
        "train_videos": len(train_entries),
        "train_abnormal": sum(
            not value.startswith("Normal_") for value in train_entries
        ),
        "train_normal": sum(value.startswith("Normal_") for value in train_entries),
        "test_videos": len(test_entries),
        "test_abnormal": sum(not value.startswith("Normal_") for value in test_entries),
        "test_normal": sum(value.startswith("Normal_") for value in test_entries),
        "test_frames": len(frame_ground_truth),
        "test_anomaly_frames": int(frame_ground_truth.sum()),
    }

    if not args.dry_run:
        args.output_directory.mkdir(parents=True, exist_ok=True)
        (args.output_directory / "view360_train_i3d.txt").write_text(
            "\n".join(train_entries) + "\n"
        )
        (args.output_directory / "view360_test_i3d.txt").write_text(
            "\n".join(test_entries) + "\n"
        )
        np.save(
            args.output_directory / "view360_frame_ground_truth.npy",
            frame_ground_truth,
        )
        np.save(
            args.output_directory / "view360_direction_ground_truth.npy",
            direction_ground_truth,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
