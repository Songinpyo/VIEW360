"""VIEW360 feature dataset and metadata utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from torch.utils.data import Dataset

FRAMES_PER_SNIPPET = 16
FEATURE_SIZE = 2048
DIRECTION_TO_ID = {"l": 0, "c": 1, "r": 2}
DIRECTION_BACKGROUND = 3
SNIPPET_DIRECTORIES = {
    "train": "view360_train_i3d",
    "test": "view360_test_i3d",
}


@dataclass(frozen=True)
class TemporalAnnotation:
    """One indexed, inclusive anomaly intervals for a VIEW360 video."""

    video_name: str
    intervals: tuple[tuple[int, int], ...]

    @property
    def directions(self) -> tuple[int, ...]:
        direction_code = self.video_name.rsplit("_", 1)[-1]
        try:
            return tuple(DIRECTION_TO_ID[value] for value in direction_code)
        except KeyError as error:
            raise ValueError(
                f"Unknown direction code in annotation {self.video_name!r}"
            ) from error


def read_manifest(path: str | Path) -> list[str]:
    """Read feature basenames from a split manifest."""

    manifest = Path(path)
    entries = [
        line.strip() for line in manifest.read_text().splitlines() if line.strip()
    ]
    if not entries:
        raise ValueError(f"Manifest is empty: {manifest}")
    return [Path(entry).name for entry in entries]


def parse_feature_filename(filename: str) -> tuple[str, int]:
    """Return the annotation stem and frame count encoded in a feature name."""

    path = Path(filename)
    stem, separator, frame_count = path.stem.rpartition("_")
    if not separator or not frame_count.isdigit():
        raise ValueError(
            f"Expected a VIEW360 feature name ending in _<frames>.npy: {filename}"
        )
    return stem, int(frame_count)


def read_temporal_annotations(
    path: str | Path,
) -> dict[str, TemporalAnnotation]:
    """Parse the temporal/directional annotation text distributed with VIEW360."""

    annotations: dict[str, TemporalAnnotation] = {}
    for line_number, raw_line in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) < 3 or len(fields[1:]) % 2:
            raise ValueError(f"Malformed annotation at line {line_number}: {raw_line}")
        boundaries = [int(value) for value in fields[1:]]
        intervals = tuple(zip(boundaries[::2], boundaries[1::2]))
        annotation = TemporalAnnotation(fields[0], intervals)
        if len(annotation.directions) != len(intervals):
            raise ValueError(
                f"Direction/interval mismatch at line {line_number}: {raw_line}"
            )
        annotations[annotation.video_name] = annotation
    return annotations


def build_frame_ground_truth(
    manifest_entries: Sequence[str],
    annotations: dict[str, TemporalAnnotation],
) -> tuple[np.ndarray, np.ndarray]:
    """Build frame anomaly and direction ground truth in manifest order."""

    anomaly_parts: list[np.ndarray] = []
    direction_parts: list[np.ndarray] = []
    for entry in manifest_entries:
        video_name, frame_count = parse_feature_filename(entry)
        anomaly = np.zeros(frame_count, dtype=np.int64)
        direction = np.full(frame_count, DIRECTION_BACKGROUND, dtype=np.int64)
        annotation = annotations.get(video_name)
        if annotation is not None:
            for (start, end), direction_id in zip(
                annotation.intervals, annotation.directions
            ):
                if not 1 <= start <= end <= frame_count:
                    raise ValueError(
                        f"Invalid interval {(start, end)} for {entry} ({frame_count} frames)"
                    )
                anomaly[start - 1 : end] = 1
                direction[start - 1 : end] = direction_id
        elif not video_name.startswith("Normal_"):
            raise ValueError(f"Missing temporal annotation for abnormal video: {entry}")
        anomaly_parts.append(anomaly)
        direction_parts.append(direction)

    return np.concatenate(anomaly_parts), np.concatenate(direction_parts)


def make_split_manifest(feature_directory: str | Path) -> list[str]:
    """Create the anomaly-first order used by the released VIEW360 ground truth."""

    files = sorted(path.name for path in Path(feature_directory).glob("*.npy"))
    abnormal = [filename for filename in files if not filename.startswith("Normal_")]
    normal = sorted(
        (filename for filename in files if filename.startswith("Normal_")), reverse=True
    )
    return abnormal + normal


def find_snippet_directory(data_root: str | Path, split: str) -> Path:
    """Return the I3D feature directory for a VIEW360 split."""

    if split not in SNIPPET_DIRECTORIES:
        raise ValueError("split must be 'train' or 'test'")
    directory = Path(data_root) / SNIPPET_DIRECTORIES[split]
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing VIEW360 {split} I3D directory: {directory}")
    return directory


def temporal_segments(length: int, segments: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("Cannot segment an empty feature sequence")
    return np.linspace(0, length, segments + 1, dtype=np.int64)


def sample_snippet_features(features: np.ndarray, segments: int = 32) -> np.ndarray:
    """Mean-pool a variable-length snippet sequence into fixed temporal segments."""

    boundaries = temporal_segments(len(features), segments)
    sampled = np.empty((segments, features.shape[1]), dtype=np.float32)
    for index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        sampled[index] = (
            features[start:end].mean(axis=0) if start != end else features[start]
        )
    return sampled


def align_frame_features(
    features: np.ndarray,
    snippet_count: int,
    frames_per_snippet: int = FRAMES_PER_SNIPPET,
) -> np.ndarray:
    """Apply the temporal alignment used to train the released FDPN checkpoint."""

    target_length = snippet_count * frames_per_snippet
    if len(features) == 0:
        raise ValueError("Frame feature sequence is empty")
    aligned = np.asarray(features, dtype=np.float32)

    if len(aligned) < target_length:
        padding = np.repeat(aligned[0:1], target_length - len(aligned), axis=0)
        aligned = np.concatenate((padding, aligned), axis=0)

    # Preserve the original release's outlier handling for the two sequences
    # whose extracted frame features are longer than their I3D timeline.
    while len(aligned) > target_length:
        difference = len(aligned) - target_length
        aligned = np.delete(aligned, [-1, difference], axis=0)
    if len(aligned) < target_length:
        padding = np.repeat(aligned[0:1], target_length - len(aligned), axis=0)
        aligned = np.concatenate((padding, aligned), axis=0)
    return aligned


def group_frame_features(
    aligned_features: np.ndarray,
    snippet_count: int,
    frames_per_snippet: int = FRAMES_PER_SNIPPET,
) -> np.ndarray:
    """Group aligned frame features according to the I3D temporal offset."""

    first_window = aligned_features[:frames_per_snippet]
    grouped = np.empty(
        (snippet_count, frames_per_snippet, aligned_features.shape[1]),
        dtype=np.float32,
    )
    for index in range(snippet_count):
        if index <= 1:
            grouped[index] = first_window
        else:
            start = frames_per_snippet * (index - 2)
            grouped[index] = aligned_features[start : start + frames_per_snippet]
    return grouped


def sample_frame_features(
    aligned_features: np.ndarray,
    original_snippets: int,
    segments: int = 32,
    frames_per_snippet: int = FRAMES_PER_SNIPPET,
) -> np.ndarray:
    """Select frame windows matching fixed-length training snippet segments."""

    grouped = group_frame_features(
        aligned_features, original_snippets, frames_per_snippet
    )
    boundaries = temporal_segments(original_snippets, segments)
    sampled = np.empty(
        (segments, frames_per_snippet, aligned_features.shape[1]), dtype=np.float32
    )
    for index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        source = 0 if start <= 1 or end <= 1 else start
        sampled[index] = grouped[source]
    return sampled


class VIEW360Dataset(Dataset[dict[str, object]]):
    """Load pre-extracted VIEW360 I3D and masked-frame features.

    The manifest order is preserved for evaluation because the concatenated
    frame ground truth follows that exact order. For training, ``normal`` can
    select one side of a paired normal/abnormal loader without hard-coded
    split indices.
    """

    def __init__(
        self,
        data_root: str | Path,
        manifest: str | Path,
        split: str,
        frame_feature_directory: str | None = None,
        saliency_map_directory: str | None = None,
        normal: bool | None = None,
        segments: int = 32,
        frames_per_snippet: int = FRAMES_PER_SNIPPET,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        self.data_root = Path(data_root)
        self.split = split
        self.snippet_directory = SNIPPET_DIRECTORIES[split]
        self.frame_feature_directory = frame_feature_directory
        self.saliency_map_directory = saliency_map_directory
        self.segments = segments
        self.frames_per_snippet = frames_per_snippet

        entries = read_manifest(manifest)
        if normal is not None:
            entries = [
                entry for entry in entries if entry.startswith("Normal_") is normal
            ]
        if not entries:
            raise ValueError("No samples remain after filtering the manifest")
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def _load(self, directory: str, filename: str) -> np.ndarray:
        path = self.data_root / directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing VIEW360 feature: {path}")
        return np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)

    def __getitem__(self, index: int) -> dict[str, object]:
        entry = self.entries[index]
        video_name, _ = parse_feature_filename(entry)
        snippets = self._load(self.snippet_directory, entry)
        if snippets.ndim != 3 or snippets.shape[1:] != (10, FEATURE_SIZE):
            raise ValueError(
                f"Unexpected I3D shape for {entry}: {tuple(snippets.shape)}"
            )
        original_snippets = len(snippets)
        snippets_by_crop = snippets.transpose(1, 0, 2)
        if self.split == "train":
            snippets_by_crop = np.stack(
                [
                    sample_snippet_features(crop, self.segments)
                    for crop in snippets_by_crop
                ]
            )

        sample: dict[str, object] = {
            "name": entry,
            "snippet_features": snippets_by_crop,
            "label": np.float32(not entry.startswith("Normal_")),
        }

        if self.frame_feature_directory is not None:
            frame_filename = f"{video_name}.npy"
            frame_features = self._load(self.frame_feature_directory, frame_filename)
            if frame_features.ndim != 2 or frame_features.shape[1] != FEATURE_SIZE:
                raise ValueError(
                    f"Unexpected frame feature shape for {frame_filename}: "
                    f"{tuple(frame_features.shape)}"
                )
            aligned = align_frame_features(
                frame_features, original_snippets, self.frames_per_snippet
            )
            sample["frame_features"] = (
                sample_frame_features(
                    aligned,
                    original_snippets,
                    self.segments,
                    self.frames_per_snippet,
                )
                if self.split == "train"
                else group_frame_features(
                    aligned, original_snippets, self.frames_per_snippet
                )
            )

        if self.saliency_map_directory is not None:
            map_filename = f"{video_name}.npy"
            saliency_maps = self._load(self.saliency_map_directory, map_filename)
            if saliency_maps.ndim != 2 or saliency_maps.shape[1] != 9:
                raise ValueError(
                    f"Unexpected saliency-map shape for {map_filename}: "
                    f"{tuple(saliency_maps.shape)}"
                )
            aligned_maps = align_frame_features(
                saliency_maps, original_snippets, self.frames_per_snippet
            )
            sample["saliency_maps"] = (
                sample_frame_features(
                    aligned_maps,
                    original_snippets,
                    self.segments,
                    self.frames_per_snippet,
                )
                if self.split == "train"
                else group_frame_features(
                    aligned_maps, original_snippets, self.frames_per_snippet
                )
            )
        return sample


def iter_video_lengths(entries: Sequence[str]) -> Iterator[tuple[str, int]]:
    for entry in entries:
        _, frame_count = parse_feature_filename(entry)
        yield entry, frame_count
