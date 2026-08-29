# VIEW360 metadata

This directory contains only split metadata and evaluation ground truth. The
videos and extracted features are distributed with VIEW360 separately.

| File | Contents |
| --- | --- |
| `view360_train_i3d.txt` | 375 training feature basenames (194 abnormal, 181 normal) |
| `view360_test_i3d.txt` | 200 test feature basenames (105 abnormal, 95 normal) |
| `view360_temporal_annotations.txt` | Inclusive, one-indexed anomaly intervals and direction codes |
| `view360_frame_ground_truth.npy` | 153,824 binary frame labels in test-manifest order |
| `view360_direction_ground_truth.npy` | Direction IDs in test-manifest order |

Direction IDs are `0 = left back`, `1 = center`, `2 = right back`, and
`3 = background/unannotated`. Direction accuracy is evaluated only where the
direction ground truth is not `3`.

[`VIEW360_features.zip`](https://1drv.ms/u/c/9121b298b1d2b494/IQACfQxRh48uQZ9jaOr7XB4kAbmoOcDI62QqxObOLDqAIXE?e=9FijPG)
contains exactly the features used by the released training and evaluation
code. Extract it with:

```bash
unzip VIEW360_features.zip -d /path/to/datasets
```

The extracted layout is:

```text
VIEW360/
├── view360_train_i3d/  # [snippets, 10 crops, 2048]
├── view360_test_i3d/   # [snippets, 10 crops, 2048]
└── view360_masked_resnet/ # masked-frame ResNet features [frames, 2048]
```

Regenerate all split files and ground-truth arrays from an extracted VIEW360
directory:

```bash
python prepare_view360.py --data-root /path/to/VIEW360
```

The generated arrays must be kept in the exact test-manifest order. All
metadata and feature filenames use the paper category name `Teasing`.
