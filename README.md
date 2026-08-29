<div align="center">

[<img src="https://img.shields.io/badge/🌐_Project_Page-VIEW360-red?style=for-the-badge" alt="Project Page" height="35">](https://songinpyo.github.io/VIEW360-Project/)
[<img src="https://img.shields.io/badge/📄_Paper-WACV-blue?style=for-the-badge" alt="Paper" height="35">](https://openaccess.thecvf.com/content/WACV2025/papers/Song_Anomaly_Detection_for_People_with_Visual_Impairments_using_an_Egocentric_WACV_2025_paper.pdf)
[<img src="https://img.shields.io/badge/📊_Features-Download-brightgreen?style=for-the-badge" alt="Features" height="35">](https://1drv.ms/u/c/9121b298b1d2b494/IQACfQxRh48uQZ9jaOr7XB4kAbmoOcDI62QqxObOLDqAIXE?e=9FijPG)
[<img src="https://img.shields.io/badge/⚖️_Weights-Download-orange?style=for-the-badge" alt="Weights" height="35">](https://1drv.ms/u/c/9121b298b1d2b494/IQD62kKCriOzQqh-oQJ2ckumAUo6gqH5pG9RsjxZ55_he8Y?e=PoFBjO)
[<img src="https://img.shields.io/badge/🔬_Lab-i2slab-darkgreen?style=for-the-badge" alt="Lab" height="35">](https://i2slab.skku.edu/)

</div>

# VIEW360

Official implementation of **“Anomaly Detection for People with Visual
Impairments Using an Egocentric 360-Degree Camera”** (WACV 2025).

![VIEW360](Figures/Front_Image.png)

## Abstract

Recent advancements in computer vision have led to a renewed interest in
developing assistive technologies for individuals with visual impairments.
Although extensive research has been conducted in the field of computer
vision-based assistive technologies, most of the focus has been on
understanding contexts in images, rather than addressing their physical safety
and security concerns. To address this challenge, we propose the first step
towards detecting anomalous situations for visually impaired people by
observing their entire surroundings using an egocentric 360-degree camera. We
first introduce a novel egocentric 360-degree video dataset called VIEW360
(Visually Impaired Equipped with Wearable 360-degree camera), which contains
abnormal activities that visually impaired individuals may encounter, such as
shoulder surfing and pickpocketing. Furthermore, we propose a new architecture
called the FDPN (Frame and Direction Prediction Network), which facilitates
frame-level prediction of abnormal events and identifying of their directions.
Finally, we evaluate our approach on our VIEW360 dataset and the publicly
available UCF-Crime and Shanghaitech datasets, demonstrating state-of-the-art
performance.

## News

- (2026.08) The source code, pre-extracted features, and model weights are available.
- (2024.10) The paper was accepted by **WACV 2025**.

## Results on VIEW360

| Model | ROC-AUC | PR-AUC | Direction accuracy |
| --- | ---: | ---: | ---: |
| MGFN | 80.43 | 20.16 | – |
| FDPN | 86.00 | 26.97 | 75.04 |

Results are reported from Table 2 of the paper.

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/Songinpyo/VIEW360.git
cd VIEW360
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the CUDA-enabled PyTorch build appropriate for your system when using
a GPU.

## Data and artifacts

Download [`VIEW360_features.zip`](https://1drv.ms/u/c/9121b298b1d2b494/IQACfQxRh48uQZ9jaOr7XB4kAbmoOcDI62QqxObOLDqAIXE?e=9FijPG),
extract it at any location, and pass the extracted top-level `VIEW360`
directory with `--data-root`:

```bash
unzip VIEW360_features.zip -d /path/to/datasets
```

```text
VIEW360/
├── view360_train_i3d/       # [snippets, 10 crops, 2048]
├── view360_test_i3d/        # [snippets, 10 crops, 2048]
└── view360_masked_resnet/   # saliency-masked frame features [frames, 2048]
```

All public filenames use the paper category name `Teasing`. Only pre-extracted
features are distributed in this archive; raw VIEW360 videos are not included.
Split metadata and ground truth are versioned in
[`data/`](data/README.md).

Download [`VIEW360_weights.zip`](https://1drv.ms/u/c/9121b298b1d2b494/IQD62kKCriOzQqh-oQJ2ckumAUo6gqH5pG9RsjxZ55_he8Y?e=PoFBjO)
and extract it at the repository root. It contains the model checkpoints and
prediction arrays under `weights/`:

```bash
unzip VIEW360_weights.zip
```

The complete artifact list is documented in
[`weights/README.md`](weights/README.md).

To validate an extracted archive and regenerate its metadata:

```bash
python prepare_view360.py --data-root /path/to/VIEW360
```

## Evaluation

Evaluate the provided prediction arrays without a GPU:

```bash
python evaluate.py artifacts
```

Run inference from the provided checkpoints:

```bash
python evaluate.py inference \
  --model mgfn \
  --checkpoint weights/mgfn_view360.pth \
  --data-root /path/to/VIEW360

python evaluate.py inference \
  --model fdpn \
  --checkpoint weights/fdpn_view360.pth \
  --data-root /path/to/VIEW360
```

The FDPN frame branch combines RTFM snippet features with ResNet features
extracted from saliency-masked frames.

## Training

The paper's coarse-to-fine procedure is exposed as two stages. First train the
RTFM snippet network, then freeze it and train the frame prediction network:

```bash
python train.py \
  --stage snippet \
  --data-root /path/to/VIEW360 \
  --output-directory runs/view360_snippet

python train.py \
  --stage frame \
  --checkpoint runs/view360_snippet/best_model.pth \
  --data-root /path/to/VIEW360 \
  --output-directory runs/view360_fdpn
```

## Repository layout

```text
datasets/view360.py   VIEW360 loading, alignment, and ground-truth generation
models/fdpn.py        RTFM-based FDPN implementation
models/mgfn.py        MGFN baseline implementation
train.py              coarse-to-fine training entry point
evaluate.py           prediction metrics and checkpoint inference
prepare_view360.py    feature validation and metadata generation
```

## Citation

```bibtex
@InProceedings{Song_2025_WACV,
    author    = {Song, Inpyo and Lee, Sanghyeon and Joo, Minjun and Lee, Jangwon},
    title     = {Anomaly Detection for People with Visual Impairments using an Egocentric 360-Degree Camera},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {2828-2837},
    doi       = {10.1109/WACV61041.2025.00280}
}
```
