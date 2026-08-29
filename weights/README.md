# Model weights and predictions

Large checkpoints and prediction arrays are intentionally excluded from Git.
Download [`VIEW360_weights.zip`](https://1drv.ms/u/c/9121b298b1d2b494/IQD62kKCriOzQqh-oQJ2ckumAUo6gqH5pG9RsjxZ55_he8Y?e=PoFBjO)
and extract it at the repository root:

```bash
unzip VIEW360_weights.zip
```

The archive creates `weights/` and contains exactly these five files:

| File | Purpose |
| --- | --- |
| `fdpn_view360.pth` | RTFM-based FDPN checkpoint |
| `mgfn_view360.pth` | MGFN baseline checkpoint |
| `fdpn_view360_scores.npy` | FDPN frame anomaly scores |
| `mgfn_view360_scores.npy` | MGFN frame anomaly scores |
| `fdpn_view360_direction_predictions.npy` | FDPN direction class IDs |

The prediction arrays can be evaluated without a GPU:

```bash
python evaluate.py artifacts
```
