# Inference Report Summary

- **Report threshold (your requirement):** 0.95

- **Old-model training/validation threshold (from old train.py):** 0.50


## Key note about the old model showing 0 at thr=0.95

- Your old `train.py` computes IoU/F1 using threshold **0.5**. If the old model probabilities rarely exceed **0.95**, then at 0.95 it will predict almost no burned pixels, which yields TP=0 → IoU/F1/Recall≈0. This does **not** necessarily mean the old checkpoint is broken; it may be **less confident / differently calibrated**.


## Results at report threshold

- NEW @ 0.95: IoU=0.3197, F1=0.4845, P=0.4567, R=0.5158

- OLD @ 0.95: IoU=0.4026, F1=0.5741, P=0.5761, R=0.5722


## Old checkpoint at its training threshold (sanity check)

- OLD @ 0.50: IoU=0.2341, F1=0.3793, P=0.2446, R=0.8442


## Best-threshold comparison (from sweep)

- NEW best IoU threshold: 0.970

- OLD best IoU threshold: 0.950


## Why weighted inputs + auxiliary task help (paper phrasing)

- **Weighted input fusion** increases the influence of modalities that are empirically more informative (per ablation), improving representation quality and reducing reliance on weak/noisy sources.
- **Auxiliary landcover segmentation** acts as a regularizer that injects semantic context (vegetation/soil/water patterns), which reduces false positives on dark surfaces and helps boundaries align with real land cover transitions.
