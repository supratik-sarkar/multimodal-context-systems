
# StackGAN (Repro) — Text/Audio → Image

**Goal.** Reproduce a modernized StackGAN-style pipeline that conditions on captions (and optional spoken captions)
to generate images. Baselines on **CUB-200-2011 (with captions)** and **Oxford-102 Flowers (with captions)**.
Optional audio modality via **SpokenCOCO** to showcase multimodal conditioning.

## Datasets (public)
- CUB-200-2011 + captions (Reed et al.).
- Oxford-102 Flowers + captions (Reed et al.).
- SpokenCOCO (audio captions aligned to MSCOCO images) — used here for demonstrating audio-text fusion in the conditional encoder.

## Key Features
- Clean repo structure, MLflow tracking, Lightning training loops.
- Text encoder: GloVe embeddings or trainable Transformers (configurable).
- Optional audio encoder (log-mel spectrogram + small CNN/GRU).
- FID & IS evaluation scripts.

## Quickstart
```bash
make init
python -m src.data.downloads --dataset cub
python -m src.train --cfg configs/cub_stackgan.yaml --experiment stackgan-cub
```

## Citations
- StackGAN: Han Zhang et al., 2017.
- Reed et al., 2016 (text embeddings for image synthesis).
- FID: Heusel et al., 2017; Inception Score: Salimans et al., 2016.
