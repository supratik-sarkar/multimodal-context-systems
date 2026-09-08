
# Signature Extraction & Similarity (ResNet-101)

**Goal.** Detect and extract handwritten signatures from scanned documents, then perform similarity verification
(genuine vs forged). We use synthetic training (signatures overlaid on open documents) + public signature corpora.

## Datasets (public)
- **CEDAR Signature** (offline signatures) — for verification (genuine/forged).
- **RVL-CDIP** (document images) — for backgrounds; we create synthetic "signed documents" to train a detector.
- Fallback: any Kaggle signature dataset for quick start.

## Pipeline
1. **Synthesis**: overlay cropped signatures onto document backgrounds with random placement → YOLO-format bboxes.
2. **Detection**: train a detector to localize signatures.
3. **Extraction**: crop detections, normalize, embed with ResNet-101, metric learning with triplet loss.
4. **Verification**: compute cosine similarity; ROC/AUC & EER.

## Quickstart
```bash
make init
python -m src.data.synthesize --out data/synth --n 5000
python -m src.detect.train --epochs 2
python -m src.verify.train --epochs 2
```

## Citations
- He et al., 2016 (ResNet).
- Signature verification literature: Hafemann et al., 2017.
