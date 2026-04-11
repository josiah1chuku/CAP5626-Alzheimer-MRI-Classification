[![CI Pipeline](https://github.com/josiah1chuku/CAP5626-Alzheimer-MRI-Classification/actions/workflows/pipeline.yml/badge.svg)](https://github.com/josiah1chuku/CAP5626-Alzheimer-MRI-Classification/actions/workflows/pipeline.yml)

# CAP5626 - Deep Learning for Five-Class Alzheimer's Disease Classification
## Florida A&M University - Spring 2026
### Student: [Josiah Chuku] | [josiah1.chuku@famu.edu]

---

## Project Overview

Complete deep learning pipeline for five-class classification of structural
MRI brain images into: AD, CN, EMCI, LMCI, and MCI.

Five CNN architectures compared:
1. Custom CNN (from scratch)
2. VGG-16 (transfer learning)
3. ResNet-50 (transfer learning)
4. DenseNet121 (transfer learning) <- Best model
5. EfficientNet-B0 (transfer learning)

---

## Best Results (DenseNet121)

| Metric      | Value  |
|-------------|--------|
| ROC-AUC     | 0.6535 |
| Macro F1    | 0.3201 |
| PR-AUC      | 0.4002 |
| LMCI Recall | 0.5455 |

---

## How to Run

### Requirements
- Google Colab with T4 GPU (free tier)
- Google Drive account
- Python 3.12

### Steps

1. Open notebook in Google Colab
2. Runtime -> Change runtime type -> T4 GPU -> Save
3. Run all cells in order (Runtime -> Run all)

The notebook will:
- Mount Google Drive
- Extract and prepare the dataset automatically
- Run preprocessing study (4 pipelines, ~5 min)
- Train all 5 CNN architectures (~35 min)
- Evaluate with full metrics and save all figures

Total runtime: ~40 minutes on T4 GPU

---

## Dataset

- Source: ADNI (Alzheimer's Disease Neuroimaging Initiative)
- Provider: Dr. Carlos Theran, CAP5626, FAMU
- Format: 2D axial MRI slices, JPEG
- Classes: AD (145), CN (493), EMCI (204), LMCI (61), MCI (198)
- Train: 1,101 images | Test: 195 images
- Imbalance: CN:LMCI = 8.1:1

---

## Project Structure

CAP5626-Alzheimer-MRI-Classification/
- alzheimer_classification.ipynb  # Main Colab notebook
- requirements.txt                # Python dependencies
- README.md                       # This file

---

## Key Pipeline

Preprocessing (Pipeline 2 selected from 4-pipeline study):
Grayscale, Resize 224x224, ImageNet normalization,
Axial slice format, RGB channel replication

Augmentation (7 techniques - all handout examples):
Random crop, rotation +/-15, horizontal flip,
color jitter, width/height shift, zoom, affine shear

Overfitting Mitigation (7 strategies):
Dropout, batch normalization, early stopping,
LR scheduling, L2 regularization, reduced complexity,
WeightedRandomSampler

---

## Dependencies

See requirements.txt for exact versions.
Key: torch==2.10.0+cu128, timm==1.0.26,
scikit-learn==1.6.1, matplotlib==3.10.0

---

## Reference

Lim et al. (2022). Deep learning-based classification of
Alzheimer's disease using brain MRI data.
Frontiers in Aging Neuroscience, 14, 876202.
