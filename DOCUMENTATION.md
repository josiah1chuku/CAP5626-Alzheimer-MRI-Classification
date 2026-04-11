# CAP5626 Project — Code and Workflow Documentation
## Deep Learning for Five-Class Alzheimer's Disease Classification
## Student: Josiah Chuku | josiah1.chuku@famu.edu
## Florida A&M University | Spring 2026

---

## 1. Repository Structure

CAP5626-Alzheimer-MRI-Classification/
├── .github/
│   ├── workflows/
│   │   ├── pipeline.yml       # CI/CD pipeline
│   │   └── security.yml       # Security scanning
│   ├── dependabot.yml         # Dependency updates
│   └── SECURITY.md            # Security policy
├── alzheimer_classification.ipynb  # Main notebook
├── requirements.txt                # Dependencies
├── README.md                       # Setup instructions
└── DOCUMENTATION.md               # This file

---

## 2. Notebook Structure (12 Cells)

| Cell | Purpose | Key Libraries |
|------|---------|---------------|
| Cell 1  | GPU check + Drive mount | torch, google.colab |
| Cell 2  | Extract + rename dataset | zipfile, shutil, pathlib |
| Cell 3  | All imports + global config | torch, timm, sklearn, matplotlib |
| Cell 4  | Dataset visualization (3 figures) | matplotlib, pathlib |
| Cell 5  | Preprocessing study (4 pipelines) | torchvision, timm |
| Cell 6  | Transforms + patient-level DataLoaders | torchvision, re, collections |
| Cell 7  | Model definitions (5 architectures) | torch.nn, timm |
| Cell 8  | Training function | torch.optim, ReduceLROnPlateau |
| Cell 9  | Train all 5 models | All above |
| Cell 10 | Evaluate all 5 models | sklearn.metrics |
| Cell 11 | Training curves + per-class F1 | matplotlib, seaborn, json |
| Cell 12 | Final results summary | pandas |

---

## 3. Dataset

- Source: ADNI via Dr. Carlos Theran (FAMU CAP5626)
- Format: 2D axial MRI slices JPEG
- Classes: AD (145), CN (493), EMCI (204), LMCI (61), MCI (198)
- Train: 1,101 | Test: 195
- Patient-level split: 489 unique patients, overlap = 0

---

## 4. Preprocessing Pipeline (Pipeline 2 — Selected)

Four pipelines were empirically compared:

| Pipeline | Configuration | Val Acc |
|----------|---------------|---------|
| P1 | No normalization, grayscale, 224px | 0.424 |
| P2 | Grayscale + normalize, 224px | 0.436 SELECTED |
| P3 | RGB + normalize, 224px | 0.412 |
| P4 | Grayscale + normalize, 128px | 0.333 |

Selected steps:
1. Grayscale conversion
2. Resize to 224x224
3. ImageNet normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
4. Axial slice format
5. RGB channel replication

---

## 5. Data Augmentation (7 Techniques)

All techniques from project handout implemented:

| # | Technique | Code | Justification |
|---|-----------|------|---------------|
| 1 | Random crop | RandomCrop(224) | FOV differences across sites |
| 2 | Small rotation | RandomRotation(15) | Head positioning variation |
| 3 | Horizontal flip | RandomHorizontalFlip(p=0.5) | Bilateral brain symmetry |
| 4 | Contrast adjust | ColorJitter(b=0.2, c=0.2) | Inter-scanner variability |
| 5 | Width shift | translate=(0.05, 0.05) | Head movement simulation |
| 6 | Height shift | translate=(0.05, 0.05) | Head movement simulation |
| 7 | Zooming | scale=(0.9, 1.1) | Scanner zoom variability |
| 8 | Light affine | shear=5 | Positioning differences |

Without augmentation: Macro F1 = 0.093
With augmentation:    Macro F1 = 0.320 (+244%)

---

## 6. Model Architectures

### Architecture 1 — Custom CNN (from scratch)
- 3 convolutional blocks
- Batch normalization + Dropout (0.25-0.50)
- AdaptiveAvgPool + FC classifier
- Total params: 174,373
- Reference: Theran handout Section 3.4

### Architecture 2 — VGG-16 (transfer learning)
- Pretrained on ImageNet via timm
- Unfrozen: head + pre_logits
- Trainable params: 119,566,341
- Reference: Simonyan & Zisserman 2014; Theran slide 5

### Architecture 3 — ResNet-50 (transfer learning)
- Pretrained on ImageNet via timm
- Unfrozen: layer4 + fc
- Trainable params: 14,974,981
- Reference: He et al. 2016; Theran slide 7

### Architecture 4 — DenseNet121 (transfer learning) BEST MODEL
- Pretrained on ImageNet via timm
- Unfrozen: denseblock4 + classifier
- Trainable params: 2,163,205
- Reference: Huang et al. 2017; Theran slide 8

### Architecture 5 — EfficientNet-B0 (transfer learning)
- Pretrained on ImageNet via timm
- Unfrozen: blocks.6 + classifier
- Trainable params: 723,637
- Reference: Tan & Le 2019; Theran slide 10

---

## 7. Training Configuration

| Hyperparameter | Custom CNN | Transfer Models |
|----------------|------------|-----------------|
| Optimizer | Adam | Adam |
| Learning rate | 1e-3 | 5e-5 |
| Weight decay | 1e-4 | 1e-4 |
| Batch size | 16 | 16 |
| Max epochs | 50 | 50 |
| Loss function | CrossEntropyLoss | CrossEntropyLoss |
| LR scheduler | ReduceLROnPlateau | ReduceLROnPlateau |
| Early stopping | patience=10 | patience=10 |
| Class balance | WeightedRandomSampler | WeightedRandomSampler |
| Hardware | NVIDIA T4 GPU (Google Colab) | NVIDIA T4 GPU |

---

## 8. Overfitting Mitigation (7 Strategies)

| # | Strategy | Implementation |
|---|----------|---------------|
| 1 | Dropout | 0.25-0.50 in Custom CNN |
| 2 | Batch Normalization | After every Conv2d |
| 3 | Early Stopping | patience=10, best checkpoint |
| 4 | LR Scheduling | ReduceLROnPlateau factor=0.5 |
| 5 | L2 Regularization | weight_decay=1e-4 |
| 6 | Reduced Complexity | Custom CNN 174K params |
| 7 | WeightedRandomSampler | w_c = 1/n_c |

Overfitting gaps:
- Custom CNN:      -0.247 (underfitting)
- VGG-16:          0.427  (severe)
- ResNet-50:       0.107  (mild)
- DenseNet121:     0.261  (moderate)
- EfficientNet-B0: 0.221  (moderate)

---

## 9. Final Results

| Model | Test Acc | Macro F1 | ROC-AUC | PR-AUC |
|-------|----------|----------|---------|--------|
| Custom CNN | 0.4308 | 0.1307 | 0.5241 | 0.2315 |
| VGG-16 | 0.4308 | 0.3101 | 0.6248 | 0.3320 |
| ResNet-50 | 0.2615 | 0.2287 | 0.5908 | 0.2727 |
| DenseNet121 BEST | 0.3436 | 0.3201 | 0.6535 | 0.4002 |
| EfficientNet-B0 | 0.3282 | 0.2726 | 0.5906 | 0.2914 |

Evaluation metrics used:
- Accuracy, Precision, Recall, F1-score
- Macro F1, Weighted F1
- ROC-AUC (macro one-vs-rest)
- PR-AUC
- Confusion matrices

---

## 10. GitHub Actions Workflows

### pipeline.yml — CI Pipeline
Triggers: push to main, pull request, manual
Steps:
1. Set up Python 3.12
2. Cache pip packages
3. Install all dependencies
4. Verify torch, timm, sklearn installations
5. Convert notebook to script
6. Smoke test — forward pass all architectures
7. Upload notebook + requirements as artifacts

### security.yml — Security Scanning
Triggers: push to main, every Monday 6AM
Jobs:
1. pip-audit — scan dependencies for CVEs
2. safety — check against vulnerability database
3. bandit — scan Python code for security issues
4. detect-secrets — scan for accidentally committed secrets

### dependabot.yml — Dependency Updates
- Weekly scan of pip dependencies
- Weekly scan of GitHub Actions versions
- Auto creates PRs for security updates

---

## 11. Key References

1. Lim et al. (2022). Deep learning-based classification of
   Alzheimer's disease using brain MRI data.
   Frontiers in Aging Neuroscience, 14, 876202.

2. He et al. (2016). Deep residual learning for image recognition.
   IEEE CVPR, pp. 770-778.

3. Huang et al. (2017). Densely connected convolutional networks.
   IEEE CVPR, pp. 4700-4708.

4. Simonyan & Zisserman (2014). Very deep convolutional networks
   for large-scale image recognition. ICLR 2015.

5. Tan & Le (2019). EfficientNet: Rethinking model scaling for
   convolutional neural networks. ICML, pp. 6105-6114.

6. Litjens et al. (2017). A survey on deep learning in medical
   image analysis. Medical Image Analysis, 42, 60-88.

7. Tajbakhsh et al. (2016). Convolutional neural networks for
   medical image analysis. IEEE TMI, 35(5), 1299-1312.

8. Johnson & Khoshgoftaar (2019). Survey on deep learning with
   class imbalance. Journal of Big Data, 6(1), 27.

9. Theran, C. (2026). Evaluation metrics for ML and DL.
   Lecture Notes CAP5626, FAMU.

10. Theran, C. (2026). CNN architectures.
    Lecture Notes CAP5626, FAMU.

---

## 12. How to Run

1. Open alzheimer_classification.ipynb in Google Colab
2. Runtime -> Change runtime type -> T4 GPU
3. Runtime -> Run all
4. Total runtime: approximately 40 minutes

Requirements: See requirements.txt
Hardware: NVIDIA T4 GPU (Google Colab free tier)
