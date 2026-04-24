# Scientific Image Forgery Detection

Pixel-level detection and segmentation of copy-move forgeries in biomedical scientific images.

**CS 6140 — Machine Learning Final Project**  
1. Seshadri Veeraraghavan Vidyalakshmi (NUID: 002519014)
2. Jaisweta Naarrayanan (NUID: 002553152)

---

## Overview

Three models are compared on the [Recod.ai/LUC Scientific Image Forgery Detection](https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection) dataset:

| Model | Approach | Task |
|---|---|---|
| Model 1 — SVM (RBF) | Classical ML, hand-crafted features | Image-level classification |
| Model 2 — Random Forest | Classical ML, same features as SVM | Image-level classification |
| Model 3 — EfficientNet-B0 + ResNet-34/UNet | Deep learning, two-stage gated pipeline | Pixel-level segmentation |

---

## Project Structure

```
scientific-image-forgery-detection/
├── data/                        # dataset
│   ├── train_images/
│   │   ├── authentic/
│   │   └── forged/
│   ├── train_masks/             # .npy files, shape (1, H, W)
│   ├── supplemental_images/
│   └── test_images/
├── notebooks/
│   └── model3.ipynb             # Kaggle notebook for Model 3
├── outputs/
│   ├── plots/                   # all PNG outputs
│   ├── predictions/             # all CSV outputs
│   └── models/                  # saved models
│       ├── gate_best.pth        # trained gate 
│       └── segmenter_best.pth   # trained segmenter 
│   └── svm_features.npz         # cached feature matrix for SVM & RF 
├── src/
│   ├── config.py                # hyperparameters and paths
│   ├── features.py              # shared feature pipeline for Models 1 & 2
│   ├── svm.py                   # Model 1
│   └── random_forest.py         # Model 2
├── model3_train.py              # Model 3 local training script
├── inference.py                 # run predictions on new images
├── requirements.txt
└── pyproject.toml
```

---

## Setup

**Requirements:** Python 3.11+, [uv](https://github.com/astral-sh/uv)

```bash
git clone https://github.com/YOUR_USERNAME/scientific-image-forgery-detection.git
cd scientific-image-forgery-detection

uv venv
source .venv/bin/activate

uv pip install -r requirements.txt
uv pip install segmentation_models_pytorch-0.5.0-py3-none-any.whl
```

---

## Reproducing Results

### Model 1 — SVM

```bash
python -m src.svm
```

Extracts features, runs 5-fold GridSearchCV, evaluates on test set.  
Outputs will be saved to `outputs/plots/svm_confusion_matrix.png`, `outputs/predictions/svm_predictions.csv`

> Features are cached to `outputs/svm_features.npz` after the first run.

---

### Model 2 — Random Forest

```bash
python -m src.random_forest
```

Reuses the cached feature matrix from Model 1 — no re-extraction needed.  
Outputs will be saved to `outputs/plots/rf_confusion_matrix.png`, `outputs/plots/rf_feature_importance.png`, `outputs/predictions/rf_predictions.csv`

---

### Model 3 — Gated Segmentation Network

Model 3 was trained on Kaggle T4 GPU. Two options to reproduce:

**Option A — Kaggle (recommended)**

1. Upload `notebooks/model3.ipynb` to Kaggle
2. Attach your dataset under **Add Input**
3. Enable **GPU T4** accelerator
4. Run all cells top to bottom
5. Download `gate_best.pth` and `segmenter_best.pth` from the output panel
6. Place them in `outputs/`

**Option B — Local**

```bash
# Train both stages
python model3_train.py

# Train gate only
python model3_train.py --gate-only

# Train segmenter only
python model3_train.py --seg-only

# Resume after interruption
python model3_train.py --resume-gate
python model3_train.py --resume-seg
```

> Local training requires MPS (Apple Silicon) or CUDA. CPU is supported but slow.

---

### Inference

Once model weights are in `outputs/`, run inference on any image or folder:

```bash
# Single image
python inference.py --image data/train_images/forged/10.png

# Folder
python inference.py --folder data/train_images/forged/

# With ground truth evaluation
python inference.py --folder data/train_images/forged/ --masks data/train_masks/
```

Results are saved to `outputs/inference/` as side-by-side PNG visualisations.

---

## Results

### Models 1 & 2 — Image-Level Classification

| Model | F1 | ROC-AUC | Accuracy |
|---|---|---|---|
| SVM (RBF) | ~0.55 | ~0.55 | ~53% |
| Random Forest | ~0.55 | ~0.55 | ~53% |

Both models plateau at ~53% — classical similarity statistics cannot overcome background texture repetition in biomedical images. This is the expected result and motivates Model 3.

### Model 3 — Pixel-Level Segmentation

| Stage | Metric | Score |
|---|---|---|
| Gate (EfficientNet-B0) | Val F1 | 0.74 |
| Segmenter (ResNet-34/UNet) | Val Dice | 0.43 |

The gate correctly identifies forged images at 74% F1. The segmenter achieves a Dice score of 0.43, performing well on large forged regions but struggling with small ones — a known limitation of this dataset's scale variability.

---

## Key Design Decisions

**Why a gated pipeline?**  
Authentic images skip segmentation entirely. This reduces false positives and saves compute at inference.

**Why BCE + Dice loss?**  
Forged pixels are only 2–5% of each image. BCE alone causes the model to predict all background. Dice forces it to cover forged regions regardless of class imbalance.

**Why features are cached**  
Extracting HOG + LBP + DCT descriptors for 2400 images takes ~15 minutes. The cached `.npz` lets you re-run Model 2 or tune hyperparameters without repeating extraction.

---

## Configuration

All hyperparameters live in `src/config.py`. Change values there — no other file needs to be touched.

Key settings:

```python
SAMPLES_PER_CLASS = 1200   # images per class for SVM/RF
GATE_NUM_EPOCHS   = 20     # gate training epochs
SEG_NUM_EPOCHS    = 30     # segmenter training epochs
GATE_LR           = 1e-4   # gate learning rate
SEG_LR            = 3e-4   # segmenter learning rate
BCE_WEIGHT        = 0.5    # loss weighting
DICE_WEIGHT       = 0.5
```

---

## Dependencies

```
numpy, pandas, scikit-learn, scikit-image
opencv-python, scipy, matplotlib
torch, torchvision
albumentations
segmentation-models-pytorch==0.5.0
tqdm
```
