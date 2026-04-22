# Pixel-Level Detection and Segmentation of Copy-Move Forgeries in Biomedical Scientific Images

**CS 6140 — Machine Learning Final Project**
**Authors:** Seshadri Veeraraghavan Vidyalakshmi, Jaisweta Naarrayanan

---

## Overview

This project detects and localizes copy-move forgeries in biomedical
scientific images using three models:

| Model | Type | Task |
|---|---|---|
| Model 1 — SVM (RBF) | Classical ML | Image-level classification |
| Model 2 — Random Forest | Classical ML | Image-level classification |
| Model 3 — EfficientNet-B0 + ResNet34/UNet | Deep Learning | Pixel-level segmentation |

---

## Dataset

Download from Kaggle:
[Recod.ai/LUC Scientific Image Forgery Detection](https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection)

Place the downloaded data in the following structure: