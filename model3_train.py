"""
Model 3 — Gated Segmentation Network
Copy-Move Forgery Detection in Biomedical Scientific Images
CS 6140 — Machine Learning Final Project
---------------------
Trains a two-stage deep learning pipeline to detect copy-move forgeries:

  Stage 1 — Gate (EfficientNet-B0)
    A lightweight binary classifier that decides whether an image is
    authentic or forged. If authentic, we skip segmentation entirely.
    This reduces false positives and saves compute at inference time.

  Stage 2 — Segmenter (ResNet-34 encoder + UNet decoder + SE attention)
    A pixel-level segmentation model that outputs a binary mask showing
    exactly which pixels were copy-moved. Only runs on images the gate
    flagged as forged.

Training details
----------------
  - Gate:      EfficientNet-B0, pretrained ImageNet, fine-tuned 20 epochs
  - Segmenter: ResNet-34/UNet with scSE attention, 30 epochs
  - Loss:      BCE + Dice (handles severe class imbalance ~2-5% forged pixels)
  - Optimizer: AdamW + CosineAnnealingLR
  - Input:     512×512 RGB, Albumentations augmentation

Usage
-----
  # Train both stages from scratch
  python model3_train.py

  # Resume gate from checkpoint
  python model3_train.py --resume-gate

  # Resume segmenter from checkpoint
  python model3_train.py --resume-seg

  # Skip gate training (gate_best.pth already exists), train segmenter only
  python model3_train.py --seg-only

  # Skip segmenter training, train gate only
  python model3_train.py --gate-only
"""

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import (
    ConfusionMatrixDisplay, classification_report,
    confusion_matrix, f1_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import segmentation_models_pytorch as smp

from config import (
    DATA_DIR, AUTHENTIC_DIR, FORGED_DIR, MASK_DIR,
    SUPP_AUTH, SUPP_FORG, OUTPUT_DIR, VALID_EXT,
    IMAGE_SIZE, RANDOM_SEED, IMAGENET_MEAN, IMAGENET_STD,
    GATE_SAMPLES_PER_CLASS, GATE_BATCH_SIZE, GATE_NUM_EPOCHS,
    GATE_LR, GATE_WEIGHT_DECAY, GATE_PATIENCE,
    GATE_CHECKPOINT, GATE_BEST,
    SEG_BATCH_SIZE, SEG_NUM_EPOCHS, SEG_LR, SEG_WEIGHT_DECAY,
    SEG_PATIENCE, BCE_WEIGHT, DICE_WEIGHT,
    SEG_CHECKPOINT, SEG_BEST,
)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# DEVICE SETUP

def get_device() -> torch.device:
    """Pick the best available device: CUDA / MPS / CPU."""
    if torch.cuda.is_available():
        log.info("Device: CUDA")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        log.info("Device: MPS (Apple Silicon)")
        return torch.device("mps")
    log.info("Device: CPU")
    return torch.device("cpu")


# DATA UTILITIES
# Shared between gate and segmenter 

def collect_pairs(
    authentic_dir: Path,
    forged_dir: Path,
    samples_per_class: int | None = None,
    seed: int = RANDOM_SEED,
) -> list[tuple[Path, int]]:
    """
    Go through authentic/ and forged/ directories and return a shuffled list of
    (image_path, label) pairs where label is 0 (authentic) or 1 (forged).

    Also folds in supplemental_images/ if the folders exist.
    Optionally sub-samples to `samples_per_class` per class for faster runs.
    """
    auth = sorted(f for f in authentic_dir.glob("*") if f.suffix.lower() in VALID_EXT)
    forg = sorted(f for f in forged_dir.glob("*")    if f.suffix.lower() in VALID_EXT)

    # Pull in supplemental data if available
    if SUPP_AUTH.exists():
        auth += sorted(f for f in SUPP_AUTH.glob("*") if f.suffix.lower() in VALID_EXT)
    if SUPP_FORG.exists():
        forg += sorted(f for f in SUPP_FORG.glob("*") if f.suffix.lower() in VALID_EXT)

    log.info(f"Found  {len(auth)} authentic  |  {len(forg)} forged")

    rng = np.random.default_rng(seed)
    if samples_per_class:
        auth = list(rng.choice(auth, size=min(samples_per_class, len(auth)), replace=False))
        forg = list(rng.choice(forg, size=min(samples_per_class, len(forg)), replace=False))

    pairs = [(p, 0) for p in auth] + [(p, 1) for p in forg]
    rng.shuffle(pairs)
    return pairs


def split_pairs(
    pairs: list,
    seed: int = RANDOM_SEED,
) -> tuple[list, list, list]:
    """
    Split pairs into train / val / test using a 70 / 15 / 15 ratio.
    Stratified so both classes are balanced across all three splits.
    """
    labels = [p[1] for p in pairs]
    train, temp = train_test_split(pairs, test_size=0.30, stratify=labels, random_state=seed)

    temp_labels = [p[1] for p in temp]
    val, test   = train_test_split(temp, test_size=0.50, stratify=temp_labels, random_state=seed)

    log.info(f"Split  train={len(train)}  val={len(val)}  test={len(test)}")
    return train, val, test


# GATE — DATASET & TRANSFORMS

class GateDataset(Dataset):
    """
    Loads images for the gate classifier.
    Returns (image_tensor, label) where label is 0 or 1.
    """
    def __init__(self, pairs: list, transform=None, image_size: int = IMAGE_SIZE):
        self.pairs      = pairs
        self.transform  = transform
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        path, label = self.pairs[idx]

        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)


def gate_transforms(train: bool) -> transforms.Compose:
    """
    Training: flip, rotate, colour jitter -> more robust to variations.
    Validation/test: just normalise -> no randomness, reproducible.
    """
    ops = [transforms.ToPILImage()]
    if train:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(ops)


# GATE — MODEL

def build_gate_model() -> nn.Module:
    """
    EfficientNet-B0 with pretrained ImageNet weights.
    The original 1000-class classifier head is replaced with a single neuron
    (raw logit) — sigmoid is applied in the loss, not the model itself.

    Dropout 0.5 is higher than the default 0.2 to reduce overfitting,
    which we observed kicking in around epoch 4-5 during experiments.
    """
    model   = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_feat = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_feat, 1),
    )
    return model


# GATE — TRAIN / EVAL LOOPS

def gate_train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    pbar = tqdm(loader, desc=f"Gate epoch [{epoch+1:02d}/{total_epochs}]", ncols=100)
    for imgs, labels in pbar:
        imgs   = imgs.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += ((torch.sigmoid(logits) >= 0.5).float() == labels).sum().item()
        n          += imgs.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/n:.4f}")

    return total_loss / n, correct / n


@torch.no_grad()
def gate_evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, float, list, list]:
    """Evaluate gate on a dataloader. Returns (loss, acc, f1, auc, preds, labels)."""
    model.eval()
    total_loss  = 0.0
    all_probs, all_preds, all_labels = [], [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device).unsqueeze(1)

        logits = model(imgs)
        total_loss += criterion(logits, labels).item() * imgs.size(0)

        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        all_probs.extend(probs)
        all_preds.extend((probs >= 0.5).astype(int))
        all_labels.extend(labels.cpu().numpy().flatten().astype(int))

    n   = len(all_labels)
    f1  = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / n

    return total_loss / n, acc, f1, auc, all_preds, all_labels


# SEGMENTER — DATASET & TRANSFORMS

class SegDataset(Dataset):
    """
    Loads images and their corresponding binary masks for segmentation.

    Mask loading order: .npy -> .png -> zeros (authentic images have no mask).
    Masks are stored as (1, H, W) uint8 arrays in this dataset, so we
    squeeze the channel dim and cast to float32 before returning.
    """
    def __init__(
        self,
        pairs: list,
        mask_dir: Path,
        transform=None,
        image_size: int = IMAGE_SIZE,
    ):
        self.pairs      = pairs
        self.mask_dir   = mask_dir
        self.transform  = transform
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, _ = self.pairs[idx]

        # Load and resize image
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))

        # Load mask — try .npy first, then .png, then fall back to zeros
        mask_npy = self.mask_dir / (img_path.stem + ".npy")
        mask_png = self.mask_dir / (img_path.stem + ".png")

        if mask_npy.exists():
            mask = np.load(str(mask_npy))
            mask = np.squeeze(mask).astype(np.float32)   # (1,H,W) to (H,W)
        elif mask_png.exists():
            mask = cv2.imread(str(mask_png), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32)
        else:
            mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        # Safety: ensure mask is 2-D after any unexpected loading quirk
        if mask.ndim != 2 or mask.shape[0] == 0 or mask.shape[1] == 0:
            mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        # Resize mask using nearest-neighbour — bilinear would create
        # intermediate values in a binary mask, corrupting the labels
        mask = cv2.resize(
            mask, (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST,
        )
        mask = (mask > 0.5).astype(np.float32)

        if self.transform:
            out  = self.transform(image=img, mask=mask)
            img  = out["image"]
            mask = out["mask"].unsqueeze(0)   # (H,W) to (1,H,W)
        else:
            img  = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0)

        return img, mask


def seg_transforms(train: bool):
    """
    Albumentations pipeline for segmentation.
    Augmentations are applied consistently to both image and mask.
    """
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.GaussNoise(p=0.3),
            A.Blur(blur_limit=3, p=0.2),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# SEGMENTER — MODEL

def build_segmenter() -> nn.Module:
    """
    UNet with ResNet-34 encoder (pretrained ImageNet).

    decoder_attention_type='scse' adds Squeeze-and-Excitation (SE) blocks
    to each decoder stage. SE blocks learn which channels and spatial
    locations to attend to — helpful for small forged regions that
    would otherwise be drowned out by background features.

    activation=None means the model outputs raw logits.
    Sigmoid is applied inside the loss function for numerical stability.
    """
    return smp.Unet(
        encoder_name           = "resnet34",
        encoder_weights        = "imagenet",
        in_channels            = 3,
        classes                = 1,
        activation             = None,
        decoder_attention_type = "scse",
    )


# SEGMENTER — LOSS FUNCTION

class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy + Dice loss.
    BCE alone biases the model toward predicting all background when
    forged pixels are only 2-5% of the image — it can get ~95% accuracy
    without ever predicting a single forged pixel.

    Dice loss directly measures region overlap and forces the model to
    actually cover the forged region, not just get background right.

    Together: BCE handles pixel-wise correctness, Dice handles spatial coverage.
    """
    def __init__(self, bce_weight: float = BCE_WEIGHT,
                 dice_weight: float = DICE_WEIGHT, smooth: float = 1e-6):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.smooth      = smooth
        self.bce         = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs  = torch.sigmoid(logits)
        flat_p = probs.view(-1)
        flat_t = targets.view(-1)
        inter  = (flat_p * flat_t).sum()
        dice   = 1 - (2 * inter + self.smooth) / (flat_p.sum() + flat_t.sum() + self.smooth)
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * dice


# SEGMENTER — METRICS

def compute_seg_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute Dice, IoU, pixel-precision, and pixel-recall for a batch.
    Pixel accuracy is intentionally excluded — it is misleading under
    severe class imbalance (predicting all background scores ~95%).
    """
    preds  = (torch.sigmoid(logits) >= threshold).float()
    smooth = 1e-6

    inter = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    tp    = inter
    fp    = (preds * (1 - targets)).sum()
    fn    = ((1 - preds) * targets).sum()

    return {
        "dice":      ((2 * inter + smooth) / (union + smooth)).item(),
        "iou":       ((inter + smooth) / (union - inter + smooth)).item(),
        "precision": ((tp + smooth) / (tp + fp + smooth)).item(),
        "recall":    ((tp + smooth) / (tp + fn + smooth)).item(),
    }


# SEGMENTER — TRAIN / EVAL LOOPS

def seg_train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> tuple[float, dict]:
    """Run one training epoch. Returns (avg_loss, avg_metrics_dict)."""
    model.train()
    total_loss = 0.0
    metrics    = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}

    pbar = tqdm(loader, desc=f"Seg   epoch [{epoch+1:02d}/{total_epochs}]", ncols=110)
    for imgs, masks in pbar:
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        m = compute_seg_metrics(logits.detach(), masks)
        for k in metrics:
            metrics[k] += m[k] * imgs.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{m['dice']:.4f}")

    n = len(loader.dataset)
    return total_loss / n, {k: v / n for k, v in metrics.items()}


@torch.no_grad()
def seg_evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict]:
    """Evaluate segmenter on a dataloader. Returns (avg_loss, avg_metrics_dict)."""
    model.eval()
    total_loss = 0.0
    metrics    = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits      = model(imgs)
        total_loss += criterion(logits, masks).item() * imgs.size(0)
        m = compute_seg_metrics(logits, masks)
        for k in metrics:
            metrics[k] += m[k] * imgs.size(0)

    n = len(loader.dataset)
    return total_loss / n, {k: v / n for k, v in metrics.items()}


# CHECKPOINT HELPERS

def save_checkpoint(state: dict, path: Path):
    """Save training state so we can resume if interrupted."""
    torch.save(state, path)
    log.info(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
) -> tuple[int, float]:
    """Load training state. Returns (start_epoch, best_metric)."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    log.info(f"Resumed from epoch {ckpt['epoch']}  "
             f"(best metric: {ckpt['best_metric']:.4f})")
    return ckpt["epoch"], ckpt["best_metric"]


# PLOTTING HELPERS

def plot_training_curves(
    train_losses: list, val_losses: list,
    val_metrics: list, metric_name: str,
    title_prefix: str, save_path: Path,
):
    """Save a two-panel plot: loss curves + validation metric curve."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses,   label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title(f"{title_prefix} — Loss"); ax1.legend()

    ax2.plot(val_metrics, color="green", label=f"Val {metric_name}")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel(metric_name)
    ax2.set_title(f"{title_prefix} — {metric_name}"); ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    log.info(f"Training curves saved to {save_path}")


def plot_confusion_matrix(
    y_true: list, y_pred: list, title: str, save_path: Path,
):
    """Save a labelled confusion matrix as a PNG."""
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Authentic", "Forged"]).plot(ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    log.info(f"Confusion matrix saved to {save_path}")


def save_seg_prediction_samples(
    model: nn.Module, loader: DataLoader,
    device: torch.device, save_path: Path, n: int = 6,
):
    """Save n side-by-side (image | ground truth | prediction) panels."""
    model.eval()
    imgs, masks = next(iter(loader))
    imgs = imgs[:n].to(device)

    with torch.no_grad():
        preds = (torch.sigmoid(model(imgs)) >= 0.5).float().cpu()

    mean = np.array(IMAGENET_MEAN)
    std  = np.array(IMAGENET_STD)

    fig, axes = plt.subplots(n, 3, figsize=(9, n * 3))
    for i in range(n):
        img_np = np.clip(imgs[i].cpu().numpy().transpose(1, 2, 0) * std + mean, 0, 1)
        axes[i, 0].imshow(img_np);                         axes[i, 0].set_title("Image")
        axes[i, 1].imshow(masks[i, 0], cmap="gray");       axes[i, 1].set_title("Ground Truth")
        axes[i, 2].imshow(preds[i, 0], cmap="gray");       axes[i, 2].set_title("Prediction")
        for ax in axes[i]: ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    log.info(f"Prediction samples saved to {save_path}")


# PHASE 1 — TRAIN GATE

def train_gate(device: torch.device, resume: bool = False):
    """
    Train the EfficientNet-B0 gate classifier.
    Saves gate_best.pth (best val F1) and gate_checkpoint.pth (every epoch).
    """
    log.info("Phase 1 — Gate Classifier (EfficientNet-B0)")
    log.info("-" * 60)

    # Data
    pairs = collect_pairs(AUTHENTIC_DIR, FORGED_DIR, GATE_SAMPLES_PER_CLASS)
    train_pairs, val_pairs, test_pairs = split_pairs(pairs)

    train_dl = DataLoader(
        GateDataset(train_pairs, gate_transforms(train=True)),
        batch_size=GATE_BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False,
    )
    val_dl = DataLoader(
        GateDataset(val_pairs, gate_transforms(train=False)),
        batch_size=GATE_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False,
    )
    test_dl = DataLoader(
        GateDataset(test_pairs, gate_transforms(train=False)),
        batch_size=GATE_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False,
    )

    # Model, loss, optimiser
    model     = build_gate_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=GATE_LR, weight_decay=GATE_WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=GATE_NUM_EPOCHS, eta_min=1e-6)

    start_epoch, best_val_f1, no_improve = 0, 0.0, 0
    train_losses, val_losses, val_f1s    = [], [], []

    if resume and GATE_CHECKPOINT.exists():
        start_epoch, best_val_f1 = load_checkpoint(GATE_CHECKPOINT, model, optimizer, scheduler)
    elif resume:
        log.warning("--resume-gate set but no checkpoint found. Training from scratch.")

    # Training loop
    for epoch in range(start_epoch, GATE_NUM_EPOCHS):
        t0 = time.time()

        train_loss, train_acc = gate_train_one_epoch(
            model, train_dl, optimizer, criterion, device, epoch, GATE_NUM_EPOCHS)
        val_loss, val_acc, val_f1, val_auc, _, _ = gate_evaluate(
            model, val_dl, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f1s.append(val_f1)

        log.info(
            f"Epoch [{epoch+1:02d}/{GATE_NUM_EPOCHS}]  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  "
            f"val_F1={val_f1:.4f} val_AUC={val_auc:.4f}  "
            f"({time.time()-t0:.1f}s)"
        )

        save_checkpoint({
            "epoch": epoch + 1,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_metric":     best_val_f1,
        }, GATE_CHECKPOINT)

        if val_f1 > best_val_f1:
            best_val_f1, no_improve = val_f1, 0
            torch.save(model.state_dict(), GATE_BEST)
            log.info(f"  New best val F1: {best_val_f1:.4f} to {GATE_BEST}")
        else:
            no_improve += 1
            log.info(f"  No improvement ({no_improve}/{GATE_PATIENCE})")

        if no_improve >= GATE_PATIENCE:
            log.info(f"Early stopping after {GATE_PATIENCE} epochs without improvement.")
            break

    # Test evaluation using best weights
    log.info("Loading best gate weights for test evaluation …")
    model.load_state_dict(torch.load(GATE_BEST, map_location=device))
    _, test_acc, test_f1, test_auc, test_preds, test_labels = gate_evaluate(
        model, test_dl, criterion, device)

    log.info("\n" + "-" * 60)
    log.info("GATE — TEST SET RESULTS")
    log.info("-" * 60)
    log.info(f"  Accuracy : {test_acc:.4f}")
    log.info(f"  F1       : {test_f1:.4f}")
    log.info(f"  ROC-AUC  : {test_auc:.4f}")
    print(classification_report(test_labels, test_preds,
                                target_names=["Authentic", "Forged"]))

    plot_training_curves(
        train_losses, val_losses, val_f1s, "F1",
        "Gate (EfficientNet-B0)",
        OUTPUT_DIR / "gate_training_curves.png",
    )
    plot_confusion_matrix(
        test_labels, test_preds,
        "Confusion Matrix — Gate (EfficientNet-B0) — Test Set",
        OUTPUT_DIR / "gate_confusion_matrix.png",
    )



# PHASE 2 — TRAIN SEGMENTER

def train_segmenter(device: torch.device, resume: bool = False):
    """
    Train the ResNet-34/UNet segmenter.
    Saves segmenter_best.pth (best val Dice) and segmenter_checkpoint.pth.
    """
    log.info("-" * 60)
    log.info("Phase 2 — Segmenter (ResNet-34 / UNet + scSE Attention)")
    log.info("-" * 60)

    # Data — use all available images (no sampling limit for segmenter)
    pairs = collect_pairs(AUTHENTIC_DIR, FORGED_DIR)
    train_pairs, val_pairs, test_pairs = split_pairs(pairs)

    train_dl = DataLoader(
        SegDataset(train_pairs, MASK_DIR, seg_transforms(train=True)),
        batch_size=SEG_BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False,
    )
    val_dl = DataLoader(
        SegDataset(val_pairs, MASK_DIR, seg_transforms(train=False)),
        batch_size=SEG_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False,
    )
    test_dl = DataLoader(
        SegDataset(test_pairs, MASK_DIR, seg_transforms(train=False)),
        batch_size=SEG_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False,
    )

    # Model, loss, optimiser
    model     = build_segmenter().to(device)
    criterion = BCEDiceLoss()
    optimizer = AdamW(model.parameters(), lr=SEG_LR, weight_decay=SEG_WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=SEG_NUM_EPOCHS, eta_min=1e-6)

    start_epoch, best_val_dice, no_improve = 0, 0.0, 0
    train_losses, val_losses, val_dices    = [], [], []

    if resume and SEG_CHECKPOINT.exists():
        start_epoch, best_val_dice = load_checkpoint(SEG_CHECKPOINT, model, optimizer, scheduler)
    elif resume:
        log.warning("--resume-seg set but no checkpoint found. Training from scratch.")

    # Training loop
    for epoch in range(start_epoch, SEG_NUM_EPOCHS):
        t0 = time.time()

        train_loss, train_m = seg_train_one_epoch(
            model, train_dl, optimizer, criterion, device, epoch, SEG_NUM_EPOCHS)
        val_loss, val_m = seg_evaluate(model, val_dl, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dices.append(val_m["dice"])

        log.info(
            f"Epoch [{epoch+1:02d}/{SEG_NUM_EPOCHS}]  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_Dice={val_m['dice']:.4f}  "
            f"val_IoU={val_m['iou']:.4f}  val_Prec={val_m['precision']:.4f}  "
            f"val_Rec={val_m['recall']:.4f}  ({time.time()-t0:.1f}s)"
        )

        save_checkpoint({
            "epoch": epoch + 1,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_metric":     best_val_dice,
        }, SEG_CHECKPOINT)

        if val_m["dice"] > best_val_dice:
            best_val_dice, no_improve = val_m["dice"], 0
            torch.save(model.state_dict(), SEG_BEST)
            log.info(f"  New best val Dice: {best_val_dice:.4f} to {SEG_BEST}")
        else:
            no_improve += 1
            log.info(f"  No improvement ({no_improve}/{SEG_PATIENCE})")

        if no_improve >= SEG_PATIENCE:
            log.info(f"Early stopping after {SEG_PATIENCE} epochs without improvement.")
            break

    # Test evaluation using best weights
    log.info("Loading best segmenter weights for test evaluation …")
    model.load_state_dict(torch.load(SEG_BEST, map_location=device))
    _, test_m = seg_evaluate(model, test_dl, criterion, device)

    log.info("\n" + "-" * 60)
    log.info("SEGMENTER — TEST SET RESULTS")
    log.info("-" * 60)
    for k, v in test_m.items():
        log.info(f"  {k:12s}: {v:.4f}")

    plot_training_curves(
        train_losses, val_losses, val_dices, "Dice",
        "Segmenter (ResNet-34/UNet)",
        OUTPUT_DIR / "segmenter_training_curves.png",
    )
    save_seg_prediction_samples(
        model, test_dl, device,
        OUTPUT_DIR / "segmenter_predictions.png",
    )


# ENTRY POINT

def main():
    parser = argparse.ArgumentParser(
        description="Train Model 3 — Gated Segmentation Network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--resume-gate", action="store_true",
                        help="Resume gate training from outputs/gate_checkpoint.pth")
    parser.add_argument("--resume-seg",  action="store_true",
                        help="Resume segmenter from outputs/segmenter_checkpoint.pth")
    parser.add_argument("--gate-only",   action="store_true",
                        help="Train gate only, skip segmenter")
    parser.add_argument("--seg-only",    action="store_true",
                        help="Train segmenter only, skip gate")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    device = get_device()

    if not args.seg_only:
        train_gate(device, resume=args.resume_gate)

    if not args.gate_only:
        train_segmenter(device, resume=args.resume_seg)

    log.info("\nAll done. Outputs written to: outputs/")


if __name__ == "__main__":
    main()