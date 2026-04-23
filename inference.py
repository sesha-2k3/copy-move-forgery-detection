"""
Gated Inference for Copy-Move Forgery Detection

Loads saved gate (EfficientNet-B0) and segmenter (ResNet34/UNet) weights
and runs the full pipeline on one or more images.

Usage:
  # Single image
  python inference.py --image data/train_images/forged/10.png

  # Entire folder
  python inference.py --folder data/train_images/forged/

  # Evaluate against ground truth masks
  python inference.py --folder data/train_images/forged/ --masks data/train_masks/
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# CONFIG

IMAGE_SIZE      = 512
GATE_THRESHOLD  = 0.5
SEG_THRESHOLD   = 0.2
GATE_PATH       = Path("outputs/gate_best.pth")
SEG_PATH        = Path("outputs/segmenter_best.pth")
OUTPUT_DIR      = Path("outputs/inference")
VALID_EXT       = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# DEVICE

def get_device():
    if torch.cuda.is_available():
        log.info("Using device: CUDA")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        log.info("Using device: MPS (Apple Silicon)")
        return torch.device("mps")
    log.info("Using device: CPU")
    return torch.device("cpu")


# MODEL BUILDERS

def load_gate(path: Path, device):
    model = models.efficientnet_b0(weights=None)
    in_f  = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(in_f, 1))
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    log.info(f"Gate loaded from {path}")
    return model


def load_segmenter(path: Path, device):
    model = smp.Unet(
        encoder_name           = "resnet34",
        encoder_weights        = None,
        in_channels            = 3,
        classes                = 1,
        activation             = None,
        decoder_attention_type = "scse",
    )
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    log.info(f"Segmenter loaded from {path}")
    return model


# TRANSFORMS

gate_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

seg_tf = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# INFERENCE

@torch.no_grad()
def predict(image_path: Path, gate_model, seg_model, device):
    """
    Full gated inference pipeline for a single image.
    Returns:
      gate_prob : float        — probability of forgery from gate
      mask      : np.ndarray  — binary mask (H, W), original resolution
                                all zeros if gate predicts authentic
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = img_rgb.shape[:2]

    # Stage 1: Gate
    gate_input = gate_tf(img_rgb).unsqueeze(0).to(device)
    gate_prob  = torch.sigmoid(gate_model(gate_input)).item()

    if gate_prob < GATE_THRESHOLD:
        log.info(f"{image_path.name}: AUTHENTIC  (gate p={gate_prob:.3f})")
        return gate_prob, np.zeros((h_orig, w_orig), dtype=np.uint8)

    # Stage 2: Segmenter
    img_resized = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    seg_input   = seg_tf(image=img_resized)["image"].unsqueeze(0).to(device)
    seg_out     = torch.sigmoid(seg_model(seg_input))
    mask_512    = (seg_out[0, 0].cpu().numpy() >= SEG_THRESHOLD).astype(np.uint8)
    
    # Resize mask back to original image resolution
    mask_orig = cv2.resize(mask_512, (w_orig, h_orig),
                           interpolation=cv2.INTER_NEAREST)

    forged_pct = mask_orig.mean() * 100
    log.info(f"{image_path.name}: FORGED     (gate p={gate_prob:.3f}, "
             f"forged pixels={forged_pct:.1f}%)")

    return gate_prob, mask_orig


# METRICS (when ground truth masks are available)

def compute_dice_iou(pred_mask: np.ndarray, gt_mask: np.ndarray):
    """Compute Dice and IoU between predicted and ground truth masks."""
    pred = pred_mask.astype(bool)
    gt   = gt_mask.astype(bool)

    inter  = (pred & gt).sum()
    union  = (pred | gt).sum()
    smooth = 1e-6

    dice = (2 * inter + smooth) / (pred.sum() + gt.sum() + smooth)
    iou  = (inter + smooth) / (union + smooth)
    return float(dice), float(iou)


def load_gt_mask(image_path: Path, mask_dir: Path) -> np.ndarray:
    """Load ground truth mask for a given image."""
    mask_npy = mask_dir / (image_path.stem + ".npy")
    mask_png = mask_dir / (image_path.stem + ".png")

    if mask_npy.exists():
        mask = np.load(str(mask_npy))
        mask = np.squeeze(mask).astype(np.float32)
    elif mask_png.exists():
        mask = cv2.imread(str(mask_png), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)
    else:
        return None

    return (mask > 0.5).astype(np.uint8)


# VISUALISATION

def save_visualisation(image_path: Path, mask: np.ndarray,
                        gate_prob: float, gt_mask=None):
    """Save side-by-side visualisation: image | prediction | (gt if available)."""
    img_bgr = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    n_cols = 3 if gt_mask is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    axes[0].imshow(img_rgb)
    axes[0].set_title(f"Input Image\nGate p={gate_prob:.3f}")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    label = "FORGED" if gate_prob >= GATE_THRESHOLD else "AUTHENTIC"
    axes[1].set_title(f"Predicted Mask\n{label}")
    axes[1].axis("off")

    if gt_mask is not None:
        axes[2].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
        if gt_mask is not None:
            dice, iou = compute_dice_iou(mask, gt_mask)
            axes[2].set_title(f"Ground Truth\nDice={dice:.3f}  IoU={iou:.3f}")
        axes[2].axis("off")

    plt.suptitle(image_path.name, fontsize=10)
    plt.tight_layout()

    out_path = OUTPUT_DIR / f"{image_path.stem}_result.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


# MAIN

def main():
    parser = argparse.ArgumentParser(
        description="Run gated copy-move forgery inference")
    parser.add_argument("--image",  type=str, default=None,
                        help="Path to a single image")
    parser.add_argument("--folder", type=str, default=None,
                        help="Path to a folder of images")
    parser.add_argument("--masks",  type=str, default=None,
                        help="Path to ground truth masks folder (optional)")
    args = parser.parse_args()

    if not args.image and not args.folder:
        parser.error("Provide --image or --folder")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()

    # Load models
    gate_model = load_gate(GATE_PATH, device)
    seg_model  = load_segmenter(SEG_PATH, device)
    mask_dir   = Path(args.masks) if args.masks else None

    # Collect images
    if args.image:
        image_paths = [Path(args.image)]
    else:
        image_paths = sorted(
            f for f in Path(args.folder).glob("*")
            if f.suffix.lower() in VALID_EXT
        )

    log.info(f"Running inference on {len(image_paths)} image(s)...")

    # Run inference
    all_dice, all_iou = [], []
    results = []

    for img_path in image_paths:
        gate_prob, mask = predict(img_path, gate_model, seg_model, device)

        gt_mask = None
        if mask_dir:
            gt_mask = load_gt_mask(img_path, mask_dir)
            if gt_mask is not None:
                # Resize gt to match prediction if needed
                if gt_mask.shape != mask.shape:
                    gt_mask = cv2.resize(gt_mask, (mask.shape[1], mask.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)
                dice, iou = compute_dice_iou(mask, gt_mask)
                all_dice.append(dice)
                all_iou.append(iou)
                results.append({
                    "image": img_path.name,
                    "gate_prob": round(gate_prob, 4),
                    "predicted": "forged" if gate_prob >= GATE_THRESHOLD else "authentic",
                    "dice": round(dice, 4),
                    "iou": round(iou, 4),
                })

        save_visualisation(img_path, mask, gate_prob, gt_mask)

    # Summary
    log.info("\n" + "-" * 50)
    log.info(f"Processed {len(image_paths)} images")
    log.info(f"Results saved to {OUTPUT_DIR}/")

    if all_dice:
        log.info(f"Mean Dice: {np.mean(all_dice):.4f}")
        log.info(f"Mean IoU: {np.mean(all_iou):.4f}")

    log.info("-" * 50)


if __name__ == "__main__":
    main()