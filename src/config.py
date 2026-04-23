"""
config.py
All hyperparameters and paths in one place.
Change values here — no need to touch any other file.
"""
from pathlib import Path

# Paths
DATA_DIR      = Path("data")
AUTHENTIC_DIR = DATA_DIR / "train_images" / "authentic"
FORGED_DIR    = DATA_DIR / "train_images" / "forged"
MASK_DIR      = DATA_DIR / "train_masks"
SUPP_AUTH     = DATA_DIR / "supplemental_images" / "authentic"
SUPP_FORG     = DATA_DIR / "supplemental_images" / "forged"
OUTPUT_DIR    = Path("outputs")
CACHE_PATH    = OUTPUT_DIR / "svm_features.npz"
PLOTS_DIR     = OUTPUT_DIR / "plots"
PREDS_DIR     = OUTPUT_DIR / "predictions"

VALID_EXT     = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# Shared
IMAGE_SIZE    = 512
RANDOM_SEED   = 42
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# SVM / Random Forest
SAMPLES_PER_CLASS = 1200
PATCH_SIZE        = 64
STRIDE            = 32
DCT_TOP_K         = 16
TOP_K_MAXIMA      = 5
FEATURE_DIM       = 4 + TOP_K_MAXIMA + 4 + 1 + 20
CV_FOLDS          = 5
# LBP
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS
LBP_METHOD = "uniform"

# HOG
HOG_ORIENTATIONS    = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

# Feature names for Random Forest importance plot
FEATURE_NAMES = (
    ["mean", "variance", "skewness", "kurtosis"]
    + [f"top_{i+1}_max" for i in range(TOP_K_MAXIMA)]
    + ["p75", "p90", "p95", "p99"]
    + ["high_sim_ratio"]
    + [f"hist_bin_{i+1}" for i in range(20)]
)

# Gate (EfficientNet-B0)
GATE_SAMPLES_PER_CLASS = 1200
GATE_BATCH_SIZE        = 32
GATE_NUM_EPOCHS        = 20
GATE_LR                = 1e-4
GATE_WEIGHT_DECAY      = 1e-4
GATE_PATIENCE          = 5
GATE_CHECKPOINT        = OUTPUT_DIR / "gate_checkpoint.pth"
GATE_BEST              = OUTPUT_DIR / "gate_best.pth"

# Segmenter (ResNet-34 / UNet)
SEG_BATCH_SIZE   = 16
SEG_NUM_EPOCHS   = 30
SEG_LR           = 3e-4
SEG_WEIGHT_DECAY = 1e-4
SEG_PATIENCE     = 7
BCE_WEIGHT       = 0.5
DICE_WEIGHT      = 0.5
SEG_CHECKPOINT   = OUTPUT_DIR / "segmenter_checkpoint.pth"
SEG_BEST         = OUTPUT_DIR / "segmenter_best.pth"