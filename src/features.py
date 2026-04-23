"""
Feature extraction pipeline for SVM and Random Forest.

1. Load images from authentic/ and forged/ subfolders
2. Extract overlapping 64x64 patches per image
3. Compute HOG + LBP + DCT descriptors per patch
4. Measure pairwise cosine similarity across spatially distant patch pairs
5. Compress the similarity distribution into a 34-dim feature vector
"""

import time
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

import cv2
from skimage.feature import hog, local_binary_pattern
from scipy.fftpack import dct
from scipy.stats import skew, kurtosis
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, roc_auc_score, ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import (
    DATA_DIR, AUTHENTIC_DIR, FORGED_DIR, OUTPUT_DIR, CACHE_PATH,
    VALID_EXT,
    SAMPLES_PER_CLASS, RANDOM_SEED, CV_FOLDS,
    PATCH_SIZE, STRIDE,
    LBP_RADIUS, LBP_POINTS, LBP_METHOD,
    HOG_ORIENTATIONS, HOG_PIXELS_PER_CELL, HOG_CELLS_PER_BLOCK,
    DCT_TOP_K, TOP_K_MAXIMA,
    FEATURE_DIM, FEATURE_NAMES,
)

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


# Data loading

def load_dataset(samples_per_class: int = SAMPLES_PER_CLASS,
                 seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Collect images from authentic/ and forged/ subfolders (plus supplemental
    if present) and return a balanced, shuffled DataFrame with columns:
      image_id, path, label  (0 = authentic, 1 = forged)
    """
    auth_files = sorted(f for f in AUTHENTIC_DIR.glob("*")
                        if f.suffix.lower() in VALID_EXT)
    forg_files = sorted(f for f in FORGED_DIR.glob("*")
                        if f.suffix.lower() in VALID_EXT)

    supp_auth = DATA_DIR / "supplemental_images" / "authentic"
    supp_forg = DATA_DIR / "supplemental_images" / "forged"
    if supp_auth.exists():
        auth_files += sorted(f for f in supp_auth.glob("*")
                             if f.suffix.lower() in VALID_EXT)
    if supp_forg.exists():
        forg_files += sorted(f for f in supp_forg.glob("*")
                             if f.suffix.lower() in VALID_EXT)

    log.info(f"Authentic: {len(auth_files)}  |  Forged: {len(forg_files)}")

    rng    = np.random.default_rng(seed)
    n_auth = min(samples_per_class, len(auth_files))
    n_forg = min(samples_per_class, len(forg_files))

    auth_sample = rng.choice(auth_files, size=n_auth, replace=False)
    forg_sample = rng.choice(forg_files, size=n_forg, replace=False)

    records = (
        [{"image_id": f.name, "path": f, "label": 0} for f in auth_sample]
        + [{"image_id": f.name, "path": f, "label": 1} for f in forg_sample]
    )

    df = pd.DataFrame(records).sample(frac=1, random_state=seed).reset_index(drop=True)
    log.info(f"Dataset: {n_auth} authentic + {n_forg} forged = {n_auth + n_forg} total")
    return df


# Patch extraction

def extract_patches(image: np.ndarray,
                    patch_size: int = PATCH_SIZE,
                    stride: int = STRIDE):
    """
    Slide a window over a grayscale image and return all patches + their
    center coordinates. Coordinates are used later to filter out nearby
    patch pairs that are trivially similar in any image.

    Returns:
      patches : (N, patch_size, patch_size)
      coords  : (N, 2)  — (row_center, col_center)
    """
    h, w = image.shape
    patches, coords = [], []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patches.append(image[y:y + patch_size, x:x + patch_size])
            coords.append([y + patch_size // 2, x + patch_size // 2])
    return np.array(patches), np.array(coords)


# Descriptors

def compute_hog_descriptor(patch: np.ndarray) -> np.ndarray:
    """Histogram of Oriented Gradients — captures edge and gradient structure."""
    return hog(
        patch,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    )


def compute_lbp_descriptor(patch: np.ndarray) -> np.ndarray:
    """Local Binary Patterns — captures micro-texture."""
    lbp    = local_binary_pattern(patch, LBP_POINTS, LBP_RADIUS, method=LBP_METHOD)
    n_bins = LBP_POINTS + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist


def compute_dct_descriptor(patch: np.ndarray) -> np.ndarray:
    """Top-K DCT coefficients — captures frequency domain information."""
    patch_f = patch.astype(np.float32) / 255.0
    coeffs  = dct(dct(patch_f, axis=0, norm="ortho"), axis=1, norm="ortho")
    flat    = coeffs.ravel()
    idx     = np.argsort(np.abs(flat))[::-1][:DCT_TOP_K]
    return flat[idx]


def compute_patch_descriptor(patch: np.ndarray) -> np.ndarray:
    """Concatenate HOG + LBP + DCT into a single 1-D descriptor per patch."""
    return np.concatenate([
        compute_hog_descriptor(patch),
        compute_lbp_descriptor(patch),
        compute_dct_descriptor(patch),
    ])


# Intra-image similarity -> feature vector

def cosine_similarity_matrix(descriptors: np.ndarray,
                              patch_coords: np.ndarray,
                              min_distance: int = 128) -> np.ndarray:
    """
    Compute pairwise cosine similarity between all patches, keeping only
    pairs whose spatial distance >= min_distance.

    Why filter by distance? Copy-move forgery creates high similarity
    between *distant* patches. Nearby patches are naturally similar in any
    image, so including them would drown out the forgery signal.
    """
    norms  = np.linalg.norm(descriptors, axis=1, keepdims=True)
    norms  = np.where(norms == 0, 1e-8, norms)
    normed = descriptors / norms
    sim_matrix = normed @ normed.T

    idx_i, idx_j = np.triu_indices(len(descriptors), k=1)
    spatial_dist = np.linalg.norm(
        patch_coords[idx_i] - patch_coords[idx_j], axis=1)
    mask = spatial_dist >= min_distance

    return sim_matrix[idx_i[mask], idx_j[mask]]


def similarity_stats(sim_values: np.ndarray) -> np.ndarray:
    """
    Compress the similarity distribution into a fixed-length feature vector.

    Returns a 34-dim vector:
      [mean, variance, skewness, kurtosis,
       top_5_maxima (5 values),
       p75, p90, p95, p99,
       high_sim_ratio,
       20-bin histogram]
    """
    top_k_vals     = np.sort(sim_values)[::-1][:TOP_K_MAXIMA]
    percentiles    = np.percentile(sim_values, [75, 90, 95, 99])
    high_sim_ratio = np.mean(sim_values > 0.95)
    hist, _        = np.histogram(sim_values, bins=20, range=(0.0, 1.0), density=True)

    return np.array([
        np.mean(sim_values), np.var(sim_values),
        float(skew(sim_values)), float(kurtosis(sim_values)),
        *top_k_vals, *percentiles, high_sim_ratio, *hist,
    ], dtype=np.float32)


def image_to_feature_vector(image_path: Path) -> np.ndarray:
    """Run the full pipeline on a single image -> 34-dim feature vector."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (512, 512))

    patches, coords = extract_patches(gray)
    if len(patches) < 2:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    descriptors = np.array([compute_patch_descriptor(p) for p in patches])
    sim_values  = cosine_similarity_matrix(descriptors, coords)

    if len(sim_values) == 0:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    return similarity_stats(sim_values)


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature vectors for every image in the dataframe. Returns (X, y)."""
    X_list, y_list, failed = [], [], []
    n, t0 = len(df), time.time()

    for i, row in df.iterrows():
        try:
            X_list.append(image_to_feature_vector(row["path"]))
            y_list.append(int(row["label"]))
        except Exception as e:
            log.warning(f"  Skipping {row['image_id']}: {e}")
            failed.append(row["image_id"])

        if (i + 1) % 50 == 0 or (i + 1) == n:
            log.info(f"  [{i+1}/{n}] features extracted  ({time.time()-t0:.1f}s)")

    if failed:
        log.warning(f"{len(failed)} images skipped.")

    return np.vstack(X_list).astype(np.float32), np.array(y_list, dtype=np.int32)


# Shared evaluation helper

def evaluate_and_save(estimator, X_test: np.ndarray, y_test: np.ndarray,
                      model_name: str):
    """
    Print classification report and save confusion matrix.
    Used by both SVM and Random Forest — model_name controls the title/filename.
    """
    y_pred = estimator.predict(X_test)
    y_prob = estimator.predict_proba(X_test)[:, 1]

    log.info("\n" + "-" * 60)
    log.info(f"{model_name.upper()} — FINAL TEST SET RESULTS")
    log.info("-" * 60)
    log.info(f"  F1      : {f1_score(y_test, y_pred):.4f}")
    log.info(f"  ROC-AUC : {roc_auc_score(y_test, y_prob):.4f}")

    log.info(f"\nClassification Report — {model_name}:")
    log.info(classification_report(y_test, y_pred,
                                target_names=["Authentic", "Forged"]))

    cm  = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Authentic", "Forged"]).plot(ax=ax)
    ax.set_title(f"Confusion Matrix — {model_name} — Test Set")
    plt.tight_layout()
    slug = model_name.lower().replace(" ", "_")
    path = OUTPUT_DIR / f"{slug}_confusion_matrix.png"
    plt.savefig(path, dpi=150)
    plt.close()
    log.info(f"Confusion matrix saved to {path}")


def load_or_build_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Load features from cache (outputs/svm_features.npz) if available,
    otherwise extract from scratch and cache them.
    This saves re-extraction time when running Random Forest after SVM.
    """
    if CACHE_PATH.exists():
        log.info(f"Loading cached features from {CACHE_PATH}...")
        data = np.load(CACHE_PATH)
        X, y = data["X"], data["y"]
        if len(X) == len(df):
            log.info(f"Cache hit - X={X.shape}")
            return X, y
        log.warning("Cache size mismatch - re-extracting...")

    log.info("Extracting features...")
    X, y = build_feature_matrix(df)
    OUTPUT_DIR.mkdir(exist_ok=True)
    np.savez_compressed(CACHE_PATH, X=X, y=y)
    log.info(f"Features cached in this path: {CACHE_PATH}")
    return X, y


def split(X: np.ndarray, y: np.ndarray, seed: int = RANDOM_SEED):
    """70/15/15 stratified train/val/test split."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=seed)
    log.info(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test