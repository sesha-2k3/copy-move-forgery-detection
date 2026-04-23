"""
Model 1: SVM with RBF Kernel
Copy-Move Forgery Detection in Biomedical Scientific Images

Imports the shared feature pipeline from src/features.py.
Only SVM-specific code lives here: the pipeline, param grid, and visualisation.

Usage:
  python -m src.svm
"""

import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from features import (
    load_dataset, load_or_build_features, split,
    evaluate_and_save,
    RANDOM_SEED, CV_FOLDS, OUTPUT_DIR,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# SVM-specific hyperparameter grid
PARAM_GRID = {
    "svm__C":      [10, 100, 500, 1000],
    "svm__gamma":  ["scale", 0.0001, 0.001, 0.01],
    "svm__kernel": ["rbf"],
}


def build_pipeline() -> Pipeline:
    """
    StandardScaler → SVC(RBF).
    Scaler lives inside the pipeline so it never sees val/test data during CV.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED)),
    ])


def tune(X: np.ndarray, y: np.ndarray) -> GridSearchCV:
    """5-fold stratified GridSearchCV over PARAM_GRID."""
    import time
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    gs = GridSearchCV(build_pipeline(), PARAM_GRID, cv=cv,
                      scoring="f1", n_jobs=-1, verbose=2, refit=True)
    log.info("Starting GridSearchCV …")
    t0 = time.time()
    gs.fit(X, y)
    log.info(f"Done in {time.time()-t0:.1f}s  |  best params: {gs.best_params_}  |  best F1: {gs.best_score_:.4f}")
    return gs


def plot_gridsearch(grid_search: GridSearchCV):
    """Heatmap of mean CV F1 over C × gamma."""
    cv_res = pd.DataFrame(grid_search.cv_results_)

    for kernel in PARAM_GRID["svm__kernel"]:
        sub = cv_res[cv_res["param_svm__kernel"] == kernel].copy()
        sub = sub[sub["param_svm__gamma"].apply(lambda g: isinstance(g, float))]
        if sub.empty:
            continue

        sub["C"]     = sub["param_svm__C"].astype(float)
        sub["gamma"] = sub["param_svm__gamma"].astype(float)
        pivot = sub.pivot_table(values="mean_test_score", index="gamma", columns="C")

        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)));   ax.set_yticklabels(pivot.index)
        ax.set_xlabel("C"); ax.set_ylabel("gamma")
        ax.set_title(f"SVM GridSearch CV F1 — kernel={kernel}")
        plt.colorbar(im, ax=ax); plt.tight_layout()
        path = OUTPUT_DIR / f"svm_gridsearch_{kernel}.png"
        plt.savefig(path, dpi=150); plt.close()
        log.info(f"GridSearch heatmap saved → {path}")


def save_predictions(grid_search: GridSearchCV,
                     X: np.ndarray, y: np.ndarray, df: pd.DataFrame):
    """Save predicted labels + probabilities for all samples to CSV."""
    best = grid_search.best_estimator_
    best.fit(X, y)
    out = df[["image_id", "label"]].copy()
    out["pred_label"] = best.predict(X)
    out["pred_prob"]  = best.predict_proba(X)[:, 1]
    path = OUTPUT_DIR / "svm_predictions.csv"
    out.to_csv(path, index=False)
    log.info(f"Predictions saved → {path}")


def main():
    log.info("Model 1 — SVM (RBF Kernel)")
    log.info("-" * 50)
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_dataset()
    X, y = load_or_build_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split(X, y)

    gs = tune(X_train, y_train)

    val_f1 = f1_score(y_val, gs.best_estimator_.predict(X_val))
    log.info(f"Val F1: {val_f1:.4f}")

    evaluate_and_save(gs.best_estimator_, X_test, y_test, model_name="SVM")
    plot_gridsearch(gs)
    save_predictions(gs, X, y, df)

    log.info("Done. Outputs → outputs/")


if __name__ == "__main__":
    main()