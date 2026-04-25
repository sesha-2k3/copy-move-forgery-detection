"""
Model 2: Random Forest Classifier
Copy-Move Forgery Detection in Biomedical Scientific Images

Reuses the cached feature matrix from SVM (outputs/svm_features.npz) if present —
no re-extraction needed.

Usage:
  python -m src.random_forest
"""

import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import (
    RANDOM_SEED, CV_FOLDS, OUTPUT_DIR, PLOTS_DIR, PREDS_DIR, FEATURE_NAMES,
)

from .features import (
    load_dataset, load_or_build_features, split,
    evaluate_and_save,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# RF-specific hyperparameter grid
PARAM_GRID = {
    "rf__n_estimators":     [100, 300, 500],
    "rf__max_depth":        [None, 10, 20, 30],
    "rf__min_samples_leaf": [1, 2, 4],
}


def build_pipeline() -> Pipeline:
    """
    Random Forest inside a Pipeline.
    No scaling needed — RF is scale-invariant by design.
    """
    return Pipeline([
        ("rf", RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)),
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


def plot_feature_importance(estimator):
    """
    Bar chart of top-20 features by mean decrease in impurity.
    """
    
    rf          = estimator.named_steps["rf"]
    importances = rf.feature_importances_
    std         = np.std([t.feature_importances_ for t in rf.estimators_], axis=0)
    indices     = np.argsort(importances)[::-1][:20]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(20), importances[indices], yerr=std[indices],
           color="steelblue", ecolor="black", capsize=3)
    ax.set_xticks(range(20))
    ax.set_xticklabels([FEATURE_NAMES[i] for i in indices],
                       rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean Decrease in Impurity")
    ax.set_title("Random Forest — Top 20 Feature Importances")
    plt.tight_layout()
    PLOTS_DIR.mkdir(exist_ok=True)
    path = PLOTS_DIR / "rf_feature_importance.png"
    plt.savefig(path, dpi=150); plt.close()
    log.info(f"Feature importance plot saved to {path}")

    log.info("Top 10 features:")
    for rank, idx in enumerate(indices[:10], 1):
        log.info(f"  {rank:2d}. {FEATURE_NAMES[idx]:20s}  {importances[idx]:.4f}")


def plot_gridsearch(grid_search: GridSearchCV):
    """Line plot of CV F1 vs n_estimators for each max_depth."""
    cv_res = pd.DataFrame(grid_search.cv_results_)
    cv_res["n_estimators"] = cv_res["param_rf__n_estimators"].astype(int)
    cv_res["max_depth"]    = cv_res["param_rf__max_depth"].astype(str)

    fig, ax = plt.subplots(figsize=(8, 5))
    for depth, grp in cv_res.groupby("max_depth"):
        grp = grp.sort_values("n_estimators")
        ax.plot(grp["n_estimators"], grp["mean_test_score"],
                marker="o", label=f"max_depth={depth}")
    ax.set_xlabel("n_estimators"); ax.set_ylabel("Mean CV F1")
    ax.set_title("Random Forest — GridSearch CV F1"); ax.legend()
    plt.tight_layout()
    PLOTS_DIR.mkdir(exist_ok=True)
    path = PLOTS_DIR / "rf_gridsearch.png"
    plt.savefig(path, dpi=150); plt.close()
    log.info(f"GridSearch plot saved to {path}")


def save_predictions(grid_search: GridSearchCV,
                     X: np.ndarray, y: np.ndarray, df: pd.DataFrame):
    """Save predicted labels + probabilities for all samples to CSV."""
    best = grid_search.best_estimator_
    out = df[["image_id", "label"]].copy()
    out["pred_label"] = best.predict(X)
    out["pred_prob"]  = best.predict_proba(X)[:, 1]
    PREDS_DIR.mkdir(exist_ok=True)
    path = PREDS_DIR / "rf_predictions.csv"
    out.to_csv(path, index=False)
    log.info(f"Predictions saved to {path}")


def main():
    log.info("Model 2 — Random Forest")
    log.info("-" * 50)
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_dataset()
    X, y = load_or_build_features(df)   # uses SVM cache if available
    X_train, X_val, X_test, y_train, y_val, y_test = split(X, y)

    gs = tune(X_train, y_train)

    val_f1 = f1_score(y_val, gs.best_estimator_.predict(X_val))
    log.info(f"Val F1: {val_f1:.4f}")

    evaluate_and_save(gs.best_estimator_, X_test, y_test, model_name="Random Forest")
    plot_feature_importance(gs.best_estimator_)
    plot_gridsearch(gs)
    save_predictions(gs, X, y, df)

    log.info("Done. Outputs saved to outputs/")


if __name__ == "__main__":
    main()