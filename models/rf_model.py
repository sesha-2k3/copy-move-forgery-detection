"""
models/rf_model.py
------------------
Model 2 — Random Forest for image-level forgery classification.

Uses the identical feature pipeline as the SVM for a controlled comparison.
Additionally provides feature importance rankings, enabling analysis of which
descriptors (HOG, LBP, or DCT) and which similarity statistics are most
informative for forgery detection.

Hyperparameters tuned via 5-fold cross-validation:
  - n_estimators
  - max_depth
  - min_samples_leaf
"""

import time
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)


class RFForgeryClassifier:
    """Random Forest binary classifier for copy-move forgery detection."""

    def __init__(self):
        self.pipeline = None
        self.best_params = None
        self.cv_results = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv_folds: int = 5,
        seed: int = 42,
        verbose: int = 1,
    ) -> dict:
        """
        Train Random Forest with GridSearchCV.

        Returns:
            dict with best_params, best_cv_score, and training time.
        """
        print("\n" + "=" * 60)
        print("  Model 2 — Random Forest")
        print("=" * 60)

        param_grid = {
            "rf__n_estimators":    [100, 200, 500],
            "rf__max_depth":       [10, 20, 30, None],
            "rf__min_samples_leaf": [1, 2, 5],
        }

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("rf",     RandomForestClassifier(random_state=seed, n_jobs=-1)),
        ])

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

        grid = GridSearchCV(
            pipe,
            param_grid,
            cv=cv,
            scoring="f1",
            n_jobs=-1,
            verbose=verbose,
            refit=True,
        )

        n_combos = (len(param_grid["rf__n_estimators"]) *
                    len(param_grid["rf__max_depth"]) *
                    len(param_grid["rf__min_samples_leaf"]))
        print(f"Running {cv_folds}-fold GridSearchCV "
              f"({n_combos} combinations) ...")
        t0 = time.time()
        grid.fit(X_train, y_train)
        elapsed = time.time() - t0

        self.pipeline = grid.best_estimator_
        self.best_params = grid.best_params_
        self.cv_results = grid.cv_results_

        result = {
            "best_params":   self.best_params,
            "best_cv_f1":    float(grid.best_score_),
            "train_time_s":  round(elapsed, 1),
        }

        print(f"\n  Best CV F1:    {grid.best_score_:.4f}")
        print(f"  Best params:   {self.best_params}")
        print(f"  Training time: {elapsed:.1f}s")

        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions (0 = authentic, 1 = forged)."""
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates [P(authentic), P(forged)]."""
        return self.pipeline.predict_proba(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate on a test set.

        Returns:
            dict with accuracy, precision, recall, f1, confusion_matrix,
            and full classification_report string.
        """
        y_pred = self.predict(X_test)

        metrics = {
            "accuracy":  float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall":    float(recall_score(y_test, y_pred)),
            "f1":        float(f1_score(y_test, y_pred)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(
                y_test, y_pred, target_names=["authentic", "forged"]
            ),
        }

        print("\n── Random Forest Evaluation ──")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"\n{metrics['classification_report']}")

        return metrics

    def get_feature_importances(self, feature_names: list[str]) -> pd.DataFrame:
        """
        Return a DataFrame of feature importances sorted by importance.

        Args:
            feature_names: list of human-readable feature names from
                           features.extract.get_feature_names()
        Returns:
            DataFrame with columns [feature, importance, descriptor]
        """
        rf = self.pipeline.named_steps["rf"]
        importances = rf.feature_importances_

        df = pd.DataFrame({
            "feature":    feature_names[:len(importances)],
            "importance": importances,
        })

        # Add descriptor category (HOG / LBP / DCT)
        df["descriptor"] = df["feature"].apply(lambda x: x.split("_")[0])

        df = df.sort_values("importance", ascending=False).reset_index(drop=True)
        return df

    def save(self, path: str):
        """Save the trained pipeline to disk."""
        joblib.dump({
            "pipeline": self.pipeline,
            "best_params": self.best_params,
        }, path)
        print(f"  RF model saved to {path}")

    def load(self, path: str):
        """Load a trained pipeline from disk."""
        data = joblib.load(path)
        self.pipeline = data["pipeline"]
        self.best_params = data["best_params"]
        print(f"  RF model loaded from {path}")
