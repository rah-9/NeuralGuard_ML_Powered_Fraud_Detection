"""
model.py — XGBoost fraud classifier training, evaluation, and persistence.

Uses scale_pos_weight to handle class imbalance (equivalent to class_weight='balanced').
Outputs probability scores via predict_proba() and comprehensive evaluation metrics.
"""

import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
from typing import Dict, Any, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
RANDOM_STATE = 42


def _compute_scale_pos_weight(y: pd.Series) -> float:
    """
    Compute scale_pos_weight = n_negative / n_positive.
    This is XGBoost's mechanism for handling class imbalance,
    analogous to class_weight='balanced' in sklearn estimators.
    """
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    return neg / pos


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    save_path: str = MODEL_PATH,
) -> XGBClassifier:
    """
    Train an XGBClassifier with balanced class weighting.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels (0 = normal, 1 = fraud).
    save_path : str
        Where to persist the trained model via joblib.

    Returns
    -------
    XGBClassifier
        Fitted model.
    """
    spw = _compute_scale_pos_weight(y_train)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=spw,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # Persist model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    print(f"[OK] Model saved -> {save_path}")

    return model


def load_model(path: str = MODEL_PATH) -> XGBClassifier:
    """Load a previously saved XGBClassifier."""
    return joblib.load(path)


def predict(model: XGBClassifier, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate class predictions and probability scores.

    Returns
    -------
    y_pred : np.ndarray   — binary predictions
    y_proba : np.ndarray  — P(fraud) for each sample
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return y_pred, y_proba


def evaluate_model(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation metrics.

    Returns
    -------
    dict with keys:
        confusion_matrix, classification_report (dict),
        roc_auc, pr_auc, roc_curve, pr_curve
    """
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)

    return {
        "confusion_matrix": cm,
        "classification_report": cr,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "roc_curve": {"fpr": fpr, "tpr": tpr},
        "pr_curve": {"precision": precision, "recall": recall},
    }


# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from preprocess import run_preprocessing

    _, X_train, X_test, y_train, y_test = run_preprocessing()
    model = train_model(X_train, y_train)
    y_pred, y_proba = predict(model, X_test)
    metrics = evaluate_model(y_test, y_pred, y_proba)

    print(f"\nROC-AUC : {metrics['roc_auc']:.4f}")
    print(f"PR-AUC  : {metrics['pr_auc']:.4f}")
    print(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}")
