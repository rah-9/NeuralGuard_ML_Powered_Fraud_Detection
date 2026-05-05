"""
risk.py — Risk scoring engine.

Converts model probability into a 0–100 risk score and assigns
human-readable risk labels (SAFE / AT RISK / CRITICAL).
"""

import pandas as pd
import numpy as np
from typing import List


def compute_risk_score(fraud_proba: np.ndarray) -> np.ndarray:
    """
    Map fraud probability [0, 1] → risk score [0, 100].

    Parameters
    ----------
    fraud_proba : np.ndarray
        P(fraud) from the classifier.

    Returns
    -------
    np.ndarray
        Risk scores in [0, 100].
    """
    return np.round(fraud_proba * 100, 2)


def assign_risk_label(risk_score: float) -> str:
    """
    Categorize a single risk score into a human-readable label.

    Categories
    ----------
    0–30   → SAFE
    30–70  → AT RISK
    70–100 → CRITICAL
    """
    if risk_score <= 30:
        return "SAFE"
    elif risk_score <= 70:
        return "AT RISK"
    else:
        return "CRITICAL"


def assign_risk_labels(risk_scores: np.ndarray) -> List[str]:
    """Vectorized label assignment across an array of scores."""
    return [assign_risk_label(s) for s in risk_scores]


def build_risk_dataframe(
    amounts: pd.Series,
    fraud_proba: np.ndarray,
    anomaly_flags: np.ndarray,
    true_labels: pd.Series = None,
) -> pd.DataFrame:
    """
    Assemble the final risk-assessment DataFrame.

    Parameters
    ----------
    amounts : pd.Series
        Original (or scaled) transaction amounts.
    fraud_proba : np.ndarray
        P(fraud) from the classifier.
    anomaly_flags : np.ndarray
        Binary anomaly flags from Isolation Forest.
    true_labels : pd.Series, optional
        Ground-truth class labels (for evaluation).

    Returns
    -------
    pd.DataFrame
        Columns: amount, risk_score, risk_label, anomaly_flag[, true_label]
    """
    risk_scores = compute_risk_score(fraud_proba)
    risk_labels = assign_risk_labels(risk_scores)

    result = pd.DataFrame({
        "amount": amounts.values,
        "risk_score": risk_scores,
        "risk_label": risk_labels,
        "anomaly_flag": anomaly_flags,
    })

    if true_labels is not None:
        result["true_label"] = true_labels.values

    return result


def save_predictions(df: pd.DataFrame, path: str) -> None:
    """Persist predictions to CSV."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[OK] Predictions saved -> {path}")


# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    scores = np.array([0.05, 0.45, 0.88, 0.30, 0.71])
    labels = assign_risk_labels(compute_risk_score(scores))
    for s, l in zip(scores, labels):
        print(f"  P(fraud)={s:.2f}  →  score={s*100:.0f}  →  {l}")
