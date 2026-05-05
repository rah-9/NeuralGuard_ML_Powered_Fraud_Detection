"""
anomaly.py — Isolation Forest–based anomaly detection.

Trains an IsolationForest on the feature set and flags transactions
as anomalies (1) or normal (0).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONTAMINATION = 0.01   # ≈ fraud rate in the creditcard dataset
RANDOM_STATE = 42


def train_isolation_forest(
    X: pd.DataFrame,
    contamination: float = CONTAMINATION,
    random_state: int = RANDOM_STATE,
) -> IsolationForest:
    """
    Fit an Isolation Forest for unsupervised anomaly detection.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (no target column).
    contamination : float
        Expected proportion of anomalies in the dataset.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    IsolationForest
        Fitted model.
    """
    iso = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    iso.fit(X)
    return iso


def flag_anomalies(
    iso_model: IsolationForest,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Predict anomaly flags.

    IsolationForest returns -1 for anomalies and 1 for normal.
    We convert to: 1 = anomaly, 0 = normal.

    Parameters
    ----------
    iso_model : IsolationForest
        Fitted isolation forest.
    X : pd.DataFrame
        Feature matrix to score.

    Returns
    -------
    np.ndarray
        Binary array — 1 if anomaly, else 0.
    """
    raw = iso_model.predict(X)
    return np.where(raw == -1, 1, 0)


def run_anomaly_detection(
    X: pd.DataFrame,
) -> Tuple[IsolationForest, np.ndarray]:
    """
    End-to-end anomaly detection pipeline.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.

    Returns
    -------
    iso_model : IsolationForest
    anomaly_flags : np.ndarray  (1 = anomaly, 0 = normal)
    """
    iso_model = train_isolation_forest(X)
    anomaly_flags = flag_anomalies(iso_model, X)
    return iso_model, anomaly_flags


# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from preprocess import run_preprocessing

    df_eng, X_train, X_test, y_train, y_test = run_preprocessing()
    X_all = df_eng.drop(columns=["Class"])
    iso, flags = run_anomaly_detection(X_all)
    print(f"Total anomalies flagged: {flags.sum()} / {len(flags)}")
    print(f"Anomaly rate           : {flags.mean():.4%}")
