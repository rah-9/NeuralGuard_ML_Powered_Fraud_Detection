"""
preprocess.py — Data loading, feature engineering, normalization, and train/test split.

Handles the creditcard.csv dataset:
  - Engineers hour-of-day from the 'Time' column
  - Drops the raw 'Time' column
  - Normalizes 'Amount' using StandardScaler
  - Performs stratified 80/20 train/test split
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "creditcard.csv")
TEST_SIZE = 0.20
RANDOM_STATE = 42


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the credit-card transaction dataset from CSV.

    Parameters
    ----------
    path : str
        Absolute or relative path to creditcard.csv.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with all original columns.
    """
    df = pd.read_csv(path)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering:
      1. Convert 'Time' (seconds from first transaction) to hour-of-day (0-23).
      2. Drop the original 'Time' column.
      3. Normalize 'Amount' using StandardScaler.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe containing 'Time' and 'Amount' columns.

    Returns
    -------
    pd.DataFrame
        Transformed dataframe with 'hour' replacing 'Time' and scaled 'Amount'.
    """
    df = df.copy()

    # --- Hour-of-day from Time (seconds elapsed → cyclic hour) ---
    df["hour"] = np.floor((df["Time"] % 86400) / 3600).astype(int)
    df.drop(columns=["Time"], inplace=True)

    # --- Normalize Amount ---
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])

    return df


def split_data(
    df: pd.DataFrame,
    target_col: str = "Class",
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train/test split.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered dataframe (must contain `target_col`).
    target_col : str
        Name of the binary target column.
    test_size : float
        Fraction reserved for the test set.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def run_preprocessing(
    path: str = DATA_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    End-to-end preprocessing pipeline.

    Returns
    -------
    df_engineered : pd.DataFrame
        Full feature-engineered dataset (for anomaly detection / risk scoring).
    X_train, X_test, y_train, y_test
        Split datasets ready for model training.
    """
    df_raw = load_data(path)
    df_eng = engineer_features(df_raw)
    X_train, X_test, y_train, y_test = split_data(df_eng)
    return df_eng, X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df_eng, X_train, X_test, y_train, y_test = run_preprocessing()
    print(f"Engineered shape : {df_eng.shape}")
    print(f"Train / Test     : {X_train.shape} / {X_test.shape}")
    print(f"Fraud rate (train): {y_train.mean():.4%}")
    print(f"Fraud rate (test) : {y_test.mean():.4%}")
