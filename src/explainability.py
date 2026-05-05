"""
explainability.py — SHAP-based model explainability for XGBoost.

Generates SHAP values and summary plots (bar chart of top features).
"""

import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from typing import Tuple, List


def compute_shap_values(model: XGBClassifier, X: pd.DataFrame) -> Tuple[shap.Explainer, np.ndarray]:
    """
    Compute SHAP values using TreeExplainer.

    Returns
    -------
    explainer : shap.TreeExplainer
    shap_values : np.ndarray  (shape: n_samples x n_features)
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values


def get_top_features(shap_values: np.ndarray, feature_names: List[str], top_n: int = 10) -> pd.DataFrame:
    """
    Rank features by mean |SHAP value| and return top N.

    Returns
    -------
    pd.DataFrame with columns: feature, importance
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({"feature": feature_names, "importance": mean_abs})
    importance_df = importance_df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
    return importance_df


def generate_shap_summary_plot(shap_values: np.ndarray, X: pd.DataFrame, save_path: str = None) -> None:
    """Generate a SHAP bar summary plot of top 10 features."""
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", max_display=10, show=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def explain_top_features(importance_df: pd.DataFrame) -> str:
    """Return a plain-English explanation of the top 3 features."""
    lines = ["### Top Feature Explanations\n"]
    for i, row in importance_df.head(3).iterrows():
        feat = row["feature"]
        imp = row["importance"]
        lines.append(f"**{i+1}. {feat}** (importance: {imp:.4f})")
        lines.append(f"   This feature has the {_ordinal(i+1)} highest impact on fraud predictions. "
                      f"Higher absolute SHAP values indicate stronger influence on the model's decision.\n")
    return "\n".join(lines)


def _ordinal(n: int) -> str:
    """Return ordinal string for a number (1st, 2nd, 3rd, ...)."""
    s = {1: "1st", 2: "2nd", 3: "3rd"}
    return s.get(n, f"{n}th")
