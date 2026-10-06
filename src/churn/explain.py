"""SHAP explanations for the champion tree model."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap


def shap_values_tree(model, X: np.ndarray, feature_names: list[str],
                     nsample: int = 500, seed: int = 42):
    """Compute SHAP values (subsampled background for speed)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(nsample, len(X)), replace=False)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X[idx])
    # Binary classifiers: shap returns list per class in older versions.
    if isinstance(sv, list):
        sv = sv[1]
    return np.asarray(sv), X[idx]


def summary_plot(sv: np.ndarray, X_sample: np.ndarray,
                 feature_names: list[str], path: str, top_n: int = 15):
    """Beeswarm summary plot, saved to ``path``."""
    plt.figure(figsize=(9, 7))
    shap.summary_plot(sv, X_sample, feature_names=feature_names,
                      max_display=top_n, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def dependence_plot(sv: np.ndarray, X_sample: np.ndarray,
                    feature_names: list[str], feature: str, path: str):
    """Dependence plot for one feature, saved to ``path``."""
    j = feature_names.index(feature)
    plt.figure(figsize=(8, 5))
    shap.dependence_plot(j, sv, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def top_features(sv: np.ndarray, feature_names: list[str], top_n: int = 10) -> list[tuple[str, float]]:
    """Mean |SHAP| ranking."""
    importance = np.abs(sv).mean(axis=0)
    order = np.argsort(importance)[::-1][:top_n]
    return [(feature_names[i], float(importance[i])) for i in order]
