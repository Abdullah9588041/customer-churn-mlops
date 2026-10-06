"""Probability calibration.

A churn model that outputs 0.7 should be right ~70% of the time — otherwise the
probabilities cannot drive business decisions (expected-value thresholding in
``evaluation.py`` assumes calibrated probabilities). We calibrate with
``CalibratedClassifierCV`` (isotonic regression by default: non-parametric, so it
can fix arbitrary miscalibration given enough data) and check the Brier score
plus reliability diagrams before/after.
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

SEED = 42


def calibrate(model, X_train: np.ndarray, y_train: np.ndarray,
              method: str = "isotonic", cv: int = 3):
    """Fit CalibratedClassifierCV around an already-configured estimator."""
    cal = CalibratedClassifierCV(estimator=model, method=method, cv=cv)
    cal.fit(X_train, y_train)
    return cal


def reliability_data(y_true: np.ndarray, y_prob: np.ndarray,
                     n_bins: int = 10) -> dict:
    """Bin predictions; return bin centers, empirical churn rates, counts."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)
    centers, frac_pos, counts = [], [], []
    for b in range(n_bins):
        mask = idx == b
        counts.append(int(mask.sum()))
        if mask.sum() == 0:
            centers.append(float((edges[b] + edges[b + 1]) / 2))
            frac_pos.append(np.nan)
        else:
            centers.append(float(y_prob[mask].mean()))
            frac_pos.append(float(y_true[mask].mean()))
    return {
        "bin_centers": centers,
        "empirical_rate": frac_pos,
        "counts": counts,
        "brier": float(brier_score_loss(y_true, y_prob)),
    }
