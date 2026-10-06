"""Model definitions and cross-validation for churn prediction.

Class imbalance (~26.5% churn) is handled explicitly per estimator:
- LogisticRegression / RandomForest: ``class_weight`` re-weights the loss so the
  minority class is not ignored.
- XGBoost / LightGBM: ``scale_pos_weight = n_neg / n_pos`` (equivalent
  re-weighting for gradient boosting).

All models are evaluated with stratified K-fold CV and proper scoring rules
(ROC-AUC, PR-AUC, log-loss) — accuracy alone would be misleading on imbalanced data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

SEED = 42


def scale_pos_weight(y: pd.Series | np.ndarray) -> float:
    """Negative-to-positive ratio for gradient-boosting re-weighting."""
    y = np.asarray(y)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return n_neg / max(n_pos, 1)


def get_model_configs(y_train: pd.Series) -> dict[str, object]:
    """Return {name: unfitted estimator} with imbalance handling baked in."""
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    spw = scale_pos_weight(y_train)
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEED
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample",
            random_state=SEED, n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            scale_pos_weight=spw, random_state=SEED,
            n_jobs=-1, eval_metric="logloss",
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            scale_pos_weight=spw, random_state=SEED,
            n_jobs=-1, verbose=-1,
        ),
    }


SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "f1": "f1",
    "log_loss": "neg_log_loss",
}


def cross_validate_models(
    configs: dict[str, object],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Stratified CV for every model; returns mean/std per metric."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    rows = []
    for name, model in configs.items():
        res = cross_validate(model, X, y, cv=cv, scoring=SCORING, n_jobs=1)
        row = {"model": name}
        for metric in SCORING:
            vals = res[f"test_{metric}"]
            if metric == "log_loss":  # stored as negative
                vals = -vals
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_std"] = float(np.std(vals))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)
