"""Hyperparameter tuning with Optuna.

Why Optuna over RandomizedSearchCV: Tree-structured Parzen Estimator (TPE)
sampling focuses the budget on promising regions instead of sampling blindly,
and median pruning stops unpromising trials early. With a fixed seed the search
is reproducible.
"""

from __future__ import annotations

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score

SEED = 42


def _xgb_space(trial: optuna.Trial, spw: float) -> dict:
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 200, 800),
        max_depth=trial.suggest_int("max_depth", 3, 8),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        scale_pos_weight=spw,
        random_state=SEED, n_jobs=-1, eval_metric="logloss",
    )


def _lgbm_space(trial: optuna.Trial, spw: float) -> dict:
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=trial.suggest_int("n_estimators", 200, 800),
        num_leaves=trial.suggest_int("num_leaves", 15, 127),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
        scale_pos_weight=spw,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )


def tune(model_name: str, X: np.ndarray, y: np.ndarray,
         n_trials: int = 30, n_splits: int = 3) -> tuple[dict, float]:
    """Tune ``model_name`` ('xgboost' | 'lightgbm'); returns (best_params, best_cv_auc)."""
    from .models import scale_pos_weight
    spw = scale_pos_weight(y)
    space_fn = {"xgboost": _xgb_space, "lightgbm": _lgbm_space}[model_name]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    def objective(trial: optuna.Trial) -> float:
        model = space_fn(trial, spw)
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, float(study.best_value)


def build_tuned(model_name: str, params: dict, spw: float):
    """Rebuild the estimator from tuned params (adds fixed imbalance/seed args)."""
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    cls = {"xgboost": XGBClassifier, "lightgbm": LGBMClassifier}[model_name]
    extra = {"scale_pos_weight": spw, "random_state": SEED, "n_jobs": -1}
    if model_name == "xgboost":
        extra["eval_metric"] = "logloss"
    else:
        extra["verbose"] = -1
    return cls(**{**params, **extra})
