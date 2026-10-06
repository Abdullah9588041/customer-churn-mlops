"""Tests for models, calibration, and evaluation (fast, on a data subsample)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from churn import calibration, evaluation, features, models  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "Telco-Customer-Churn.csv")


@pytest.fixture(scope="module")
def prepared():
    X_raw, y = features.prepare(DATA)
    # Deterministic subsample for speed; stratify to keep both classes.
    from sklearn.model_selection import train_test_split
    Xs, _, ys, _ = train_test_split(
        X_raw, y, train_size=800, random_state=0, stratify=y)
    pre = features.make_preprocessor()
    Xt = pre.fit_transform(Xs)
    return Xt, ys.to_numpy(), Xs


def test_scale_pos_weight(prepared):
    _, y, _ = prepared
    spw = models.scale_pos_weight(y)
    assert spw == pytest.approx((y == 0).sum() / (y == 1).sum())


def test_get_model_configs_have_imbalance_handling(prepared):
    _, y, _ = prepared
    configs = models.get_model_configs(pd.Series(y))
    assert set(configs) == {"logistic_regression", "random_forest", "xgboost", "lightgbm"}
    assert configs["logistic_regression"].class_weight == "balanced"
    assert configs["xgboost"].scale_pos_weight > 1.0


def test_cross_validate_returns_expected_metrics(prepared):
    Xt, y, _ = prepared
    configs = {"logistic_regression": models.get_model_configs(pd.Series(y))["logistic_regression"]}
    df = models.cross_validate_models(configs, Xt, y, n_splits=3)
    assert list(df["model"]) == ["logistic_regression"]
    for m in ["roc_auc_mean", "pr_auc_mean", "f1_mean", "log_loss_mean"]:
        assert m in df.columns
    assert 0.5 < df["roc_auc_mean"].iloc[0] <= 1.0


def test_calibration_improves_or_matches_brier(prepared):
    from sklearn.linear_model import LogisticRegression
    Xt, y, _ = prepared
    base = LogisticRegression(max_iter=500, random_state=0).fit(Xt, y)
    cal = calibration.calibrate(base, Xt, y, method="isotonic", cv=2)
    p_raw = base.predict_proba(Xt)[:, 1]
    p_cal = cal.predict_proba(Xt)[:, 1]
    rel = calibration.reliability_data(y, p_cal, n_bins=5)
    assert 0.0 <= rel["brier"] <= 1.0
    assert len(rel["bin_centers"]) == 5
    # In-sample isotonic should not be worse than uncalibrated.
    from sklearn.metrics import brier_score_loss
    assert rel["brier"] <= brier_score_loss(y, p_raw) + 1e-9


def test_compute_metrics_keys(prepared):
    _, y, _ = prepared
    rng = np.random.default_rng(0)
    proba = rng.random(len(y))
    m = evaluation.compute_metrics(y, proba, threshold=0.5)
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
              "brier", "tn", "fp", "fn", "tp"]:
        assert k in m
    assert m["tn"] + m["fp"] + m["fn"] + m["tp"] == len(y)


def test_threshold_sweep_picks_max_ev(prepared):
    _, y, _ = prepared
    rng = np.random.default_rng(1)
    proba = rng.random(len(y))
    charges = np.full(len(y), 70.0)
    sweep = evaluation.threshold_sweep(y, proba, charges)
    assert sweep["expected_value"].is_monotonic_decreasing  # sorted desc
    # Best row really is the max.
    assert sweep["expected_value"].iloc[0] == sweep["expected_value"].max()


def test_error_segments_structure(prepared):
    Xt, y, Xs = prepared
    rng = np.random.default_rng(2)
    proba = rng.random(len(y))
    seg = evaluation.error_segments(Xs, y, proba, threshold=0.5)
    assert {"segment_col", "segment", "n", "churn_rate", "fp_rate", "fn_rate"} <= set(seg.columns)
    assert (seg["n"] > 0).all()
