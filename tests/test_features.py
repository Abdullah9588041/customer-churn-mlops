"""Tests for feature engineering."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from churn import features  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "Telco-Customer-Churn.csv")


@pytest.fixture(scope="module")
def raw():
    return features.load_raw(DATA)


@pytest.fixture(scope="module")
def clean(raw):
    return features.clean(raw)


def test_load_row_count(raw):
    assert len(raw) == 7043
    assert "Churn" in raw.columns


def test_clean_totalcharges_no_missing(clean):
    # The 11 blank TotalCharges values must be gone after cleaning.
    assert clean["TotalCharges"].isna().sum() == 0
    assert (clean["TotalCharges"] > 0).all()


def test_clean_target_binary(clean):
    assert set(clean["Churn"].unique()) <= {0, 1}
    assert clean["Churn"].mean() == pytest.approx(0.265, abs=0.01)


def test_clean_drops_id(clean):
    assert "customerID" not in clean.columns


def test_clean_normalizes_service_nas(clean):
    for col in ["MultipleLines", "OnlineSecurity", "TechSupport"]:
        assert "No phone service" not in clean[col].unique()
        assert "No internet service" not in clean[col].unique()


def test_engineer_adds_features(clean):
    df = features.engineer(clean)
    for col in ["tenure_bucket", "avg_monthly", "service_count", "is_new",
                "is_monthly_contract", "is_electronic_check", "monthly_x_tenure"]:
        assert col in df.columns
    assert df["avg_monthly"].isna().sum() == 0
    assert df["service_count"].between(0, 6).all()


def test_preprocessor_no_nans_and_names(clean):
    df = features.engineer(clean)
    X = df.drop(columns=["Churn"])
    pre = features.make_preprocessor()
    Xt = pre.fit_transform(X)
    assert not np.isnan(Xt).any()
    names = features.feature_names(pre)
    assert len(names) == Xt.shape[1]
    assert len(names) > 30  # one-hot expansion happened


def test_prepare_split_shapes():
    X, y = features.prepare(DATA)
    assert len(X) == len(y) == 7043
    assert "Churn" not in X.columns
