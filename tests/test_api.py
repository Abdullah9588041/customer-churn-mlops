"""API tests with FastAPI TestClient (no server needed)."""

import os
import pickle
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "Telco-Customer-Churn.csv")

CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Train a tiny real bundle and point the API at it."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    from churn import features

    X_raw, y = features.prepare(DATA)
    Xs, _, ys, _ = train_test_split(
        X_raw, y, train_size=600, random_state=0, stratify=y)
    pre = features.make_preprocessor()
    Xt = pre.fit_transform(Xs)
    mdl = LogisticRegression(max_iter=500, random_state=0).fit(Xt, ys.to_numpy())

    bundle_path = tmp_path_factory.mktemp("bundle") / "champion.pkl"
    with open(bundle_path, "wb") as f:
        pickle.dump({
            "model": mdl,
            "preprocessor": pre,
            "threshold": 0.5,
            "model_name": "logistic_regression",
            "trained_at": "test",
        }, f)

    import churn.api as api
    api.BUNDLE_PATH = str(bundle_path)
    api._bundle = None
    return TestClient(api.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"] == "logistic_regression"


def test_predict_single(client):
    r = client.post("/predict", json=CUSTOMER)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in (0, 1)


def test_predict_invalid_rejected(client):
    bad = dict(CUSTOMER)
    bad["Contract"] = "Forever"
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_batch(client):
    r = client.post("/predict/batch", json=[CUSTOMER, CUSTOMER])
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 2


def test_predict_batch_empty_rejected(client):
    r = client.post("/predict/batch", json=[])
    assert r.status_code == 422
