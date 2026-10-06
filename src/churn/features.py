"""Feature engineering for the IBM Telco Customer Churn dataset.

Data quirks handled explicitly:
- ``TotalCharges`` is stored as a string column and contains blank strings for
  customers with ``tenure == 0`` (new customers with no billing history yet).
  These are coerced to NaN and imputed with the median.
- ``customerID`` is a pure identifier and is dropped (it must never be a feature).
- Several service columns use the value ``"No phone service"`` / ``"No internet
  service"`` instead of ``"No"``; these are normalized to ``"No"`` so one-hot
  encoding does not create redundant columns.

Feature-engineering story (why each derived feature exists):
- ``tenure_bucket``: churn risk is strongly non-linear in tenure (new customers
  churn far more); buckets capture this without forcing a linear/log form.
- ``avg_monthly`` (TotalCharges / tenure): separates "expensive plan" from
  "long tenure" effects that raw TotalCharges conflates.
- ``service_count``: number of add-on services; engagement proxy — customers
  with more services are stickier (testable hypothesis, checked in EDA).
- ``is_new`` (tenure <= 12): explicit flag for the high-risk cohort.
- ``is_monthly_contract`` / ``is_electronic_check``: the two categories with the
  highest observed churn in EDA; kept as explicit flags for interpretability.
- ``monthly_x_tenure``: interaction proxy for lifetime value exposure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"
ID_COL = "customerID"

NUMERIC_RAW = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_RAW = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

# Columns that use "No <service> service" instead of "No".
SERVICE_NA_MAP = {
    "MultipleLines": "No phone service",
    "OnlineSecurity": "No internet service",
    "OnlineBackup": "No internet service",
    "DeviceProtection": "No internet service",
    "TechSupport": "No internet service",
    "StreamingTV": "No internet service",
    "StreamingMovies": "No internet service",
}

ADDON_SERVICES = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]


def load_raw(path: str) -> pd.DataFrame:
    """Load the raw CSV exactly as downloaded."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw data: fix TotalCharges blanks, normalize service NAs, drop ID, encode target."""
    df = df.copy()
    # TotalCharges blanks -> NaN (these are tenure-0 customers), then numeric.
    df["TotalCharges"] = df["TotalCharges"].replace(r"^\s*$", np.nan, regex=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    median_total = df["TotalCharges"].median()
    df["TotalCharges"] = df["TotalCharges"].fillna(median_total)

    # Normalize "No phone/internet service" -> "No".
    for col, na_val in SERVICE_NA_MAP.items():
        df[col] = df[col].replace(na_val, "No")

    df = df.drop(columns=[ID_COL], errors="ignore")
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features. Expects the output of :func:`clean`."""
    df = df.copy()
    df["tenure_bucket"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 200],
        labels=["0-12", "12-24", "24-48", "48+"],
    ).astype(str)
    # tenure == 0 only occurred via blanks already imputed; guard anyway.
    df["avg_monthly"] = df["TotalCharges"] / df["tenure"].replace(0, np.nan)
    df["avg_monthly"] = df["avg_monthly"].fillna(df["MonthlyCharges"])
    df["service_count"] = (df[ADDON_SERVICES] == "Yes").sum(axis=1).astype(int)
    df["is_new"] = (df["tenure"] <= 12).astype(int)
    df["is_monthly_contract"] = (df["Contract"] == "Month-to-month").astype(int)
    df["is_electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    df["monthly_x_tenure"] = df["MonthlyCharges"] * df["tenure"]
    return df


ENGINEERED_NUMERIC = [
    "tenure", "MonthlyCharges", "TotalCharges", "avg_monthly",
    "service_count", "is_new", "is_monthly_contract",
    "is_electronic_check", "monthly_x_tenure",
]
ENGINEERED_CATEGORICAL = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "tenure_bucket",
]


def make_preprocessor() -> ColumnTransformer:
    """Build the preprocessing transformer (fit on train only)."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, ENGINEERED_NUMERIC),
        ("cat", categorical, ENGINEERED_CATEGORICAL),
    ])


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return output feature names after fitting."""
    names: list[str] = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "num":
            names.extend(cols)
        elif name == "cat":
            ohe = trans.named_steps["onehot"]
            names.extend(ohe.get_feature_names_out(cols).tolist())
    return names


def prepare(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Full pipeline: load -> clean -> engineer -> split X/y."""
    df = engineer(clean(load_raw(path)))
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    return X, y
