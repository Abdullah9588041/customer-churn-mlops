"""Evaluation: standard metrics + cost-based threshold selection.

Business model (assumptions stated explicitly — change them and the optimal
threshold moves, which is the point):
- A retention offer (discount / outreach call) costs ``OFFER_COST`` per contacted customer.
- A contacted customer who would have churned accepts the offer and stays with
  probability ``SAVE_RATE``.
- A retained customer is worth ``RETENTION_VALUE`` (their expected remaining
  lifetime value; here approximated per-customer as 6x MonthlyCharges).
- Contacting a customer who would NOT have churned wastes the offer cost.

Expected value of contacting customer i with churn probability p_i:
    EV_i = p_i * SAVE_RATE * V_i - OFFER_COST
Contact iff EV_i > 0  <=>  p_i > OFFER_COST / (SAVE_RATE * V_i).

Because V_i varies per customer, we sweep a global threshold on a validation set
and pick the one maximizing total expected value. This is the "metrics tied to
the cost of being wrong" discipline: the threshold is a business decision, not 0.5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)

OFFER_COST = 20.0      # USD per retention offer (assumption)
SAVE_RATE = 0.30       # P(stay | offer, would-have-churned) (assumption)
LTV_MONTHS = 6         # retained-customer value = 6x monthly charges (assumption)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    threshold: float = 0.5) -> dict:
    """Standard classification metrics at a fixed threshold."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def expected_value(y_true: np.ndarray, y_prob: np.ndarray,
                   monthly_charges: np.ndarray, threshold: float,
                   offer_cost: float = OFFER_COST, save_rate: float = SAVE_RATE,
                   ltv_months: int = LTV_MONTHS) -> dict:
    """Total expected business value of the retention campaign at ``threshold``.

    Uses true labels (evaluation on labeled validation data): a contacted true
    churner yields save_rate * V - cost; a contacted non-churner yields -cost.
    """
    y_true = np.asarray(y_true)
    contacted = np.asarray(y_prob) >= threshold
    value = np.asarray(monthly_charges) * ltv_months
    n_contacted = int(contacted.sum())
    if n_contacted == 0:
        return {"threshold": float(threshold), "n_contacted": 0,
                "expected_value": 0.0, "expected_saves": 0.0}
    saved_value = (y_true[contacted] * save_rate * value[contacted]).sum()
    total = float(saved_value - offer_cost * n_contacted)
    return {
        "threshold": float(threshold),
        "n_contacted": n_contacted,
        "expected_value": total,
        "expected_saves": float((y_true[contacted] * save_rate).sum()),
    }


def threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray,
                    monthly_charges: np.ndarray,
                    thresholds: np.ndarray | None = None) -> pd.DataFrame:
    """Evaluate expected value across thresholds; returns sorted DataFrame."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    rows = [expected_value(y_true, y_prob, monthly_charges, t) for t in thresholds]
    df = pd.DataFrame(rows)
    return df.sort_values("expected_value", ascending=False).reset_index(drop=True)


def error_segments(X_raw: pd.DataFrame, y_true: np.ndarray,
                   y_prob: np.ndarray, threshold: float) -> pd.DataFrame:
    """Where is the model wrong? Churn rate among FP/FN vs overall, by segment."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    df = X_raw.copy()
    df["_true"] = np.asarray(y_true)
    df["_pred"] = y_pred
    df["_fp"] = ((df["_true"] == 0) & (df["_pred"] == 1)).astype(int)
    df["_fn"] = ((df["_true"] == 1) & (df["_pred"] == 0)).astype(int)
    rows = []
    for col in ["Contract", "InternetService", "tenure_bucket", "PaymentMethod"]:
        if col not in df.columns:
            continue
        g = df.groupby(col, observed=True).agg(
            n=("_true", "size"),
            churn_rate=("_true", "mean"),
            fp_rate=("_fp", "mean"),
            fn_rate=("_fn", "mean"),
        )
        for seg, r in g.iterrows():
            rows.append({"segment_col": col, "segment": str(seg),
                         "n": int(r["n"]), "churn_rate": float(r["churn_rate"]),
                         "fp_rate": float(r["fp_rate"]), "fn_rate": float(r["fn_rate"])})
    return pd.DataFrame(rows)
