"""End-to-end churn analysis: EDA -> CV comparison -> tuning -> calibration ->
threshold selection -> SHAP -> champion bundle + MLflow tracking.

Produces: results/figures/*.png, results/metrics.json, models/champion.pkl,
mlruns/ (local MLflow tracking).

Run:  .venv/bin/python scripts/run_analysis.py
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from churn import features, models, tuning, calibration, evaluation, explain  # noqa: E402

SEED = 42
ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(ROOT, "data", "Telco-Customer-Churn.csv")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(RESULTS, "figures")
MODELS_DIR = os.path.join(ROOT, "models")

os.makedirs(FIGURES, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

import mlflow  # noqa: E402

mlflow.set_tracking_uri("sqlite:///" + os.path.join(ROOT, "mlruns.db"))
mlflow.set_experiment("customer-churn")


def eda(df_raw: pd.DataFrame, df: pd.DataFrame) -> dict:
    """Exploratory figures + summary stats. Returns key EDA findings."""
    out: dict = {}
    out["n_rows"] = int(len(df))
    out["churn_rate"] = float(df["Churn"].mean())
    out["totalcharges_blanks"] = int((df_raw["TotalCharges"].astype(str).str.strip() == "").sum())

    # 1. Churn rate by contract / internet service / tenure bucket
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col in zip(axes, ["Contract", "InternetService", "tenure_bucket"]):
        rates = df.groupby(col, observed=True)["Churn"].mean().sort_values()
        rates.plot(kind="barh", ax=ax, color="steelblue")
        ax.set_xlabel("Churn rate")
        ax.set_title(f"Churn rate by {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "eda_churn_segments.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Tenure / MonthlyCharges distributions by churn
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, col in zip(axes, ["tenure", "MonthlyCharges"]):
        for churn, lab in [(0, "Stayed"), (1, "Churned")]:
            ax.hist(df.loc[df["Churn"] == churn, col], bins=30,
                    alpha=0.6, label=lab, density=True)
        ax.set_xlabel(col)
        ax.legend()
        ax.set_title(f"{col} distribution by churn")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "eda_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Correlation of numeric features with churn
    num = df[features.ENGINEERED_NUMERIC + ["Churn"]].corr(numeric_only=True)["Churn"].drop("Churn")
    num = num.sort_values()
    plt.figure(figsize=(8, 5))
    num.plot(kind="barh", color="steelblue")
    plt.xlabel("Correlation with churn")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "eda_correlations.png"), dpi=150, bbox_inches="tight")
    plt.close()
    out["top_corr"] = {k: float(v) for k, v in
                        num.abs().sort_values(ascending=False).head(5).round(3).to_dict().items()}
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    df_raw = features.load_raw(DATA_PATH)
    df = features.engineer(features.clean(df_raw))
    eda_out = eda(df_raw, df)
    print(f"EDA: n={eda_out['n_rows']}, churn_rate={eda_out['churn_rate']:.3f}, "
          f"TotalCharges blanks={eda_out['totalcharges_blanks']}")

    X_raw, y = features.prepare(DATA_PATH)

    # Train/validation/test split: 60/20/20 stratified. Test is touched ONCE.
    from sklearn.model_selection import train_test_split
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X_raw, y, test_size=0.4, random_state=SEED, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=SEED, stratify=y_tmp)

    pre = features.make_preprocessor()
    Xtr = pre.fit_transform(X_tr)
    Xva = pre.transform(X_val)
    Xte = pre.transform(X_te)
    feat_names = features.feature_names(pre)
    print(f"Features after encoding: {Xtr.shape[1]}")

    ytr = y_tr.to_numpy()
    yva = y_val.to_numpy()
    yte = y_te.to_numpy()

    # --- 1. Cross-validated model comparison (on train only) ---
    configs = models.get_model_configs(y_tr)
    cv_df = models.cross_validate_models(configs, Xtr, ytr, n_splits=5)
    print(cv_df[["model", "roc_auc_mean", "pr_auc_mean", "f1_mean"]].to_string(index=False))
    cv_records = [{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                   for k, v in row.items()}
                  for row in cv_df.round(4).to_dict(orient="records")]

    # --- 2. Tune the two gradient boosters (Optuna, 3-fold CV inside) ---
    tuned: dict[str, dict] = {}
    for name in ["xgboost", "lightgbm"]:
        params, best_auc = tuning.tune(name, Xtr, ytr, n_trials=25, n_splits=3)
        tuned[name] = {"params": params, "cv_auc": round(best_auc, 4)}
        print(f"Tuned {name}: cv_auc={best_auc:.4f}")

    # --- 3. Fit all candidates on full train, evaluate on validation ---
    from churn.models import scale_pos_weight
    spw = scale_pos_weight(ytr)
    candidates: dict[str, object] = {
        "logistic_regression": configs["logistic_regression"],
        "random_forest": configs["random_forest"],
        "xgboost_tuned": tuning.build_tuned("xgboost", tuned["xgboost"]["params"], spw),
        "lightgbm_tuned": tuning.build_tuned("lightgbm", tuned["lightgbm"]["params"], spw),
    }
    val_results: dict[str, dict] = {}
    for name, mdl in candidates.items():
        with mlflow.start_run(run_name=name):
            mdl.fit(Xtr, ytr)
            proba = mdl.predict_proba(Xva)[:, 1]
            m = evaluation.compute_metrics(yva, proba, threshold=0.5)
            mlflow.log_params(getattr(mdl, "get_params", lambda: {})())
            mlflow.log_metrics({k: v for k, v in m.items()
                                if isinstance(v, float)})
            val_results[name] = m
            print(f"{name}: val ROC-AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f} "
                  f"F1={m['f1']:.4f} brier={m['brier']:.4f}")

    # --- 4. Champion = best validation ROC-AUC; calibrate it ---
    champion_name = max(val_results, key=lambda k: val_results[k]["roc_auc"])
    print(f"Champion: {champion_name}")
    base_model = candidates[champion_name]
    champion = calibration.calibrate(base_model, Xtr, ytr, method="isotonic", cv=3)

    proba_va_raw = base_model.predict_proba(Xva)[:, 1]
    proba_va = champion.predict_proba(Xva)[:, 1]
    brier_before = float(__import__("sklearn.metrics", fromlist=["brier_score_loss"])
                         .brier_score_loss(yva, proba_va_raw))
    brier_after = float(__import__("sklearn.metrics", fromlist=["brier_score_loss"])
                        .brier_score_loss(yva, proba_va))

    # Reliability diagram before/after calibration
    rel_raw = calibration.reliability_data(yva, proba_va_raw)
    rel_cal = calibration.reliability_data(yva, proba_va)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, rel, title in zip(axes, [rel_raw, rel_cal],
                              ["Uncalibrated", "Calibrated (isotonic)"]):
        xs = [c for c, e in zip(rel["bin_centers"], rel["empirical_rate"])
              if e is not None and not np.isnan(e)]
        ys = [e for e in rel["empirical_rate"] if e is not None and not np.isnan(e)]
        ax.plot(xs, ys, "o-", label="model")
        ax.plot([0, 1], [0, 1], "k--", label="perfect")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Empirical churn rate")
        ax.set_title(f"{title} (Brier={rel['brier']:.4f})")
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "calibration.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- 5. Cost-based threshold on validation ---
    monthly_va = X_val["MonthlyCharges"].to_numpy()
    sweep = evaluation.threshold_sweep(yva, proba_va, monthly_va)
    best = sweep.iloc[0]
    threshold = float(best["threshold"])
    print(f"Best threshold={threshold:.2f} expected_value=${best['expected_value']:,.0f} "
          f"n_contacted={int(best['n_contacted'])}")

    plt.figure(figsize=(8, 5))
    plt.plot(sweep["threshold"], sweep["expected_value"], "o-")
    plt.axvline(threshold, color="red", linestyle="--",
                label=f"best={threshold:.2f}")
    plt.xlabel("Decision threshold")
    plt.ylabel("Expected campaign value (USD)")
    plt.title("Retention campaign expected value vs. threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "cost_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- 6. FINAL test evaluation (single touch) ---
    proba_te = champion.predict_proba(Xte)[:, 1]
    test_metrics = evaluation.compute_metrics(yte, proba_te, threshold=threshold)
    test_ev = evaluation.expected_value(yte, proba_te, X_te["MonthlyCharges"].to_numpy(),
                                        threshold)
    print(f"TEST: ROC-AUC={test_metrics['roc_auc']:.4f} PR-AUC={test_metrics['pr_auc']:.4f} "
          f"F1={test_metrics['f1']:.4f} EV=${test_ev['expected_value']:,.0f}")
    with mlflow.start_run(run_name="champion_" + champion_name):
        mlflow.log_metrics({k: v for k, v in test_metrics.items()
                            if isinstance(v, float)})
        mlflow.log_param("threshold", threshold)

    # Confusion matrix figure
    cm = np.array([[test_metrics["tn"], test_metrics["fp"]],
                   [test_metrics["fn"], test_metrics["tp"]]])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Stay", "Churn"], yticklabels=["Stay", "Churn"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Test confusion matrix (threshold={threshold:.2f})")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- 7. SHAP explanations (tree champion assumed; fallback: skip gracefully) ---
    shap_top: list = []
    try:
        # CalibratedClassifierCV wraps fitted estimators; use a fresh fitted copy for SHAP.
        raw_for_shap = candidates[champion_name]
        if not hasattr(raw_for_shap, "feature_importances_"):
            raise RuntimeError("non-tree champion")
        sv, Xs = explain.shap_values_tree(raw_for_shap, Xtr, feat_names, nsample=500)
        shap_top = explain.top_features(sv, feat_names, top_n=10)
        explain.summary_plot(sv, Xs, feat_names,
                             os.path.join(FIGURES, "shap_summary.png"))
        explain.dependence_plot(sv, Xs, feat_names, shap_top[0][0],
                                os.path.join(FIGURES, "shap_dependence.png"))
        print("SHAP top features:", [f[0] for f in shap_top[:5]])
    except Exception as e:  # noqa: BLE001 - SHAP is explanatory, not critical path
        print(f"SHAP skipped: {e}")

    # --- 8. Error analysis ---
    seg = evaluation.error_segments(X_val, yva, proba_va, threshold)
    seg.to_csv(os.path.join(RESULTS, "error_segments.csv"), index=False)
    worst_fn = seg.sort_values("fn_rate", ascending=False).head(5)
    print("Highest FN-rate segments:\n",
          worst_fn[["segment_col", "segment", "n", "fn_rate"]].to_string(index=False))

    # --- 9. Save champion bundle + metrics ---
    bundle = {
        "model": champion,
        "preprocessor": pre,
        "threshold": threshold,
        "model_name": champion_name,
        "feature_names": feat_names,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "test_metrics": test_metrics,
    }
    with open(os.path.join(MODELS_DIR, "champion.pkl"), "wb") as f:
        pickle.dump(bundle, f)
    with mlflow.start_run(run_name="champion_artifact"):
        # MLflow 3.x requires explicitly trusted types for sklearn-wrapped boosters.
        try:
            from sklearn.calibration import _CalibratedClassifier
            trusted = [_CalibratedClassifier, type(base_model)]
            try:
                from xgboost.core import Booster
                trusted.append(Booster)
            except ImportError:
                pass
            mlflow.sklearn.log_model(champion, name="champion_model",
                                     skops_trusted_types=trusted)
        except Exception as e:  # noqa: BLE001 - artifact logging is auxiliary
            print(f"MLflow model artifact logging skipped: {e}")

    metrics = {
        "seed": SEED,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data": {"source": "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
                 "n_rows": int(len(df_raw)), "churn_rate": round(float(df["Churn"].mean()), 4)},
        "eda": eda_out,
        "cv_comparison": cv_records,
        "tuned": tuned,
        "validation": {k: {kk: (float(round(vv, 4)) if isinstance(vv, (float, np.floating)) else vv)
                           for kk, vv in v.items()} for k, v in val_results.items()},
        "champion": champion_name,
        "calibration": {"brier_before": round(brier_before, 4),
                        "brier_after": round(brier_after, 4)},
        "threshold_selection": {
            "threshold": threshold,
            "offer_cost_usd": evaluation.OFFER_COST,
            "save_rate": evaluation.SAVE_RATE,
            "ltv_months": evaluation.LTV_MONTHS,
            "validation_expected_value_usd": round(float(best["expected_value"]), 2),
            "validation_n_contacted": int(best["n_contacted"]),
        },
        "test": {k: (float(round(v, 4)) if isinstance(v, (float, np.floating)) else v)
                 for k, v in test_metrics.items()},
        "test_expected_value_usd": round(float(test_ev["expected_value"]), 2),
        "shap_top_features": shap_top,
    }
    with open(os.path.join(RESULTS, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Done. Metrics -> results/metrics.json")


if __name__ == "__main__":
    main()
