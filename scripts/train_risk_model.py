"""Stage 3: train the risk model and compare it against simpler alternatives.

Every alternative is scored on the same held-out test patients:

* what clinics rely on today (the last monthly potassium lab)
* each data stream alone (EHR only, wearables only, smartwatch ECG only)
* the mechanistic twin with no wearable data (physics only)
* the full Antaral twin, and the twin plus the LightGBM risk layer

Outputs models/risk_model.joblib, reports/model_metrics.json, reports/ablation.csv,
reports/roc_curves.json, reports/feature_importance.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from antaral.features import ALL_FEATURES, FEATURE_LABELS, FEATURE_SETS  # noqa: E402
from antaral.risk_model import RiskModel, evaluate  # noqa: E402


def within_patient_auroc(df: pd.DataFrame, score: np.ndarray) -> float:
    vals = []
    for _, g in df.assign(s=score).groupby("patient_idx"):
        if g["y"].nunique() == 2 and g["s"].notna().all():
            vals.append(roc_auc_score(g["y"], g["s"]))
    return float(np.mean(vals)) if vals else float("nan")


def bootstrap_auroc(df: pd.DataFrame, score: np.ndarray, n: int = 300, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    pats = df["patient_idx"].unique()
    groups = {p: np.flatnonzero(df["patient_idx"].to_numpy() == p) for p in pats}
    y = df["y"].to_numpy()
    out = []
    for _ in range(n):
        idx = np.concatenate([groups[p] for p in rng.choice(pats, len(pats))])
        if y[idx].min() != y[idx].max():
            out.append(roc_auc_score(y[idx], score[idx]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> None:
    train = pd.read_csv(ROOT / "data" / "train_features.csv.gz")
    test = pd.read_csv(ROOT / "data" / "test_features.csv.gz")
    y = test["y"].to_numpy()
    rows, curves, scores = [], {}, {}

    def fill(s: pd.Series) -> np.ndarray:        # missing value -> lowest score
        return s.fillna(s.min() - 1).to_numpy(float)

    scores["Last monthly potassium lab (today's practice)"] = fill(test["last_lab_k"])
    scores["Smartwatch ECG potassium alone"] = fill(test["ecg_k_last"])
    scores["Physics-only twin (EHR, no wearables)"] = test["physics_only_p_exceed"].to_numpy()
    scores["Antaral twin forecast (EHR + wearables, Kalman fusion)"] = test["tw_p_exceed"].to_numpy()

    models = {}
    for name, feats in FEATURE_SETS.items():
        m = RiskModel(feats, seed=0).fit(train)
        models[name] = m
        label = name if not name.startswith("Antaral") else "Antaral twin + risk model (final)"
        scores[f"LightGBM: {label}" if not name.startswith("Antaral") else label] = m.predict_proba(test)

    for name, s in scores.items():
        r = evaluate(y, s)
        is_prob = name.startswith(("Physics", "Antaral", "LightGBM"))
        if not is_prob:
            r["brier"] = float("nan")
        r["within_patient_auroc"] = within_patient_auroc(test, s)
        rows.append({"model": name, **r})
        fpr, tpr, _ = roc_curve(y, s)
        keep = np.unique(np.linspace(0, len(fpr) - 1, 120).astype(int))
        curves[name] = {"fpr": fpr[keep].round(4).tolist(), "tpr": tpr[keep].round(4).tolist()}

    ab = pd.DataFrame(rows)
    final_name = "Antaral twin + risk model (final)"
    final = scores[final_name]
    lo, hi = bootstrap_auroc(test, final)
    lab_lo, lab_hi = bootstrap_auroc(test, scores["Last monthly potassium lab (today's practice)"])

    full = models["Antaral (twin + EHR + wearables)"]
    shap = full.explain(test)
    imp = shap.abs().mean().sort_values(ascending=False)
    imp_df = pd.DataFrame({"feature": imp.index, "label": [FEATURE_LABELS.get(f, f) for f in imp.index],
                           "mean_abs_shap": imp.values})

    # severe hyperkalaemia (>= 6.5) is a harder, rarer target: how well does the same score rank it?
    sev = evaluate(test["y_severe"].to_numpy(), final)

    ROOT.joinpath("reports").mkdir(exist_ok=True)
    joblib.dump(full, ROOT / "models" / "risk_model.joblib")
    ab.to_csv(ROOT / "reports" / "ablation.csv", index=False)
    imp_df.to_csv(ROOT / "reports" / "feature_importance.csv", index=False)
    (ROOT / "reports" / "roc_curves.json").write_text(json.dumps(curves))
    metrics = {
        "train_patients": int(train["patient_idx"].nunique()), "train_decisions": len(train),
        "test_patients": int(test["patient_idx"].nunique()), "test_decisions": len(test),
        "event_rate_test": float(y.mean()),
        "final": {**evaluate(y, final), "auroc_95ci": [lo, hi],
                  "within_patient_auroc": within_patient_auroc(test, final)},
        "last_lab_auroc_95ci": [lab_lo, lab_hi],
        "final_on_severe_k_ge_6_5": sev,
        "features_used": ALL_FEATURES,
    }
    (ROOT / "reports" / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.set_option("display.width", 200)
    print(ab[["model", "auroc", "auprc", "brier", "sensitivity_at_90_specificity",
              "within_patient_auroc"]].round(3).to_string(index=False))
    print(json.dumps({k: v for k, v in metrics.items() if k != "features_used"}, indent=2))
    print(imp_df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
