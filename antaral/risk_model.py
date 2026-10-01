"""Risk layer: LightGBM on top of the twin, with per-patient explanations.

The twin already produces a probability of crossing K >= 6.0 before the next session.
The risk model learns where that mechanistic forecast is systematically wrong, using
context the physics does not model (illness signs in heart rate and steps, fluid
gain, ECG trend, logging habits), and returns a calibrated probability.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


class RiskModel:
    def __init__(self, features: list[str], seed: int = 0):
        self.features = list(features)
        self.seed = seed
        self.model: lgb.LGBMClassifier | None = None
        self.calibrator: IsotonicRegression | None = None

    def fit(self, df: pd.DataFrame, cal_frac: float = 0.2) -> "RiskModel":
        rng = np.random.default_rng(self.seed)
        pats = df["patient_idx"].unique()
        cal_p = set(rng.choice(pats, size=max(1, int(len(pats) * cal_frac)), replace=False))
        cal = df["patient_idx"].isin(cal_p).to_numpy()
        self.model = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31,
                                        min_child_samples=50, subsample=0.8, subsample_freq=1,
                                        colsample_bytree=0.8, reg_lambda=1.0, random_state=self.seed,
                                        verbose=-1)
        self.model.fit(df.loc[~cal, self.features], df.loc[~cal, "y"],
                       eval_X=(df.loc[cal, self.features],), eval_y=(df.loc[cal, "y"],),
                       callbacks=[lgb.early_stopping(50, verbose=False)])
        raw = self.model.predict_proba(df.loc[cal, self.features])[:, 1]
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, df.loc[cal, "y"])
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        raw = self.model.predict_proba(df[self.features])[:, 1]
        return self.calibrator.predict(raw)

    def explain(self, df: pd.DataFrame) -> pd.DataFrame:
        """SHAP values (log-odds contributions) computed natively by LightGBM."""
        contrib = self.model.predict(df[self.features], pred_contrib=True)
        return pd.DataFrame(contrib[:, :-1], columns=self.features, index=df.index)


def evaluate(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y)
    p = np.asarray(p)
    fpr, tpr, thr = roc_curve(y, p)
    spec90 = tpr[np.searchsorted(fpr, 0.10, side="right") - 1]
    # alert threshold chosen to catch 85% of events
    j = np.searchsorted(tpr, 0.85)
    alert_thr = float(thr[min(j, len(thr) - 1)])
    flagged = p >= alert_thr
    return {
        "n": int(len(y)),
        "prevalence": float(y.mean()),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)) if (p.min() >= 0 and p.max() <= 1) else float("nan"),
        "sensitivity_at_90_specificity": float(spec90),
        "alert_threshold_85_sensitivity": alert_thr,
        "ppv_at_85_sensitivity": float(y[flagged].mean()) if flagged.any() else float("nan"),
        "alerts_per_100_patient_days": float(100 * flagged.mean()),
    }
