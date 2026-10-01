"""Potassium-from-ECG estimator (the 'bloodless potassium' sensor of the twin).

A gradient-boosted regressor maps smartwatch median-beat features (raw and relative to
the patient's own post-dialysis beat) to serum potassium. Its validation error becomes
the measurement noise the Kalman filter assigns to each ECG reading, so the twin
automatically trusts an ECG less than a lab, and trusts an un-personalised ECG less
than a personalised one.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd


class EcgPotassiumEstimator:
    def __init__(self, params: dict | None = None):
        self.params = params or dict(n_estimators=400, learning_rate=0.04, num_leaves=31,
                                     min_child_samples=40, subsample=0.8, subsample_freq=1,
                                     colsample_bytree=0.8, verbose=-1)
        self.model = None
        self.sd_by_baseline = {0: 0.8, 1: 0.6}
        self.columns: list[str] = []
        self.metrics: dict = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, seed: int = 0) -> "EcgPotassiumEstimator":
        rng = np.random.default_rng(seed)
        ug = np.unique(groups)
        val_g = set(rng.choice(ug, size=max(1, len(ug) // 5), replace=False))
        val = np.isin(groups, list(val_g))
        self.columns = list(X.columns)
        m = lgb.LGBMRegressor(random_state=seed, **self.params)
        m.fit(X[~val], y[~val])
        pred = m.predict(X[val])
        res = y[val] - pred
        hb = X.loc[val, "has_baseline"].to_numpy()
        for flag in (0, 1):
            sel = hb == flag
            if sel.sum() > 20:
                self.sd_by_baseline[flag] = float(np.sqrt(np.mean(res[sel] ** 2)))
        hyper = y[val] >= 5.5
        self.metrics = {
            "val_patients": len(val_g),
            "val_recordings": int(val.sum()),
            "mae": float(np.mean(np.abs(res))),
            "rmse": float(np.sqrt(np.mean(res ** 2))),
            "rmse_personalised": self.sd_by_baseline[1],
            "rmse_unpersonalised": self.sd_by_baseline[0],
            "corr": float(np.corrcoef(pred, y[val])[0, 1]),
        }
        try:
            from sklearn.metrics import roc_auc_score
            self.metrics["auroc_k_ge_5_5"] = float(roc_auc_score(hyper, pred))
        except ValueError:
            pass
        # refit on everything for deployment
        self.model = lgb.LGBMRegressor(random_state=seed, **self.params).fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        k = self.model.predict(X[self.columns])
        sd = np.where(X["has_baseline"].to_numpy() > 0, self.sd_by_baseline[1], self.sd_by_baseline[0])
        return k, sd
