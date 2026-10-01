"""The patient twin: a potassium state estimator with what-if forecasting.

For every patient the twin keeps a belief over two hidden quantities::

    K  - current serum potassium (mEq/L)
    b  - personal drift: how much faster or slower this patient's potassium rises than
         the textbook model predicts from their EHR (mEq/L per hour)

It runs a Kalman filter hour by hour:

* **Predict** with a mechanistic model personalised from the EHR (body weight,
  diabetes, beta-blocker, ACE-inhibitor/ARB, residual urine output, bicarbonate),
  known dialysis sessions, and any food the patient logged.
* **Correct** with observations: pre-dialysis potassium labs (precise, rare) and
  smartwatch-ECG potassium estimates (noisy, frequent).

The filter learns ``b`` over days, which is how the twin adapts to a patient who eats
more potassium than they log, or whose cells buffer potassium poorly.

Forecasts are Monte Carlo: sample (K, b) from the belief, roll the model forward with
process noise under a scenario (dialysis timing, extra food, a potassium binder), and
count how often potassium crosses the danger threshold before the next session.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .ecg import DELTA_FEATURES, RAW_FEATURES, delta_features

# population physiology the twin assumes (it never sees an individual's hidden values)
POP_V_FRAC = 1.15
POP_GI = 0.33
POP_KD = 0.22
POP_UNLOGGED_ITEMS_MEQ = 2.5          # expected unlogged potassium-rich food per day
Q_K = 0.03                            # process noise on K per hour
Q_B = 0.0008                          # random walk on drift per hour
R_LAB = 0.15 ** 2


class EcgPersonalizer:
    """Tracks each patient's own 'normal' beat, refreshed right after every dialysis.

    Right after a session potassium is reliably near or below normal, so the beat
    recorded then is a natural personal reference. Later beats are compared with it.
    """

    def __init__(self, n: int, alpha: float = 0.3):
        self.base = pd.DataFrame(np.nan, index=range(n), columns=RAW_FEATURES)
        self.count = np.zeros(n, int)
        self.alpha = alpha

    def deltas(self, idx: np.ndarray, raw: pd.DataFrame) -> pd.DataFrame:
        b = self.base.loc[idx].reset_index(drop=True)
        out = delta_features(raw.reset_index(drop=True), b)
        out["has_baseline"] = (self.count[idx] > 0).astype(int)
        return out

    def update(self, idx: np.ndarray, raw: pd.DataFrame, post_mask: np.ndarray) -> None:
        for j in np.flatnonzero(post_mask):
            i = idx[j]
            row = raw.iloc[j][RAW_FEATURES].to_numpy(float)
            if self.count[i] == 0:
                self.base.loc[i] = row
            else:
                self.base.loc[i] = (1 - self.alpha) * self.base.loc[i].to_numpy(float) + self.alpha * row
            self.count[i] += 1


def ecg_model_input(raw: pd.DataFrame, deltas: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([raw[RAW_FEATURES].reset_index(drop=True),
                      deltas[DELTA_FEATURES + ["has_baseline"]].reset_index(drop=True)], axis=1)


class TwinBank:
    """Kalman-filter twins for a whole unit, vectorised over patients."""

    def __init__(self, ehr: pd.DataFrame, ecg_estimator=None, use_ecg: bool = True,
                 record: bool = False, n_hours: int = 0, seed: int = 0):
        self.ehr = ehr.reset_index(drop=True)
        e = self.ehr
        n = len(e)
        self.N = n
        diab = e["diabetes"].to_numpy(bool)
        bb = e["beta_blocker"].to_numpy(bool)
        ace = e["ace_arb"].to_numpy(bool)
        self.V = POP_V_FRAC * e["dry_weight_kg"].to_numpy(float) * np.where(diab, 0.88, 1) * np.where(bb, 0.92, 1)
        self.gi = POP_GI * np.where(ace, 0.85, 1.0)
        renal_h = e["urine_ml_day"].to_numpy(float) / 1000 * 25 * np.where(ace, 0.8, 1.0) / 24
        acid_h = np.maximum(0, 20 - e["bicarbonate"].to_numpy(float)) * 1.2 / 24
        diet = 39 * 1.04 - 0.10 * (e["age"].to_numpy(float) - 52) + POP_UNLOGGED_ITEMS_MEQ
        prof = np.zeros((n, 24))
        for hr, frac in C.MEAL_FRACTIONS.items():
            prof[:, hr] = diet * frac * (1 - self.gi)
        self.u = (prof - renal_h[:, None] + acid_h[:, None]) / self.V[:, None]   # mEq/L per hour of day

        self.x = np.column_stack([np.full(n, 4.5), np.zeros(n)])
        self.P = np.tile(np.diag([0.8 ** 2, 0.008 ** 2]), (n, 1, 1))
        self.pool = np.zeros(n)
        self.pre_k = np.zeros(n)
        self.last_end = np.full(n, -10_000)
        self.hour = 0
        self.ecg_estimator = ecg_estimator
        self.use_ecg = use_ecg and ecg_estimator is not None
        self.personal = EcgPersonalizer(n)
        self.rng = np.random.default_rng(seed)
        self.record = record
        if record:
            self.hist_mean = np.full((n, n_hours), np.nan, np.float32)
            self.hist_sd = np.full((n, n_hours), np.nan, np.float32)
        self.last_ecg = {}                      # idx -> dict(hour, k_hat, sd, deltas)
        self.ecg_points = []                    # (idx, hour, k_hat, sd) for charts

    # ---------------------------------------------------------------------------
    def _update(self, idx: int, z: float, r: float) -> None:
        P = self.P[idx]
        s = P[0, 0] + r
        k = P[:, 0] / s
        innov = z - self.x[idx, 0]
        self.x[idx] += k * innov
        self.P[idx] = P - np.outer(k, P[0, :])

    def step(self, ev) -> None:
        h = ev.hour
        hr = h % 24
        on = ev.in_session
        for idx, _dur, _w in ev.session_starts:
            self.pre_k[idx] = self.x[idx, 0]

        # --- predict: dialysis ---
        a = np.exp(-POP_KD)
        if on.any():
            self.x[on, 0] = C.BATH_K + (self.x[on, 0] - C.BATH_K) * a
            F = np.array([[a, 0.0], [0.0, 1.0]])
            self.P[on] = F @ self.P[on] @ F.T + np.diag([0.05 ** 2, Q_B ** 2])
        # --- predict: between sessions ---
        off = ~on
        logged = np.zeros(self.N)
        for idx, _item, meq in ev.diet_logs:
            logged[idx] += meq * (1 - self.gi[idx]) / self.V[idx]
        transfer = 0.35 * self.pool[off]
        self.pool[off] -= transfer
        self.x[off, 0] += self.u[off, hr] + self.x[off, 1] + logged[off] + transfer
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.P[off] = F @ self.P[off] @ F.T + np.diag([Q_K ** 2, Q_B ** 2])

        for idx in ev.session_ends:
            self.pool[idx] = 0.15 * (self.pre_k[idx] - self.x[idx, 0])
            self.last_end[idx] = h

        # --- correct: labs ---
        for idx, val in ev.labs:
            self._update(idx, val, R_LAB)

        # --- correct: smartwatch ECG ---
        if ev.ecg_idx is not None and len(ev.ecg_idx):
            idx = ev.ecg_idx
            d = self.personal.deltas(idx, ev.ecg_raw)
            if self.use_ecg:
                X = ecg_model_input(ev.ecg_raw, d)
                k_hat, sd = self.ecg_estimator.predict(X)
                for j, i in enumerate(idx):
                    self._update(int(i), float(k_hat[j]), float(sd[j]) ** 2)
                    prev = self.last_ecg.get(int(i))
                    self.last_ecg[int(i)] = {"hour": h, "k_hat": float(k_hat[j]), "sd": float(sd[j]),
                                             "k_prev": prev["k_hat"] if prev else np.nan,
                                             **{c: float(d.iloc[j][c]) for c in DELTA_FEATURES + ["has_baseline"]}}
                    self.ecg_points.append((int(i), h, float(k_hat[j]), float(sd[j])))
            post = (h - self.last_end[idx]) <= 3
            self.personal.update(idx, ev.ecg_raw, post)

        self.x[:, 0] = np.clip(self.x[:, 0], 2.0, 10.0)
        if self.record and h < self.hist_mean.shape[1]:
            self.hist_mean[:, h] = self.x[:, 0]
            self.hist_sd[:, h] = np.sqrt(self.P[:, 0, 0])
        self.hour = h + 1

    # ---------------------------------------------------------------------------
    def forecast(self, idx: np.ndarray, horizon: int, sessions: list | None = None,
                 extra_meq: np.ndarray | None = None, binder_meq_day: float | np.ndarray = 0.0,
                 drift_scale: float | np.ndarray = 1.0, n_samples: int = 300,
                 rng: np.random.Generator | None = None) -> np.ndarray:
        """Monte Carlo potassium paths starting at the current hour.

        idx       patients to forecast (array of ints)
        horizon   hours ahead
        sessions  per patient, a list of (offset_hour, duration) dialysis sessions
        extra_meq (n, horizon) additional potassium eaten at each future hour (what-if)
        binder_meq_day  potassium removed by a binder per day (what-if)
        drift_scale     multiplier on the learned personal drift (what-if)
        Returns samples with shape (n, n_samples, horizon).
        """
        rng = rng or self.rng
        idx = np.asarray(idx)
        n = idx.size
        mean = self.x[idx]
        cov = self.P[idx] + np.eye(2)[None] * 1e-10
        L = np.linalg.cholesky(cov)
        z = rng.standard_normal((n, n_samples, 2))
        s = mean[:, None, :] + np.einsum("nij,nsj->nsi", L, z)
        K = s[..., 0].copy()
        b = s[..., 1].copy() * np.broadcast_to(np.asarray(drift_scale, float), (n,))[:, None]
        pool = np.repeat(self.pool[idx][:, None], n_samples, axis=1)
        on = np.zeros((n, horizon), bool)
        if sessions is not None:
            for j, sess in enumerate(sessions):
                for off, dur in sess:
                    on[j, max(off, 0):max(off + dur, 0)] = True
        a = np.exp(-POP_KD)
        gi = self.gi[idx][:, None]
        V = self.V[idx][:, None]
        binder = np.broadcast_to(np.asarray(binder_meq_day, float), (n,))[:, None] / 24 / V
        out = np.empty((n, n_samples, horizon), np.float32)
        pre = K.copy()
        for t in range(horizon):
            hr = (self.hour + t) % 24
            o = on[:, t][:, None]
            starting = o & ~(on[:, t - 1][:, None] if t > 0 else np.zeros_like(o))
            pre = np.where(starting, K, pre)
            k_dial = C.BATH_K + (K - C.BATH_K) * a
            trans = 0.35 * pool
            add = self.u[idx, hr][:, None] + b + trans - binder
            if extra_meq is not None:
                add = add + extra_meq[:, t][:, None] * (1 - gi) / V
            k_off = K + add + rng.normal(0, Q_K, K.shape)
            K = np.where(o, k_dial, k_off)
            pool = np.where(o, 0.0, pool - trans)
            ending = o & ~(on[:, t + 1][:, None] if t + 1 < horizon else np.zeros_like(o))
            pool = np.where(ending, 0.15 * (pre - K), pool)
            b = b + rng.normal(0, Q_B, b.shape)
            K = np.clip(K, 2.0, 10.0)
            out[:, :, t] = K
        return out

    def window_risk(self, idx: np.ndarray, window_end: np.ndarray, n_samples: int = 300,
                    threshold: float = C.K_THRESHOLD, **kw) -> pd.DataFrame:
        """Risk of crossing ``threshold`` before each patient's next session.

        window_end: hours from now until the next session starts (per patient).
        """
        idx = np.asarray(idx)
        window_end = np.clip(np.asarray(window_end, int), 1, C.MAX_WINDOW_H)
        horizon = int(window_end.max())
        paths = self.forecast(idx, horizon, n_samples=n_samples, **kw)
        mask = np.arange(horizon)[None, None, :] < window_end[:, None, None]
        peak = np.where(mask, paths, -np.inf).max(axis=2)
        end = paths[np.arange(idx.size), :, window_end - 1]
        return pd.DataFrame({
            "tw_p_exceed": (peak >= threshold).mean(axis=1),
            "tw_p_severe": (peak >= C.K_SEVERE).mean(axis=1),
            "tw_peak_mean": peak.mean(axis=1),
            "tw_k_end_mean": end.mean(axis=1),
            "tw_k_end_sd": end.std(axis=1),
        })
