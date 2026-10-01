"""Observable patient record and the feature vector used by the risk model.

``PublicRecord`` is the unit's view of each patient: sessions, labs, smart-scale
weights, daily wearable summaries and the food diary. It never touches ground truth.
``build_features`` fuses that record with the EHR and the twin's forecast into one row
per patient at a decision time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

EHR_FEATURES = ["age", "dry_weight_kg", "diabetes", "ace_arb", "beta_blocker", "urine_ml_day",
                "bicarbonate", "vintage_months", "hd_freq"]
TWIN_FEATURES = ["tw_k", "tw_k_sd", "tw_drift_day", "tw_drift_sd_day", "tw_p_exceed",
                 "tw_p_severe", "tw_peak_mean", "tw_k_end_mean", "tw_k_end_sd"]
TIMING_FEATURES = ["hours_to_next", "hours_since_last"]
ECG_FEATURES = ["ecg_k_last", "ecg_age_h", "n_ecg_48h", "ecg_k_change", "d_log_t_r_last",
                "d_t_fwhm_last", "d_qrs_ms_last", "d_pr_ms_last", "has_ecg_baseline"]
WEIGHT_FEATURES = ["fluid_obs_kg", "fluid_obs_pct", "weight_rate_kg_day"]
VITAL_FEATURES = ["sbp_last", "rest_hr_delta", "steps_ratio", "rmssd_ratio", "sleep_delta"]
DIET_FEATURES = ["logged_meq_24h", "logged_meq_since_session", "coconut_since_session"]
LAB_FEATURES = ["last_lab_k", "days_since_lab"]

ALL_FEATURES = (EHR_FEATURES + TWIN_FEATURES + TIMING_FEATURES + ECG_FEATURES + WEIGHT_FEATURES
                + VITAL_FEATURES + DIET_FEATURES + LAB_FEATURES)

FEATURE_SETS = {
    "EHR only": EHR_FEATURES + TIMING_FEATURES + LAB_FEATURES,
    "Wearables only": TIMING_FEATURES + ECG_FEATURES + WEIGHT_FEATURES + VITAL_FEATURES + DIET_FEATURES,
    "EHR + wearables (no twin)": EHR_FEATURES + TIMING_FEATURES + LAB_FEATURES + ECG_FEATURES
                                 + WEIGHT_FEATURES + VITAL_FEATURES + DIET_FEATURES,
    "Antaral (twin + EHR + wearables)": ALL_FEATURES,
}

FEATURE_LABELS = {
    "tw_p_exceed": "Twin forecast: chance of K ≥ 6 before session",
    "tw_peak_mean": "Twin forecast: expected peak potassium",
    "tw_k_end_mean": "Twin forecast: potassium at session start",
    "tw_k": "Twin estimate: potassium now",
    "tw_k_sd": "Twin uncertainty now",
    "tw_drift_day": "Learned personal potassium drift",
    "tw_p_severe": "Twin forecast: chance of K ≥ 6.5",
    "hours_to_next": "Hours until next session",
    "hours_since_last": "Hours since last session",
    "ecg_k_last": "Latest smartwatch-ECG potassium",
    "d_log_t_r_last": "T-wave height vs personal baseline",
    "d_t_fwhm_last": "T-wave width vs personal baseline",
    "d_qrs_ms_last": "QRS width vs personal baseline",
    "d_pr_ms_last": "PR interval vs personal baseline",
    "fluid_obs_kg": "Weight above dry weight",
    "fluid_obs_pct": "Weight gain (% of dry weight)",
    "logged_meq_since_session": "Logged potassium-rich food since session",
    "coconut_since_session": "Coconut water since session",
    "logged_meq_24h": "Logged potassium-rich food, 24 h",
    "steps_ratio": "Steps vs personal normal",
    "rest_hr_delta": "Resting heart rate vs personal normal",
    "last_lab_k": "Last pre-dialysis potassium lab",
    "urine_ml_day": "Residual urine output",
    "ace_arb": "On ACE inhibitor / ARB",
    "diabetes": "Diabetes",
    "hd_freq": "Sessions per week",
}


class PublicRecord:
    def __init__(self, n: int, n_days: int):
        self.n = n
        self.sessions: list[list] = [[] for _ in range(n)]       # (start_hour, duration)
        self.labs: list[list] = [[] for _ in range(n)]           # (hour, value)
        self.weights: list[list] = [[] for _ in range(n)]        # (hour, kg)
        self.diet: list[list] = [[] for _ in range(n)]           # (hour, item, meq)
        self.ecg_hours: list[list] = [[] for _ in range(n)]
        cols = ["sbp", "rest_hr", "steps", "rmssd", "sleep_h"]
        self.vitals = {c: np.full((n, n_days), np.nan) for c in cols}

    def ingest(self, ev) -> None:
        for idx, dur, _w in ev.session_starts:
            self.sessions[idx].append((ev.hour, dur))
        for idx, val in ev.labs:
            self.labs[idx].append((ev.hour, val))
        for idx, kg in ev.weights:
            self.weights[idx].append((ev.hour, kg))
        for idx, item, meq in ev.diet_logs:
            self.diet[idx].append((ev.hour, item, meq))
        if ev.ecg_idx is not None:
            for i in ev.ecg_idx:
                self.ecg_hours[int(i)].append(ev.hour)
        if ev.vitals is not None:
            d = ev.hour // 24
            for c in self.vitals:
                self.vitals[c][:, d] = ev.vitals[c].to_numpy()

    def last_session_end(self, i: int) -> int:
        if not self.sessions[i]:
            return -10_000
        s, dur = self.sessions[i][-1]
        return s + dur


def _personal_ratio(series: np.ndarray, d: int, log: bool = True) -> float:
    hist = series[max(0, d - 14):d]
    hist = hist[~np.isnan(hist)]
    cur = series[d]
    if hist.size < 4 or np.isnan(cur):
        return np.nan
    base = np.median(hist)
    return float(cur / base) if log else float(cur - base)


def build_features(t: int, idx: np.ndarray, next_start: np.ndarray, ehr: pd.DataFrame,
                   twin, record: PublicRecord, n_samples: int = 300, twin_kw: dict | None = None,
                   twin_noecg=None) -> pd.DataFrame:
    """Feature rows at hour ``t`` (state after hour t) for patients ``idx``.

    next_start: absolute hour of each patient's next dialysis session start.
    The risk window runs from hour t+1 to the hour before ``next_start``.
    """
    idx = np.asarray(idx)
    next_start = np.asarray(next_start)
    window = np.clip(next_start - t - 1, 1, C.MAX_WINDOW_H)
    tw = twin.window_risk(idx, window, n_samples=n_samples, **(twin_kw or {}))
    rows = []
    d = t // 24
    for j, i in enumerate(idx):
        i = int(i)
        last_end = record.last_session_end(i)
        r = {
            "patient_idx": i, "t": t,
            "tw_k": twin.x[i, 0], "tw_k_sd": np.sqrt(twin.P[i, 0, 0]),
            "tw_drift_day": twin.x[i, 1] * 24, "tw_drift_sd_day": np.sqrt(twin.P[i, 1, 1]) * 24,
            "hours_to_next": float(next_start[j] - t), "hours_since_last": float(t - last_end),
        }
        e = twin.last_ecg.get(i)
        if e is not None:
            r.update(ecg_k_last=e["k_hat"], ecg_age_h=t - e["hour"], d_log_t_r_last=e["d_log_t_r"],
                     d_t_fwhm_last=e["d_t_fwhm"], d_qrs_ms_last=e["d_qrs_ms"], d_pr_ms_last=e["d_pr_ms"],
                     has_ecg_baseline=e["has_baseline"])
        hours = record.ecg_hours[i]
        r["n_ecg_48h"] = sum(1 for h in hours if h > t - 48)
        r["ecg_k_change"] = e["k_hat"] - e["k_prev"] if e is not None else np.nan

        dry = ehr.at[i, "dry_weight_kg"]
        w = [x for x in record.weights[i] if x[0] > t - 96]
        if w:
            r["fluid_obs_kg"] = w[-1][1] - dry
            r["fluid_obs_pct"] = 100 * (w[-1][1] - dry) / dry
            since = [x for x in w if x[0] > last_end]
            if len(since) >= 2:
                r["weight_rate_kg_day"] = (since[-1][1] - since[0][1]) / max((since[-1][0] - since[0][0]) / 24, 0.5)
        v = record.vitals
        r["sbp_last"] = v["sbp"][i, d]
        r["rest_hr_delta"] = _personal_ratio(v["rest_hr"][i], d, log=False)
        r["steps_ratio"] = _personal_ratio(v["steps"][i], d)
        r["rmssd_ratio"] = _personal_ratio(v["rmssd"][i], d)
        r["sleep_delta"] = _personal_ratio(v["sleep_h"][i], d, log=False)
        diet = record.diet[i]
        r["logged_meq_24h"] = sum(m for h, _, m in diet if h > t - 24)
        r["logged_meq_since_session"] = sum(m for h, _, m in diet if h > last_end)
        r["coconut_since_session"] = sum(1 for h, it, _ in diet if h > last_end and "coconut" in it)
        if record.labs[i]:
            lh, lv = record.labs[i][-1]
            r["last_lab_k"], r["days_since_lab"] = lv, (t - lh) / 24
        rows.append(r)
    df = pd.DataFrame(rows)
    df = pd.concat([df, tw.reset_index(drop=True)], axis=1)
    if twin_noecg is not None:
        tw2 = twin_noecg.window_risk(idx, window, n_samples=n_samples)
        df["physics_only_p_exceed"] = tw2["tw_p_exceed"].to_numpy()
    for c in EHR_FEATURES:
        df[c] = ehr.loc[idx, c].to_numpy().astype(float)
    for c in ALL_FEATURES:
        if c not in df:
            df[c] = np.nan
    return df
