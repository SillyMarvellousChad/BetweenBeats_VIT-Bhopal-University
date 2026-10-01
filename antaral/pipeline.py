"""Runs a dialysis unit: simulator (truth) + twins + public record + a scheduling policy."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from .cohort import ehr_view
from .features import PublicRecord, build_features
from .physiology import UnitSimulator
from .scheduler import FixedPolicy
from .twin import TwinBank


@dataclass
class RiskContext:
    """What a policy may consult at decision time (no ground truth in here)."""
    t: int
    ehr: pd.DataFrame
    twin: TwinBank
    record: PublicRecord
    risk_model: object = None
    mode: str = "ml"                 # "ml" (twin + risk model) or "twin" (twin forecast only)
    n_samples: int = 300
    log: list = field(default_factory=list)

    def features(self, idx, next_start):
        return build_features(self.t, idx, next_start, self.ehr, self.twin, self.record, self.n_samples)

    def risk(self, idx, next_start) -> np.ndarray:
        """Probability of K >= 6 before ``next_start`` (absolute hour)."""
        f = self.features(idx, next_start)
        if self.mode == "twin" or self.risk_model is None:
            return f["tw_p_exceed"].to_numpy()
        return self.risk_model.predict_proba(f)

    def window_risk_simple(self, idx, next_start) -> np.ndarray:
        if len(idx) == 0:
            return np.zeros(0)
        w = np.clip(np.asarray(next_start) - self.t - 1, 1, C.MAX_WINDOW_H)
        return self.twin.window_risk(np.asarray(idx), w, n_samples=self.n_samples)["tw_p_exceed"].to_numpy()

    def horizon_burden(self, idx, s1_start, horizon_end) -> np.ndarray:
        """Expected hyperkalaemia burden until ``horizon_end`` if the next session starts at ``s1_start``.
        This is a counterfactual question, so it is answered by the twin's forecast."""
        idx = np.asarray(idx)
        base = self.twin.hour
        off = np.asarray(s1_start) - base
        end = np.clip(np.asarray(horizon_end) - base, off + C.SESSION_HOURS + 1, None)
        horizon = int(min(end.max(), 7 * 24))
        paths = self.twin.forecast(idx, horizon, sessions=[[(int(o), C.SESSION_HOURS)] for o in off],
                                   n_samples=self.n_samples)
        inside = np.arange(horizon)[None, None, :] < end[:, None, None]
        hrs = sum(((paths >= k) & inside).sum(axis=2) for k in (C.K_THRESHOLD, C.K_SEVERE, C.K_CRITICAL))
        return hrs.mean(axis=1)


@dataclass
class RunResult:
    sim: UnitSimulator
    twin: TwinBank
    record: PublicRecord
    ehr: pd.DataFrame
    policy: object
    features: pd.DataFrame | None = None


def run_unit(cohort: pd.DataFrame, n_days: int, seed: int, policy=None, ecg_estimator=None,
             risk_model=None, risk_mode: str = "ml", warmup_days: int = 14,
             record_features: bool = False, physics_only_twin: bool = False,
             record_twin: bool = False, keep_beats=None, scripted_items=None,
             n_samples: int = 300, stop_hour: int | None = None, progress: bool = False) -> RunResult:
    ehr = ehr_view(cohort)
    n = len(ehr)
    policy = policy or FixedPolicy(ehr)
    sim = UnitSimulator(cohort, n_days, seed, scripted_items=scripted_items, keep_beats=keep_beats)
    twin = TwinBank(ehr, ecg_estimator, record=record_twin, n_hours=n_days * 24, seed=seed + 1)
    twin_b = TwinBank(ehr, ecg_estimator, use_ecg=False, seed=seed + 2) if physics_only_twin else None
    rec = PublicRecord(n, n_days)
    sim.set_schedule(0, policy.baseline(0))
    rows = []
    last_hour = stop_hour if stop_hour is not None else n_days * 24
    for h in range(last_hour):
        ev = sim.step()
        rec.ingest(ev)
        twin.step(ev)
        if twin_b is not None:
            twin_b.step(ev)
        if h % 24 != C.DECISION_HOUR:
            continue
        day = h // 24
        ctx = RiskContext(h, ehr, twin, rec, risk_model, risk_mode, n_samples)
        if day + 1 < n_days:
            sim.set_schedule(day + 1, policy.plan(day + 1, ctx))
        if record_features and day >= warmup_days and day + 5 < n_days:
            nxt = np.array([policy.next_start(i, h) for i in range(n)])
            f = build_features(h, np.arange(n), nxt, ehr, twin, rec, n_samples, twin_noecg=twin_b)
            f["planned_next_start"] = nxt
            rows.append(f)
        if progress and day % 7 == 0:
            print(f"  day {day}/{n_days}", flush=True)
    res = RunResult(sim, twin, rec, ehr, policy)
    if rows:
        res.features = add_labels(pd.concat(rows, ignore_index=True), sim)
    return res


def add_labels(df: pd.DataFrame, sim: UnitSimulator) -> pd.DataFrame:
    """Ground-truth label: does true potassium reach the threshold before the next session?"""
    starts = sim.session_starts_by_patient()
    y, ysev, peak, actual = [], [], [], []
    for i, t in zip(df["patient_idx"].to_numpy(), df["t"].to_numpy()):
        s = next((x for x in starts.get(int(i), []) if x > t), t + C.MAX_WINDOW_H + 1)
        end = min(s, t + 1 + C.MAX_WINDOW_H, sim.H)
        seg = sim.true_K[int(i), t + 1:end]
        pk = float(seg.max()) if seg.size else float(sim.true_K[int(i), t])
        peak.append(pk)
        y.append(int(pk >= C.K_THRESHOLD))
        ysev.append(int(pk >= C.K_SEVERE))
        actual.append(s)
    df = df.copy()
    df["y"] = y
    df["y_severe"] = ysev
    df["true_peak_k"] = peak
    df["actual_next_start"] = actual
    df["true_k_now"] = sim.true_K[df["patient_idx"].to_numpy(), df["t"].to_numpy()]
    return df


def unit_outcomes(res: RunResult, from_day: int) -> dict:
    """Clinical outcomes over the evaluation period (uses ground truth: evaluation only)."""
    sim = res.sim
    h0 = from_day * 24
    K = sim.true_K[:, h0:]
    above = K >= C.K_THRESHOLD
    starts = np.diff(np.concatenate([np.zeros((K.shape[0], 1), bool), above], axis=1).astype(int), axis=1) == 1
    sess = [s for s in sim.session_log if s["day"] >= from_day]
    n_weeks = (sim.n_days - from_day) / 7
    pre = np.array([s["pre_k_true"] for s in sess])
    per_patient = np.bincount([s["idx"] for s in sess], minlength=sim.N)
    return {
        "hyperk_episodes": int(starts.sum()),
        "patient_hours_k_ge_6": int(above.sum()),
        "patient_hours_k_ge_6_5": int((K >= C.K_SEVERE).sum()),
        "patient_hours_k_ge_7": int((K >= C.K_CRITICAL).sum()),
        "danger_hours_weighted": int(sum((K >= k).sum() for k in (C.K_THRESHOLD, C.K_SEVERE, C.K_CRITICAL))),
        "pre_dialysis_k_ge_6": int((pre >= C.K_THRESHOLD).sum()),
        "sessions": len(sess),
        "sessions_per_patient_week": float(len(sess) / sim.N / n_weeks),
        "min_sessions_patient": int(per_patient.min()),
        "patients_with_any_k_ge_6": int(above.any(axis=1).sum()),
        "peak_k_mean": float(K.max(axis=1).mean()),
    }
