"""Ground-truth simulator of a dialysis unit, hour by hour.

The simulator is the "real world" of the experiment. It owns the hidden physiology
and the true serum potassium of every patient, and it emits only what a real unit
would observe:

* dialysis sessions (start, duration, pre-dialysis weight)
* pre-dialysis potassium labs on some sessions
* smartwatch ECG median beats (morning and night, when the patient remembers)
* smart-scale morning weight
* daily wearable summary: steps, resting HR, HRV (RMSSD), sleep, home BP
* food-diary entries for potassium-rich items (only those the patient bothers to log)

Potassium model (per patient, per hour, outside dialysis)::

    dK/dt = [ intake * (1 - gi_excretion) - renal_excretion + tissue_release ] / V_app
            + rebound

``V_app`` is the apparent potassium distribution volume: much larger than the
extracellular fluid because cells soak up most of a potassium load. It is calibrated so
that a typical patient gains ~0.6-0.8 mEq/L per day between sessions. During dialysis,
plasma potassium decays exponentially towards the dialysate bath; after the session a
fraction of the removed gradient rebounds from cells over a few hours.

Hidden disturbances the twin does not observe directly: unlogged potassium-rich food,
feast days, constipation (less colonic potassium excretion), intercurrent illness
(tissue breakdown releases potassium, also visible as higher resting HR / fewer steps),
and shortened sessions.

All random draws are made up front from the seed, so two runs with different
schedules face the same patients, the same meals and the same illnesses
(common random numbers). That makes scheduling policies directly comparable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from .ecg import beat_params, extract_features, synth_beats


@dataclass
class HourEvents:
    """Everything observable that happened during one hour."""
    hour: int
    session_starts: list = field(default_factory=list)      # (idx, duration_h, pre_weight)
    session_ends: list = field(default_factory=list)        # idx
    in_session: np.ndarray | None = None                    # bool mask
    labs: list = field(default_factory=list)                # (idx, value)
    ecg_idx: np.ndarray | None = None
    ecg_raw: pd.DataFrame | None = None
    weights: list = field(default_factory=list)             # (idx, kg)
    diet_logs: list = field(default_factory=list)           # (idx, item, meq)
    vitals: pd.DataFrame | None = None                      # one row per patient at sync hour


class UnitSimulator:
    def __init__(self, cohort: pd.DataFrame, n_days: int, seed: int,
                 scripted_items: list | None = None, keep_beats: set | None = None):
        self.c = cohort.reset_index(drop=True)
        self.N = len(self.c)
        self.n_days = n_days
        self.H = n_days * 24
        self.hour = 0
        rng = np.random.default_rng(seed)
        self.rng_obs = np.random.default_rng(seed + 7919)
        c = self.c
        N, D, H = self.N, n_days, self.H

        self.dry = c["dry_weight_kg"].to_numpy(float)
        self.V = (c["v_frac"] * c["dry_weight_kg"]).to_numpy(float)
        self.kd = c["kd"].to_numpy(float)
        ace = c["ace_arb"].to_numpy(bool)
        self.gi = c["gi_frac"].to_numpy(float) * np.where(ace, 0.85, 1.0)
        self.renal_h = c["urine_ml_day"].to_numpy(float) / 1000 * 25 * np.where(ace, 0.8, 1.0) / 24
        self.acid_release_h = np.maximum(0, 20 - c["bicarbonate"].to_numpy(float)) * 1.2 / 24

        # ---- daily disturbances --------------------------------------------------
        # habits drift from week to week (festivals, seasons, appetite): AR(1) in log space
        ar = np.zeros((N, D))
        ar[:, 0] = rng.normal(0, 0.27, N)
        for d_ in range(1, D):
            ar[:, d_] = 0.85 * ar[:, d_ - 1] + rng.normal(0, 0.14, N)
        day_mult = np.exp(ar + rng.normal(0, 0.12, (N, D)))
        feast = rng.random((N, D)) < c["feast_rate"].to_numpy()[:, None]
        day_mult *= np.where(feast, 1.4, 1.0)
        self.feast = feast

        def episodes(rate, length):
            on = np.zeros((N, D), bool)
            starts = rng.random((N, D)) < rate[:, None]
            for i, d in zip(*np.nonzero(starts)):
                on[i, d:d + length] = True
            return on

        self.constipation = episodes(c["constipation_rate"].to_numpy(), 3)
        self.illness = episodes(c["illness_rate"].to_numpy(), 3)

        intake = np.zeros((N, H))
        base = c["diet_k_meq"].to_numpy(float)[:, None] * day_mult
        for hr, frac in C.MEAL_FRACTIONS.items():
            intake[:, hr::24] += base * frac

        # potassium-rich items: some logged, most not
        names = list(C.K_ITEMS)
        self.item_log = []        # (idx, hour, item, meq, logged)
        n_items = rng.poisson(c["item_rate"].to_numpy()[:, None] * np.where(feast, 2.0, 1.0), (N, D))
        logp = c["log_prob"].to_numpy()
        for i, d in zip(*np.nonzero(n_items)):
            for _ in range(n_items[i, d]):
                item = names[rng.choice(len(names), p=C.K_ITEM_PROBS)]
                hr = int(rng.integers(9, 21))
                meq = C.K_ITEMS[item] * rng.lognormal(0, 0.2)
                self.item_log.append((int(i), d * 24 + hr, item, float(meq), bool(rng.random() < logp[i])))
        for (i, d, hr, item, logged) in (scripted_items or []):
            meq = C.K_ITEMS[item]
            self.item_log.append((int(i), d * 24 + hr, item, float(meq), bool(logged)))
        for (i, h, _, meq, _) in self.item_log:
            if h < H:
                intake[i, h] += meq
        self.intake = intake

        self.release_h = self.acid_release_h[:, None] + np.repeat(self.illness, 24, axis=1) * (16 / 24)
        gi_hourly = np.where(np.repeat(self.constipation, 24, axis=1), 0.10, self.gi[:, None])
        self.absorbed_frac = 1 - gi_hourly

        gain_day = c["fluid_gain_kg_day"].to_numpy(float)[:, None] * rng.lognormal(0, 0.25, (N, D))
        gain_day *= np.where(feast, 1.3, 1.0)
        fluid = np.zeros((N, H))
        for hr in range(7, 23):
            fluid[:, hr::24] = gain_day / 16
        self.fluid_in = fluid

        self.short_session = rng.random((N, D)) < 0.05
        self.ecg_taken = rng.random((N, H)) < c["ecg_adherence"].to_numpy()[:, None]
        self.post_ecg_taken = rng.random((N, D)) < 0.8
        self.scale_used = rng.random((N, D)) < 0.85
        self.vitals_noise = rng.normal(0, 1, (N, D, 6))

        # ---- state -----------------------------------------------------------------
        self.K = np.clip(rng.normal(4.6, 0.4, N), 3.6, 5.6)
        self.pool = np.zeros(N)
        self.excess = rng.uniform(0.5, 1.5, N)          # kg above dry weight
        self.session_left = np.zeros(N, int)
        self.session_pre_k = np.zeros(N)
        self.session_count = np.zeros(N, int)
        self.last_session_end = np.full(N, -10_000)
        self.schedule: dict[int, list] = {}             # day -> [(idx, shift)]
        self.session_log = []                           # dicts
        self.true_K = np.zeros((N, H), dtype=np.float32)
        self.true_excess = np.zeros((N, H), dtype=np.float32)
        self.in_session = np.zeros((N, H), bool)
        self.ecg_log = []                               # dicts incl. beat params
        self.keep_beats = keep_beats or set()

    # ------------------------------------------------------------------------------
    def set_schedule(self, day: int, entries: list) -> None:
        """entries: list of (patient_idx, shift_idx)."""
        self.schedule[day] = list(entries)

    def step(self) -> HourEvents:
        h = self.hour
        d, hr = divmod(h, 24)
        ev = HourEvents(hour=h)

        # sessions starting this hour
        for idx, shift in self.schedule.get(d, []):
            if C.SHIFT_STARTS[shift] == hr and self.session_left[idx] == 0:
                dur = 3 if self.short_session[idx, d] else C.SESSION_HOURS
                self.session_left[idx] = dur
                self.session_pre_k[idx] = self.K[idx]
                pre_w = self.dry[idx] + self.excess[idx]
                ev.session_starts.append((idx, dur, round(float(pre_w), 1)))
                self.session_count[idx] += 1
                lab = None
                if self.session_count[idx] % 6 == 1:      # roughly monthly labs
                    lab = float(np.round(self.K[idx] + self.rng_obs.normal(0, 0.12), 1))
                    ev.labs.append((idx, lab))
                self.session_log.append({"idx": idx, "hour": h, "day": d, "shift": shift,
                                         "duration": dur, "pre_k_true": float(self.K[idx]),
                                         "pre_weight": pre_w, "lab_k": lab})

        on = self.session_left > 0
        off = ~on
        ev.in_session = on.copy()

        # potassium
        a = np.exp(-self.kd[on])
        self.K[on] = C.BATH_K + (self.K[on] - C.BATH_K) * a
        net = (self.intake[off, h] * self.absorbed_frac[off, h] - self.renal_h[off]
               + self.release_h[off, h])
        transfer = 0.35 * self.pool[off]
        self.pool[off] -= transfer
        self.K[off] += net / self.V[off] + transfer
        self.K = np.clip(self.K, 2.5, 9.5)

        # fluid
        uf_max = np.maximum(0.016 * self.dry[on], 1.1)   # ultrafiltration ceiling, kg/h
        target = np.maximum(self.excess[on] - 0.2, 0)
        self.excess[on] -= np.minimum(target / np.maximum(self.session_left[on], 1), uf_max)
        self.excess[off] += self.fluid_in[off, h]

        # session bookkeeping
        self.session_left[on] -= 1
        ended = np.flatnonzero(on & (self.session_left == 0))
        for idx in ended:
            self.pool[idx] = 0.15 * (self.session_pre_k[idx] - self.K[idx])
            self.last_session_end[idx] = h
            ev.session_ends.append(int(idx))

        self.true_K[:, h] = self.K
        self.true_excess[:, h] = self.excess
        self.in_session[:, h] = on

        # ---- observations ---------------------------------------------------------
        rec = np.zeros(self.N, bool)
        if hr in C.ECG_HOURS:
            rec |= self.ecg_taken[:, h] & off
        post = (h - self.last_session_end == 2) & self.post_ecg_taken[:, d]
        rec |= post
        if rec.any():
            idx = np.flatnonzero(rec)
            c = self.c.iloc[idx]
            p = beat_params(self.K[idx], c["ecg_r_amp0"].to_numpy(), c["ecg_t_amp0"].to_numpy(),
                            c["ecg_t_sigma0"].to_numpy(), c["ecg_gamma"].to_numpy(), self.rng_obs)
            ev.ecg_idx = idx
            ev.ecg_raw = extract_features(synth_beats(p))
            for j, i in enumerate(idx):
                entry = {"idx": int(i), "hour": h, "k_true": float(self.K[i]),
                         "post_dialysis": bool(post[i])}
                if int(i) in self.keep_beats:
                    entry["params"] = {k: (v[j].item() if hasattr(v[j], "item") else v[j]) for k, v in p.items()}
                self.ecg_log.append(entry)

        if hr == C.WEIGHT_HOUR:
            for i in np.flatnonzero(self.scale_used[:, d] & off):
                w = self.dry[i] + self.excess[i] + self.rng_obs.normal(0, 0.15)
                ev.weights.append((int(i), round(float(w), 1)))

        for (i, ih, item, meq, logged) in self._items_at(h):
            if logged:
                ev.diet_logs.append((i, item, C.K_ITEMS[item]))

        if hr == C.VITALS_SYNC_HOUR:
            ev.vitals = self._vitals(d)

        self.hour += 1
        return ev

    def _items_at(self, h):
        if not hasattr(self, "_items_by_hour"):
            self._items_by_hour = {}
            for it in self.item_log:
                self._items_by_hour.setdefault(it[1], []).append(it)
        return self._items_by_hour.get(h, [])

    def _vitals(self, d: int) -> pd.DataFrame:
        c = self.c
        hs = slice(d * 24, d * 24 + 21)
        ex = self.true_excess[:, hs].mean(axis=1)
        kmax = self.true_K[:, hs].max(axis=1)
        ill = self.illness[:, d].astype(float)
        dialysed = self.in_session[:, hs].any(axis=1).astype(float)
        z = self.vitals_noise[:, d, :]
        hyperk = np.maximum(0, kmax - 6.0)
        sbp = c["sbp_base"].to_numpy() + 5.0 * ex - 6 * dialysed + 8 * z[:, 0]
        hr = c["hr_base"].to_numpy() + 2.2 * ex + 11 * ill - 5 * np.maximum(0, kmax - 6.5) + 3 * z[:, 1]
        steps = (c["steps_base"].to_numpy() * (1 - 0.07 * ex) * (1 - 0.35 * ill)
                 * (1 - 0.22 * hyperk) * (1 - 0.18 * dialysed) * np.exp(0.22 * z[:, 2]))
        rmssd = c["rmssd_base"].to_numpy() * np.exp(-0.07 * ex - 0.3 * ill + 0.18 * z[:, 3])
        sleep = c["sleep_base"].to_numpy() - 0.25 * ex - 0.9 * ill + 0.5 * z[:, 4]
        worn = z[:, 5] > -1.6      # watch not worn on ~5% of days
        df = pd.DataFrame({"sbp": sbp.round(), "rest_hr": hr.round(1), "steps": np.maximum(steps, 50).round(),
                           "rmssd": rmssd.round(1), "sleep_h": np.clip(sleep, 2, 10).round(1)})
        df.loc[~worn, ["rest_hr", "steps", "rmssd", "sleep_h"]] = np.nan
        return df

    # ------------------------------------------------------------------------------
    def session_starts_by_patient(self) -> dict:
        out: dict[int, list] = {}
        for s in self.session_log:
            out.setdefault(s["idx"], []).append(s["hour"])
        return out
