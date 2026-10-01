"""Synthetic dialysis cohort: EHR records plus hidden physiology.

Each patient row carries two kinds of columns:

* EHR columns (``EHR_COLUMNS``): what a dialysis unit in India would actually have on
  file. These are the only static facts the twin and the risk model may read.
* Hidden columns: the patient's true physiology (diet, gut potassium excretion,
  buffering volume, dialyser efficiency, behaviour). The simulator uses them to
  generate ground truth; the twin never sees them and has to learn their effect
  from data.

Everything here is synthetic. No real patient data is used anywhere in Antaral.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PATTERNS

FIRST_F = ["Lakshmi", "Sunita", "Meena", "Fatima", "Kavita", "Anjali", "Radha", "Geeta",
           "Pooja", "Shabana", "Asha", "Rekha", "Nirmala", "Savitri", "Usha", "Farida",
           "Jyoti", "Kamala", "Seema", "Bhavna", "Mary", "Parvati", "Sarita", "Rukmini"]
FIRST_M = ["Ramesh", "Suresh", "Abdul", "Mahesh", "Rajesh", "Arjun", "Imran", "Vijay",
           "Mohan", "Ravi", "Sanjay", "Harish", "Gopal", "Prakash", "Joseph", "Dinesh",
           "Anil", "Kishore", "Naveen", "Salim", "Manoj", "Balaji", "Tarun", "Venkat"]
SURNAME_INITIALS = list("ABCDGHIJKMNPRSTVY")

CAUSES = ["Diabetic nephropathy", "Hypertensive nephrosclerosis",
          "CKD of unknown aetiology", "Chronic glomerulonephritis",
          "Polycystic kidney disease"]

EHR_COLUMNS = [
    "patient_id", "name", "sex", "age", "dry_weight_kg", "cause", "diabetes",
    "hypertension", "ace_arb", "beta_blocker", "urine_ml_day", "vintage_months",
    "bicarbonate", "haemoglobin", "hd_freq", "pattern",
]

HIDDEN_COLUMNS = [
    "diet_k_meq", "gi_frac", "v_frac", "kd", "fluid_gain_kg_day", "item_rate",
    "log_prob", "ecg_adherence", "ecg_r_amp0", "ecg_t_amp0", "ecg_t_sigma0",
    "ecg_gamma", "sbp_base", "hr_base", "steps_base", "rmssd_base", "sleep_base",
    "constipation_rate", "illness_rate", "feast_rate",
]


def _names(rng: np.random.Generator, sex: np.ndarray) -> list[str]:
    used, out = set(), []
    for s in sex:
        pool = FIRST_F if s == "F" else FIRST_M
        for _ in range(200):
            name = f"{rng.choice(pool)} {rng.choice(SURNAME_INITIALS)}."
            if name not in used:
                break
        used.add(name)
        out.append(name)
    return out


def _balanced_patterns(rng: np.random.Generator, freq: np.ndarray) -> np.ndarray:
    """Spread patients evenly over the weekly patterns, like a real unit would."""
    out = np.empty(len(freq), dtype=object)
    for f, pats in ((2, ["Mon-Thu", "Tue-Fri", "Wed-Sat"]), (3, ["Mon-Wed-Fri", "Tue-Thu-Sat"])):
        idx = np.flatnonzero(freq == f)
        cycle = np.resize(pats, len(idx))
        rng.shuffle(cycle)
        out[idx] = cycle
    return out


def generate_cohort(n: int, seed: int, frac_twice_weekly: float = 0.6,
                    id_prefix: str = "P") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    clip = np.clip

    sex = rng.choice(["F", "M"], n, p=[0.38, 0.62])
    age = clip(rng.normal(52, 13, n), 22, 82).round().astype(int)
    dry = np.where(sex == "F", rng.normal(52, 8, n), rng.normal(60, 9, n))
    dry = clip(dry, 38, 95).round(1)
    diabetes = rng.random(n) < 0.45
    other_causes = rng.choice(CAUSES[1:], n, p=[0.38, 0.27, 0.25, 0.10])
    cause = np.where(diabetes & (rng.random(n) < 0.85), CAUSES[0], other_causes)
    hypertension = rng.random(n) < 0.85
    ace_arb = rng.random(n) < np.where(hypertension, 0.35, 0.08)
    beta_blocker = rng.random(n) < 0.30
    urine = np.where(rng.random(n) < 0.35, 0.0, rng.gamma(2.0, 170.0, n))
    urine = clip(urine, 0, 1200).round(-1)
    vintage = clip(rng.gamma(2.0, 14.0, n), 2, 180).round().astype(int)
    bicarb = clip(rng.normal(20.5, 2.4, n), 14, 27).round(1)
    hb = clip(rng.normal(9.4, 1.3, n), 6, 13.5).round(1)
    freq = np.where(rng.random(n) < frac_twice_weekly, 2, 3)

    df = pd.DataFrame({
        "patient_id": [f"{id_prefix}{i:04d}" for i in range(n)],
        "name": _names(rng, sex),
        "sex": sex, "age": age, "dry_weight_kg": dry, "cause": cause,
        "diabetes": diabetes, "hypertension": hypertension, "ace_arb": ace_arb,
        "beta_blocker": beta_blocker, "urine_ml_day": urine, "vintage_months": vintage,
        "bicarbonate": bicarb, "haemoglobin": hb, "hd_freq": freq,
        "pattern": _balanced_patterns(rng, freq),
    })

    # ---- hidden physiology ----------------------------------------------------
    df["diet_k_meq"] = clip(rng.normal(39, 5, n) - 0.10 * (age - 52), 24, 60)
    df["gi_frac"] = clip(rng.normal(0.33, 0.06, n), 0.18, 0.48)
    v = clip(rng.normal(1.15, 0.10, n), 0.85, 1.50)
    df["v_frac"] = v * np.where(diabetes, 0.88, 1.0) * np.where(beta_blocker, 0.92, 1.0)
    df["kd"] = clip(rng.normal(0.22, 0.035, n), 0.14, 0.32)
    df["fluid_gain_kg_day"] = clip(rng.normal(0.72, 0.22, n) - urine / 1000 * 0.6, 0.10, 1.25)
    df["item_rate"] = clip(rng.gamma(2.0, 0.18, n), 0.03, 1.2)
    df["log_prob"] = rng.beta(3, 2, n)
    df["ecg_adherence"] = rng.beta(5, 1.6, n)
    df["ecg_r_amp0"] = clip(rng.normal(1.0, 0.25, n), 0.45, 1.8)
    df["ecg_t_amp0"] = clip(rng.lognormal(np.log(0.26), 0.40, n), 0.08, 0.70)
    df["ecg_t_sigma0"] = clip(rng.normal(0.048, 0.006, n), 0.035, 0.065)
    df["ecg_gamma"] = clip(rng.normal(0.33, 0.09, n), 0.10, 0.60)
    df["sbp_base"] = clip(rng.normal(142, 14, n), 105, 185)
    df["hr_base"] = clip(rng.normal(78, 8, n), 55, 100)
    df["steps_base"] = clip(rng.lognormal(np.log(4200), 0.45, n), 600, 14000)
    df["rmssd_base"] = clip(rng.lognormal(np.log(22), 0.35, n), 6, 70)
    df["sleep_base"] = clip(rng.normal(6.6, 0.7, n), 4.5, 8.5)
    df["constipation_rate"] = clip(rng.gamma(1.5, 1 / 60, n), 0, 0.08)
    df["illness_rate"] = clip(rng.gamma(1.5, 1 / 50, n), 0, 0.08)
    df["feast_rate"] = clip(rng.gamma(2.0, 1 / 40, n), 0, 0.15)
    return df


def ehr_view(cohort: pd.DataFrame) -> pd.DataFrame:
    """The static record the twin is allowed to see."""
    return cohort[EHR_COLUMNS].copy()
