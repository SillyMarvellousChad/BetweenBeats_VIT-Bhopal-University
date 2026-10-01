"""Stage 1: learn potassium-from-ECG on a separate calibration cohort.

Outputs models/ecg_estimator.joblib and reports/ecg_estimator.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from antaral.cohort import ehr_view, generate_cohort  # noqa: E402
from antaral.ecg_estimator import EcgPotassiumEstimator  # noqa: E402
from antaral.physiology import UnitSimulator  # noqa: E402
from antaral.scheduler import FixedPolicy  # noqa: E402
from antaral.twin import EcgPersonalizer, ecg_model_input  # noqa: E402

N_PATIENTS, N_DAYS, SEED = 150, 42, 101


def collect(cohort: pd.DataFrame, n_days: int, seed: int) -> pd.DataFrame:
    ehr = ehr_view(cohort)
    pol = FixedPolicy(ehr)
    sim = UnitSimulator(cohort, n_days, seed)
    pers = EcgPersonalizer(len(ehr))
    last_end = np.full(len(ehr), -10_000)
    for d in range(n_days):
        sim.set_schedule(d, pol.baseline(d))
    out = []
    for h in range(n_days * 24):
        ev = sim.step()
        for i in ev.session_ends:
            last_end[i] = h
        if ev.ecg_idx is None:
            continue
        idx = ev.ecg_idx
        X = ecg_model_input(ev.ecg_raw, pers.deltas(idx, ev.ecg_raw))
        X["k_true"] = sim.true_K[idx, h]
        X["patient_idx"] = idx
        X["hour"] = h
        out.append(X)
        pers.update(idx, ev.ecg_raw, (h - last_end[idx]) <= 3)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    t0 = time.time()
    cohort = generate_cohort(N_PATIENTS, SEED, id_prefix="CAL")
    data = collect(cohort, N_DAYS, SEED)
    X = data.drop(columns=["k_true", "patient_idx", "hour"])
    est = EcgPotassiumEstimator().fit(X, data["k_true"].to_numpy(), data["patient_idx"].to_numpy())
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    joblib.dump(est, ROOT / "models" / "ecg_estimator.joblib")
    report = {"calibration_patients": N_PATIENTS, "recordings": len(data), **est.metrics}
    (ROOT / "reports" / "ecg_estimator.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
