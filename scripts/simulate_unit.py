"""Stage 4: does risk-based scheduling actually prevent hyperkalaemia?

A 60-patient unit is simulated for 10 weeks (2 warm-up weeks on the usual schedule,
then 8 evaluation weeks) under two policies, with the same patients, meals,
illnesses and machines (common random numbers):

1. Fixed weekly pattern (today's practice)
2. Antaral: each evening the twin forecasts every patient's potassium under "dialyse
   tomorrow" vs "wait", and the integer programme picks tomorrow's list

Capacity is identical: the busiest day of the fixed schedule. Each patient keeps
their weekly number of sessions (no extra sessions are created).

Outputs reports/unit_simulation.csv and reports/unit_simulation.json
"""
from __future__ import annotations

import argparse
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
from antaral.pipeline import run_unit, unit_outcomes  # noqa: E402
from antaral.scheduler import AntaralPolicy, FixedPolicy  # noqa: E402

N_PATIENTS, N_DAYS, WARMUP = 60, 70, 14


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=6)
    args = ap.parse_args()
    est = joblib.load(ROOT / "models" / "ecg_estimator.joblib")
    risk = joblib.load(ROOT / "models" / "risk_model.joblib")
    rows = []
    for rep in range(args.replicates):
        seed = 5000 + rep
        cohort = generate_cohort(N_PATIENTS, seed, id_prefix="U")
        ehr = ehr_view(cohort)
        cap = FixedPolicy(ehr).capacity()
        arms = {
            "Fixed weekly pattern": (FixedPolicy(ehr), "ml"),
            "Antaral (same machines)": (AntaralPolicy(ehr, cap, warmup_days=WARMUP), "twin"),
            "Antaral (+ standby machine for re-timing)": (AntaralPolicy(ehr, cap + 3, warmup_days=WARMUP), "twin"),
        }
        for arm, (pol, mode) in arms.items():
            t0 = time.time()
            res = run_unit(cohort, N_DAYS, seed, policy=pol, ecg_estimator=est, risk_model=risk,
                           risk_mode=mode, warmup_days=WARMUP, n_samples=200)
            out = unit_outcomes(res, WARMUP)
            decs = getattr(pol, "decisions", [])
            pulled = sum(len(d["moved_in"]) for d in decs)
            moves = pulled + sum(len(d["deferred"]) for d in decs)
            out.update(replicate=rep, policy=arm, capacity=getattr(pol, "cap", cap), reschedules=moves, pulled_forward=pulled,
                       reschedules_per_week=moves / ((N_DAYS - WARMUP) / 7),
                       overflow=getattr(pol, "overflow", 0), seconds=round(time.time() - t0, 1))
            rows.append(out)
            print(f"rep {rep} | {arm:40s} | episodes {out['hyperk_episodes']:4d} | hours>=6 "
                  f"{out['patient_hours_k_ge_6']:6d} | pre-HD>=6 {out['pre_dialysis_k_ge_6']:4d} | "
                  f"sessions {out['sessions']} | moves {moves} | {out['seconds']}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "reports" / "unit_simulation.csv", index=False)
    metrics = ["hyperk_episodes", "patient_hours_k_ge_6", "patient_hours_k_ge_6_5", "patient_hours_k_ge_7",
               "danger_hours_weighted", "pre_dialysis_k_ge_6",
               "patients_with_any_k_ge_6", "sessions", "reschedules_per_week"]
    summary = df.groupby("policy")[metrics].agg(["mean", "std"]).round(2)
    base = df[df.policy == "Fixed weekly pattern"].set_index("replicate")
    rel = {}
    for arm in df.policy.unique():
        if arm == "Fixed weekly pattern":
            continue
        a = df[df.policy == arm].set_index("replicate")
        rel[arm] = {m: {"mean_reduction_pct": float(100 * (1 - a[m].sum() / base[m].sum())),
                        "per_replicate_pct": (100 * (1 - a[m] / base[m])).round(1).tolist()}
                    for m in ["hyperk_episodes", "patient_hours_k_ge_6", "patient_hours_k_ge_6_5",
                              "patient_hours_k_ge_7", "danger_hours_weighted", "pre_dialysis_k_ge_6"]}
    report = {"patients_per_unit": N_PATIENTS, "evaluation_weeks": (N_DAYS - WARMUP) / 7,
              "replicates": args.replicates,
              "summary": {f"{k[0]}|{k[1]}": v for k, v in summary.to_dict().items()},
              "reduction_vs_fixed": rel}
    (ROOT / "reports" / "unit_simulation.json").write_text(json.dumps(report, indent=2, default=str))
    print(summary.to_string())
    print(json.dumps(rel, indent=2))


if __name__ == "__main__":
    main()
