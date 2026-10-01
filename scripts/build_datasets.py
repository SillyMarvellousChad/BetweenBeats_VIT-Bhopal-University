"""Stage 2: simulate training and test units and record decision-time features.

Each evening at 9 PM, for every patient, we record what the unit can see (EHR, twin
state and forecast, wearables, food diary, labs) plus the ground-truth label: does true
potassium reach 6.0 mEq/L before the next session? Train and test cohorts are
different synthetic patients, so the evaluation is on people the models never saw.

Outputs data/train_features.csv.gz and data/test_features.csv.gz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from antaral.cohort import generate_cohort  # noqa: E402
from antaral.pipeline import run_unit  # noqa: E402

COHORTS = {"train": (400, 202), "test": (200, 303)}
N_DAYS, WARMUP = 84, 14


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0, help="shrink cohorts for a quick run")
    args = ap.parse_args()
    est = joblib.load(ROOT / "models" / "ecg_estimator.joblib")
    (ROOT / "data").mkdir(exist_ok=True)
    for split, (n, seed) in COHORTS.items():
        t0 = time.time()
        n = max(20, int(n * args.scale))
        cohort = generate_cohort(n, seed, id_prefix=split[:2].upper())
        res = run_unit(cohort, N_DAYS, seed, ecg_estimator=est, warmup_days=WARMUP,
                       record_features=True, physics_only_twin=True, n_samples=200)
        df = res.features
        df["patient_id"] = cohort["patient_id"].to_numpy()[df["patient_idx"]]
        df.to_csv(ROOT / "data" / f"{split}_features.csv.gz", index=False)
        cohort.to_csv(ROOT / "data" / f"{split}_cohort.csv", index=False)
        print(f"{split}: {n} patients, {len(df)} decision points, event rate {df['y'].mean():.3f} "
              f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
