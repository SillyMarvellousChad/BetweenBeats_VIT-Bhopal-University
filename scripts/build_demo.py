"""Stage 5: build the dashboard's demo unit (a frozen Tuesday evening at 9 PM).

A 40-patient unit runs on the usual schedule for two weeks, then with Antaral for two
more. The snapshot is taken on the evening Antaral plans Wednesday's list.
One patient carries the pitch story: Lakshmi R., twice weekly (Mon-Thu), diabetic, on
an ARB, who drank tender coconut water on Monday and Tuesday and logged it.

Output: data/demo_snapshot.joblib
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from antaral import config as C  # noqa: E402
from antaral.cohort import ehr_view, generate_cohort  # noqa: E402
from antaral.features import build_features  # noqa: E402
from antaral.pipeline import run_unit  # noqa: E402
from antaral.scheduler import AntaralPolicy, FixedPolicy  # noqa: E402

N, SEED = 40, 777
DECISION_DAY = 29          # Tuesday (day 0 is a Monday) -> plans Wednesday, day 30
STORY = 0                  # index of Lakshmi


def main() -> None:
    est = joblib.load(ROOT / "models" / "ecg_estimator.joblib")
    risk = joblib.load(ROOT / "models" / "risk_model.joblib")
    cohort = generate_cohort(N, SEED, id_prefix="BLR-")
    story = {"name": "Lakshmi R.", "sex": "F", "age": 58, "dry_weight_kg": 54.0, "diabetes": True,
             "cause": "Diabetic nephropathy", "ace_arb": True, "beta_blocker": False, "hypertension": True,
             "urine_ml_day": 150.0, "hd_freq": 2, "pattern": "Mon-Thu", "bicarbonate": 19.5,
             "vintage_months": 26, "diet_k_meq": 40.0, "v_frac": 0.98, "item_rate": 0.15, "log_prob": 0.9,
             "ecg_adherence": 0.95, "ecg_gamma": 0.42, "illness_rate": 0.0, "feast_rate": 0.0}
    for k, v in story.items():
        cohort.loc[STORY, k] = v
    # Monday 4 PM and Tuesday 1 PM: tender coconut water (logged); Tuesday 5 PM: banana (logged)
    scripted = [(STORY, 28, 16, "Tender coconut water", True), (STORY, 29, 13, "Tender coconut water", True),
                (STORY, 29, 17, "Banana", True)]
    ehr = ehr_view(cohort)
    cap = FixedPolicy(ehr).capacity()
    policy = AntaralPolicy(ehr, cap + 3, warmup_days=14)
    t = DECISION_DAY * 24 + C.DECISION_HOUR
    res = run_unit(cohort, DECISION_DAY + 8, SEED, policy=policy, ecg_estimator=est, risk_model=risk,
                   risk_mode="twin", record_twin=True, keep_beats=set(range(N)), scripted_items=scripted,
                   n_samples=400, stop_hour=t + 1)

    plan_day = DECISION_DAY + 1
    decision = policy.decisions[-1]
    assert decision["day"] == plan_day
    plan = res.sim.schedule[plan_day]
    usual = FixedPolicy(ehr)
    # each patient's next session under the usual pattern, for the risk table
    nxt = np.array([usual.next_start(i, t) for i in range(N)])
    planned = {i: plan_day * 24 + C.SHIFT_STARTS[s] for i, s in plan}
    deferred_to = plan_day + 1 if (plan_day + 1) % 7 != C.CLOSED_WEEKDAY else plan_day + 2
    nxt_plan = np.array([planned[i] if i in planned
                         else deferred_to * 24 + C.SHIFT_STARTS[usual.shift[i]] if i in decision["deferred"]
                         else nxt[i] for i in range(N)])
    feats = build_features(t, np.arange(N), nxt, ehr, res.twin, res.record, n_samples=400)
    feats["risk"] = risk.predict_proba(feats)
    shap = risk.explain(feats)

    snap = {
        "t": t, "plan_day": plan_day, "capacity": policy.cap, "usual_capacity": cap,
        "ehr": ehr, "cohort_hidden": cohort,
        "twin": res.twin, "record": res.record,
        "decision": decision, "plan": plan, "baseline_plan": usual.baseline(plan_day),
        "usual_shift": usual.shift, "next_usual_start": nxt, "next_planned_start": nxt_plan,
        "features": feats, "shap": shap,
        "true_K": res.sim.true_K[:, : t + 1], "in_session": res.sim.in_session[:, : t + 1],
        "session_log": res.sim.session_log, "ecg_log": res.sim.ecg_log,
        "story_idx": STORY,
    }
    (ROOT / "data").mkdir(exist_ok=True)
    joblib.dump(snap, ROOT / "data" / "demo_snapshot.joblib", compress=3)
    top = feats.assign(name=ehr["name"]).sort_values("risk", ascending=False)[["name", "risk", "tw_k", "hours_to_next"]]
    print(top.head(8).round(2).to_string(index=False))
    print("moved in:", [ehr.at[i, "name"] for i in decision["moved_in"]],
          "deferred:", [ehr.at[i, "name"] for i in decision["deferred"]])
    j = int(np.flatnonzero(decision["candidates"] == STORY)[0]) if STORY in decision["candidates"] else None
    if j is not None:
        print("Lakshmi: danger-hours if wait", round(decision["r_no"][j], 1), "if tomorrow", round(decision["r_yes"][j], 1))
    print("true K Lakshmi now", round(float(res.sim.true_K[STORY, t]), 2), "twin", round(float(res.twin.x[STORY, 0]), 2))


if __name__ == "__main__":
    main()
