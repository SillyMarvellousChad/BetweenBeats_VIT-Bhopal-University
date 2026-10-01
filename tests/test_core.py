"""Sanity tests for the physiology, the ECG sensor, the twin and the scheduler's rules."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from antaral import config as C  # noqa: E402
from antaral.cohort import EHR_COLUMNS, ehr_view, generate_cohort  # noqa: E402
from antaral.ecg import beat_params, extract_features, synth_beats  # noqa: E402
from antaral.physiology import UnitSimulator  # noqa: E402
from antaral.scheduler import AntaralPolicy, FixedPolicy  # noqa: E402
from antaral.twin import TwinBank  # noqa: E402


@pytest.fixture(scope="module")
def unit():
    coh = generate_cohort(30, 11)
    pol = FixedPolicy(ehr_view(coh))
    sim = UnitSimulator(coh, 21, 11)
    for d in range(21):
        sim.set_schedule(d, pol.baseline(d))
    twin = TwinBank(ehr_view(coh), None, use_ecg=False)
    for _ in range(21 * 24):
        twin.step(sim.step())
    return coh, sim, twin


def test_twin_sees_only_ehr():
    coh = generate_cohort(5, 1)
    assert list(ehr_view(coh).columns) == EHR_COLUMNS
    assert "diet_k_meq" not in ehr_view(coh).columns


def test_potassium_rises_between_and_falls_during_dialysis(unit):
    _, sim, _ = unit
    s = [x for x in sim.session_log if x["day"] >= 7]
    drops = [sim.true_K[x["idx"], x["hour"] + x["duration"] - 1] - x["pre_k_true"] for x in s]
    assert np.mean(drops) < -1.0
    post = np.array([sim.true_K[x["idx"], x["hour"] + x["duration"] - 1] for x in s])
    assert 2.5 <= post.mean() <= 4.2
    pre = np.array([x["pre_k_true"] for x in s])
    assert 4.0 <= pre.mean() <= 6.0


def test_twice_weekly_patients_run_higher(unit):
    coh, sim, _ = unit
    pre = {2: [], 3: []}
    for x in sim.session_log:
        if x["day"] >= 7:
            pre[int(coh.hd_freq[x["idx"]])].append(x["pre_k_true"])
    assert np.mean(pre[2]) > np.mean(pre[3])


def test_peaked_t_wave_with_rising_potassium():
    rng = np.random.default_rng(0)
    k = np.repeat([4.0, 5.0, 6.0, 7.0], 300)
    f = extract_features(synth_beats(beat_params(k, 1.0, 0.26, 0.048, 0.35, rng)))
    med = f.groupby(k).median()
    assert med["t_r_ratio"].is_monotonic_increasing
    assert med["t_fwhm"].is_monotonic_decreasing
    assert med.loc[7.0, "qrs_ms"] > med.loc[4.0, "qrs_ms"]


def test_twin_with_labs_only_tracks_truth(unit):
    _, sim, twin = unit
    err = np.abs(twin.x[:, 0] - sim.true_K[:, -1])
    assert np.median(err) < 1.0
    assert np.all(np.sqrt(twin.P[:, 0, 0]) > 0)


def test_forecast_respects_dialysis(unit):
    _, _, twin = unit
    paths = twin.forecast(np.arange(5), 30, sessions=[[(10, 4)]] * 5, n_samples=100)
    assert np.all(paths[:, :, 13].mean(axis=1) < paths[:, :, 9].mean(axis=1))


def test_scheduler_rules():
    coh = generate_cohort(40, 3)
    ehr = ehr_view(coh)
    fixed = FixedPolicy(ehr)
    cap = fixed.capacity()
    pol = AntaralPolicy(ehr, cap, warmup_days=0)

    class Rec:
        sessions = [[] for _ in range(40)]

    class Ctx:
        record = Rec()
        rng = np.random.default_rng(0)

        def horizon_burden(self, idx, s1, end):
            return self.rng.uniform(0, 30, len(idx))

        def window_risk_simple(self, idx, s):
            return np.zeros(len(idx))

    ctx = Ctx()
    weekly = np.zeros(40, int)
    for day in range(14):
        plan = pol.plan(day, ctx)
        assert len(plan) <= cap
        if day % 7 == C.CLOSED_WEEKDAY:
            assert plan == []
        if day % 7 == 0:
            weekly[:] = 0
        for i, _ in plan:
            ctx.record.sessions[i].append((day * 24 + 7, 4))
            weekly[i] += 1
        assert np.all(weekly <= ehr["hd_freq"].to_numpy())
