"""Dialysis-unit scheduling policies.

FixedPolicy
    Today's practice: every patient keeps the same weekdays and shift
    (e.g. Mon-Thu, 7 AM shift), whatever their potassium is doing.

AntaralPolicy
    Same machines, same number of sessions per patient. Each evening at 9 PM it
    decides tomorrow's list by solving a small integer programme:

        maximise   sum_i x_i * coef_i,
                   coef_i = max(benefit_i, 0) + lambda   if i is due tomorrow
                            benefit_i - lambda           otherwise
        subject to sum_i x_i <= capacity
                   x_i = 1  for patients who cannot safely wait any longer (max gap)
                   x_i = 0  for patients not eligible (min gap, or weekly quota used)

    benefit_i = expected danger-hours if NOT dialysed tomorrow minus expected danger-hours
    if dialysed tomorrow, from the twin's what-if forecast over the next two gaps
    (danger-hours: each hour with K >= 6.0 counts 1, >= 6.5 counts 2, >= 7.0 counts 3).
    lambda (6 danger-hours) keeps the plan stable: an off-day patient is only pulled in
    when that saves at least 6 danger-hours, and a due patient only gives up their slot
    when someone else would gain at least 12 danger-hours more.
    The highest-risk patients are put on the earliest shift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from . import config as C


def next_open_day(day: int) -> int:
    d = day + 1
    while d % 7 == C.CLOSED_WEEKDAY:
        d += 1
    return d


class FixedPolicy:
    name = "Fixed weekly pattern"

    def __init__(self, ehr: pd.DataFrame):
        self.ehr = ehr.reset_index(drop=True)
        self.days = [C.PATTERNS[p] for p in self.ehr["pattern"]]
        self.shift = (self.ehr.groupby("pattern").cumcount().to_numpy() % len(C.SHIFT_STARTS)).astype(int)
        self.freq = self.ehr["hd_freq"].to_numpy(int)
        self.decisions: list[dict] = []

    def baseline(self, day: int) -> list:
        wd = day % 7
        return [(i, int(self.shift[i])) for i in range(len(self.days)) if wd in self.days[i]]

    def capacity(self) -> int:
        return max(len(self.baseline(d)) for d in range(6))

    def next_pattern_day(self, i: int, after_day: int) -> int:
        d = after_day + 1
        while (d % 7) not in self.days[i]:
            d += 1
        return d

    def next_start(self, i: int, t: int) -> int:
        """Absolute hour of the next planned session start strictly after hour t."""
        day = t // 24
        for d in range(day, day + 8):
            if d % 7 in self.days[i]:
                s = d * 24 + C.SHIFT_STARTS[self.shift[i]]
                if s > t:
                    return s
        raise RuntimeError("no session within a week")

    def plan(self, day: int, ctx) -> list:
        return self.baseline(day)


class AntaralPolicy(FixedPolicy):
    """Risk-based daily re-planning under the same machines and weekly session quotas.

    Rules every plan respects:
      * capacity: never more patients than the fixed schedule's busiest day
      * weekly quota (Mon-Sat calendar week): each patient keeps their 2 or 3 sessions
      * minimum gap: 2 days (twice weekly) or 1 day (thrice weekly)
      * safety net: anyone at their maximum gap, or who would otherwise miss this
        week's quota, is scheduled no matter what
    The benefit of dialysing patient i tomorrow is judged over BOTH upcoming gaps,
    because pulling a session earlier makes the following gap longer.
    """

    def __init__(self, ehr: pd.DataFrame, capacity: int | None = None, lam: float = 6.0,
                 warmup_days: int = 14, name: str = "Antaral (risk-based)"):
        super().__init__(ehr)
        self.cap = capacity or self.capacity()
        self.lam = lam
        self.warmup_days = warmup_days
        self.name = name
        self.overflow = 0
        self.g = np.array([C.MIN_GAP_DAYS[f] for f in self.freq])
        self.G = np.array([C.MAX_GAP_DAYS[f] for f in self.freq]) + 1

    def _week_state(self, ctx, day: int):
        n = len(self.days)
        last_day = np.full(n, -99)
        done = np.zeros(n, int)
        wk = day - day % 7
        for i in range(n):
            ss = ctx.record.sessions[i]
            if ss:
                last_day[i] = ss[-1][0] // 24
                done[i] = sum(1 for s, _ in ss if wk <= s // 24 < day)
        return last_day, done

    def _max_fit(self, i: int, first: int, last_sat: int, last_day: int) -> int:
        """How many sessions fit from day ``first`` to Saturday ``last_sat``."""
        count, prev = 0, last_day
        for d in range(first, last_sat + 1):
            if d - prev >= self.g[i]:
                count += 1
                prev = d
        return count

    def _next_after(self, i: int, a: int, rem: int) -> int:
        """Next session day after a session on day ``a`` with ``rem`` sessions left that week."""
        sat = a - a % 7 + 5
        if rem > 0:
            for d in range(a + self.g[i], sat + 1):
                if d % 7 in self.days[i]:
                    return d
            if a + self.g[i] <= sat:
                return a + self.g[i]
        d = a - a % 7 + 7
        while d % 7 not in self.days[i]:
            d += 1
        return max(d, a + self.g[i])

    def _due(self, i: int, day: int, last_day: int, rem: int) -> bool:
        if rem <= 0:
            return False
        if day % 7 in self.days[i]:
            return True
        wk = day - day % 7
        missed = [d for d in range(wk, day) if d % 7 in self.days[i] and d > last_day]
        return bool(missed)

    def plan(self, day: int, ctx) -> list:
        if day % 7 == C.CLOSED_WEEKDAY:
            return []
        if day < self.warmup_days:
            return self.baseline(day)
        n = len(self.days)
        last_day, done = self._week_state(ctx, day)
        rem = self.freq - done
        gap = day - last_day
        eligible = (rem > 0) & (gap >= self.g)
        sat = day - day % 7 + 5
        fit_later = np.array([self._max_fit(i, day + 1, sat, last_day[i]) for i in range(n)])
        mandatory = eligible & ((fit_later < rem) | (gap >= self.G))
        due = np.array([self._due(i, day, last_day[i], rem[i]) for i in range(n)])

        cand = np.flatnonzero(eligible)
        if cand.size == 0:
            return []
        sh = lambda i, d: d * 24 + C.SHIFT_STARTS[self.shift[i]]  # noqa: E731
        y1 = np.array([sh(i, day) for i in cand])
        y2 = np.array([sh(i, self._next_after(i, day, rem[i] - 1)) for i in cand])
        n1_day = []
        for i in cand:
            if due[i]:
                d = day + 1 if (day + 1) % 7 != C.CLOSED_WEEKDAY else day + 2
            else:
                d = self._next_after(i, max(last_day[i], day - self.g[i]), rem[i])
                d = max(d, day + 1)
            n1_day.append(d)
        n1 = np.array([sh(i, d) for i, d in zip(cand, n1_day)])
        n2 = np.array([sh(i, self._next_after(i, d, rem[i] - 1 if d // 7 == day // 7 else self.freq[i] - 1))
                       for i, d in zip(cand, n1_day)])
        horizon = np.minimum(y2, n2)
        r_yes = ctx.horizon_burden(cand, y1, horizon)
        r_no = ctx.horizon_burden(cand, n1, horizon)
        benefit = r_no - r_yes            # expected danger-hours avoided by dialysing tomorrow
        # due patients keep their slot unless the slot is worth far more to someone else
        coef = np.where(due[cand], np.maximum(benefit, 0) + self.lam, benefit - self.lam)

        lb = mandatory[cand].astype(float)
        ub = np.ones(cand.size)
        n_mand = int(lb.sum())
        if n_mand > self.cap:                      # more urgent patients than machines
            keep = np.argsort(-(coef + 10 * lb))[: self.cap]
            new_lb = np.zeros_like(lb)
            new_lb[keep] = lb[keep]
            lb = new_lb
            self.overflow += n_mand - self.cap
        res = milp(c=-coef, constraints=[LinearConstraint(np.ones((1, cand.size)), 0, self.cap)],
                   integrality=np.ones(cand.size), bounds=Bounds(lb, ub))
        x = np.round(res.x).astype(bool) if res.success else (lb > 0)
        chosen = cand[x]

        # earliest shift for the patients whose potassium is closest to danger
        urgency = ctx.window_risk_simple(chosen, np.array([sh(i, day) for i in chosen]))
        order = np.argsort(-urgency)
        per_shift = int(np.ceil(self.cap / len(C.SHIFT_STARTS)))
        plan = [(int(chosen[o]), min(k // per_shift, len(C.SHIFT_STARTS) - 1)) for k, o in enumerate(order)]

        self.decisions.append({
            "day": day, "candidates": cand, "r_yes": r_yes, "r_no": r_no, "benefit": benefit,
            "due": due[cand], "chosen": chosen,
            "moved_in": sorted(int(i) for i in chosen if not due[i]),
            "deferred": sorted(int(i) for i in cand if due[i] and i not in set(chosen.tolist())),
            "mandatory": cand[lb > 0],
        })
        return plan
