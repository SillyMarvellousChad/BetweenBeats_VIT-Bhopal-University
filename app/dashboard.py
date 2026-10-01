"""Antaral — doctor's dashboard for the dialysis digital twin.

Run:  streamlit run app/dashboard.py
Everything shown is synthetic. Antaral is a research prototype for clinician decision
support; it is not a medical device and must not guide real treatment.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from antaral import config as C  # noqa: E402
from antaral.ecg import T as ECG_T  # noqa: E402
from antaral.ecg import synth_beats  # noqa: E402
from antaral.features import FEATURE_LABELS, build_features  # noqa: E402
from antaral.scheduler import AntaralPolicy  # noqa: E402

st.set_page_config(page_title="Antaral · Dialysis Digital Twin", page_icon="🩺", layout="wide")

TEAL, RED, AMBER, GREEN, INK, MUTED = "#0F766E", "#DC2626", "#D97706", "#059669", "#1C1917", "#78716C"
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1350px;}
.ant-card {background: #FFFFFF; border: 1px solid #E7E5E4; border-radius: 12px; padding: 14px 16px; margin-bottom: 10px;}
.ant-card h4 {margin: 0 0 4px 0; font-size: 1.02rem;}
.ant-pill {display:inline-block; padding: 2px 9px; border-radius: 999px; font-size: 0.78rem; font-weight: 600; margin-right: 6px;}
.ant-high {background:#FEE2E2; color:#991B1B;} .ant-watch {background:#FEF3C7; color:#92400E;}
.ant-low {background:#D1FAE5; color:#065F46;} .ant-move {background:#CCFBF1; color:#115E59;}
.ant-muted {color:#78716C; font-size: 0.86rem;}
.ant-chair {background:#FFFFFF; border:1px solid #E7E5E4; border-left: 5px solid #A8A29E; border-radius: 8px;
            padding: 6px 10px; margin-bottom: 6px; font-size: 0.9rem;}
.ant-banner {background:#FFF7ED; border:1px solid #FED7AA; border-radius:10px; padding:8px 12px; font-size:0.85rem;}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- data
@st.cache_resource
def load():
    snap = joblib.load(ROOT / "data" / "demo_snapshot.joblib")
    risk = joblib.load(ROOT / "models" / "risk_model.joblib")
    rep = {}
    for name in ["model_metrics", "unit_simulation", "ecg_estimator"]:
        p = ROOT / "reports" / f"{name}.json"
        rep[name] = json.loads(p.read_text()) if p.exists() else None
    for name in ["ablation", "unit_simulation", "feature_importance"]:
        p = ROOT / "reports" / f"{name}.csv"
        rep[name + "_csv"] = pd.read_csv(p) if p.exists() else None
    p = ROOT / "reports" / "roc_curves.json"
    rep["roc"] = json.loads(p.read_text()) if p.exists() else None
    return snap, risk, rep


snap, risk_model, reports = load()
ehr: pd.DataFrame = snap["ehr"]
twin = snap["twin"]
record = snap["record"]
t_now: int = snap["t"]
plan_day: int = snap["plan_day"]
feats: pd.DataFrame = snap["features"]
decision = snap["decision"]
N = len(ehr)
ALERT = (reports["model_metrics"] or {}).get("final", {}).get("alert_threshold_85_sensitivity", 0.35)


def when(h: int) -> str:
    d = h // 24
    rel = {plan_day - 1: "Today", plan_day: "Tomorrow"}.get(d, C.WEEKDAYS[d % 7])
    return f"{rel} {h % 24:02d}:00" if rel in ("Today", "Tomorrow") else f"{C.WEEKDAYS[d % 7]} {h % 24:02d}:00"


def band(p: float) -> tuple[str, str]:
    if p >= 0.6:
        return "High", "ant-high"
    if p >= ALERT:
        return "Watch", "ant-watch"
    return "Stable", "ant-low"


plan = dict(snap["plan"])          # idx -> shift tomorrow (Antaral)
base_plan = dict(snap["baseline_plan"])
moved_in = set(decision["moved_in"])
deferred = set(decision["deferred"])
cand_pos = {int(i): j for j, i in enumerate(decision["candidates"])}


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 🩺 Antaral")
    st.caption("Digital twin for the gap between dialysis sessions")
    page = st.radio("View", ["Unit tonight", "Patient twin", "Evidence", "About"], label_visibility="collapsed")
    st.divider()
    st.markdown(f"**Shanti Dialysis Centre** · demo unit  \n{N} patients · {snap['usual_capacity'] // 3} chairs + 1 standby  \n"
                f"Now: **{C.WEEKDAYS[(plan_day - 1) % 7]}, 9:00 PM** — planning **{C.WEEKDAYS[plan_day % 7]}**")
    st.divider()
    st.markdown('<div class="ant-banner">All patients and data are <b>synthetic</b>. Research prototype for '
                'clinician decision support, not a medical device.</div>', unsafe_allow_html=True)


# ============================================================================ UNIT VIEW
def unit_view() -> None:
    st.markdown(f"### Tonight's review · plan for {C.WEEKDAYS[plan_day % 7]}")
    f = feats.assign(name=ehr["name"], pattern=ehr["pattern"], hd=ehr["hd_freq"])
    high = int((f["risk"] >= 0.6).sum())
    watch = int(((f["risk"] >= ALERT) & (f["risk"] < 0.6)).sum())
    critical = int((f["tw_k"] >= C.K_CRITICAL).sum())
    c = st.columns(5)
    c[0].metric("Patients", N)
    c[1].metric("Sessions tomorrow", len(plan), f"{len(plan) - len(base_plan):+d} vs usual" if len(plan) != len(base_plan) else None)
    c[2].metric("High risk before next session", high)
    c[3].metric("Watch list", watch)
    c[4].metric("Twin estimate ≥ 7.0 now", critical)

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("#### Antaral's suggestions")
        any_card = False
        for i in list(moved_in) + list(deferred):
            any_card = True
            j = cand_pos.get(i)
            name = ehr.at[i, "name"]
            r = feats.at[i, "risk"]
            if i in moved_in:
                saved = decision["r_no"][j] - decision["r_yes"][j]
                usual_next = when(int(snap["next_usual_start"][i]))
                title = f"Pull forward <b>{name}</b> to tomorrow {C.SHIFT_STARTS[plan[i]]:02d}:00"
                why = (f"Usual session {usual_next}. {r:.0%} risk of K ≥ 6.0 before it. "
                       f"Dialysing tomorrow avoids ≈{saved:.0f} danger-hours over the next two gaps.")
                pill = '<span class="ant-pill ant-move">Move in</span>'
            else:
                title = f"Defer <b>{name}</b> by one day"
                why = (f"Stays low risk ({decision['r_no'][j]:.0f} danger-hours expected if deferred); "
                       "frees a chair for a patient who needs it more.")
                pill = '<span class="ant-pill ant-low">Defer</span>'
            st.markdown(f'<div class="ant-card">{pill}<h4>{title}</h4><div class="ant-muted">{why}</div></div>',
                        unsafe_allow_html=True)
            st.checkbox("Approve", key=f"appr_{i}", value=False)
        if not any_card:
            st.info("No schedule changes needed tonight.")

        kept = f[(f["risk"] >= 0.6) & ~f.index.isin(list(moved_in))].sort_values("risk", ascending=False)
        if len(kept):
            st.markdown("#### High risk, schedule kept — consider other action")
            for i, row in kept.iterrows():
                j = cand_pos.get(i)
                reason = ""
                if j is not None and decision["r_yes"][j] >= decision["r_no"][j]:
                    reason = ("Moving the session earlier would lengthen the following gap and add danger-hours "
                              f"({decision['r_yes'][j]:.0f} vs {decision['r_no'][j]:.0f}). ")
                elif i in plan:
                    reason = f"Already on tomorrow's list (shift {C.SHIFT_STARTS[plan[i]]:02d}:00). "
                urgent = " <b>Twin estimate ≥ 7.0 now — urgent lab / ECG.</b>" if row["tw_k"] >= C.K_CRITICAL else ""
                st.markdown(
                    f'<div class="ant-card"><span class="ant-pill ant-high">{row["risk"]:.0%}</span>'
                    f'<h4>{row["name"]}</h4><div class="ant-muted">{reason}Open the twin to compare a binder or '
                    f'diet call.{urgent}</div></div>', unsafe_allow_html=True)

    with right:
        st.markdown(f"#### {C.WEEKDAYS[plan_day % 7]}'s chairs")
        cols = st.columns(3)
        for s, start in enumerate(C.SHIFT_STARTS):
            with cols[s]:
                st.markdown(f"**{start:02d}:00 – {start + 4:02d}:00**")
                ids = sorted([i for i, sh in plan.items() if sh == s], key=lambda i: -feats.at[i, "risk"])
                for i in ids:
                    p = feats.at[i, "risk"]
                    colr = RED if p >= 0.6 else AMBER if p >= ALERT else "#A8A29E"
                    tag = " ⇢ moved in" if i in moved_in else ""
                    st.markdown(f'<div class="ant-chair" style="border-left-color:{colr}">{ehr.at[i, "name"]}'
                                f'<span class="ant-muted">{tag}</span></div>', unsafe_allow_html=True)
        st.caption("Highest-risk patients are seated on the earliest shift. Same weekly sessions per patient.")

    st.markdown("#### Every patient: risk of K ≥ 6.0 before their next session")
    table = pd.DataFrame({
        "Patient": ehr["name"], "Schedule": ehr["pattern"],
        "Next session": [when(int(h)) for h in snap["next_planned_start"]],
        "Risk": feats["risk"].round(3),
        "Twin K now": feats["tw_k"].round(2), "± (1 sd)": feats["tw_k_sd"].round(2),
        "Drift / day": feats["tw_drift_day"].round(2),
        "Watch ECG K": feats["ecg_k_last"].round(1),
        "Weight gain %": feats["fluid_obs_pct"].round(1),
        "Logged K food (since HD)": feats["logged_meq_since_session"].round(0),
    }).sort_values("Risk", ascending=False)
    st.dataframe(table, hide_index=True, width="stretch", height=420, column_config={
        "Risk": st.column_config.ProgressColumn("Risk", min_value=0.0, max_value=1.0, format="percent"),
        "Drift / day": st.column_config.NumberColumn(help="Learned personal potassium drift (mEq/L per day) beyond what the EHR model expects"),
    })


# ============================================================================ PATIENT VIEW
policy_helper = AntaralPolicy(ehr, snap["capacity"])


def week_done(i: int, day: int) -> int:
    wk = day - day % 7
    return sum(1 for s, _ in record.sessions[i] if wk <= s // 24 < day)


def following_session(i: int, s1: int) -> int:
    d1 = s1 // 24
    rem = ehr.at[i, "hd_freq"] - week_done(i, d1) - 1
    d2 = policy_helper._next_after(i, d1, rem)
    return d2 * 24 + C.SHIFT_STARTS[snap["usual_shift"][i]]


def session_options(i: int) -> dict:
    opts = {}
    usual = int(snap["next_planned_start"][i])
    opts[f"Current plan · {when(usual)}"] = usual
    for d in range(plan_day, plan_day + 3):
        if d % 7 == C.CLOSED_WEEKDAY:
            continue
        for s in C.SHIFT_STARTS:
            h = d * 24 + s
            if h != usual:
                opts[when(h)] = h
    return opts


def extra_intake(horizon: int, coconut: int, diet_cut: bool) -> np.ndarray:
    x = np.zeros((1, horizon + 200))
    tomorrow_1pm = plan_day * 24 + 13 - (t_now + 1)
    for k in range(coconut):
        x[0, tomorrow_1pm + 3 * k] += C.K_ITEMS["Tender coconut water"]
    if diet_cut:   # dietitian call: potassium-rich foods cut by a quarter for 3 days
        for h in range(72):
            hr = (t_now + 1 + h) % 24
            if hr in C.MEAL_FRACTIONS:
                x[0, h] -= 0.25 * 40 * C.MEAL_FRACTIONS[hr]
    return x


def scenario(i: int, s1: int, coconut: int, binder: bool, diet_cut: bool, n: int = 500) -> dict:
    s2 = following_session(i, s1)
    base = t_now + 1
    horizon = s2 - base + 16
    extra = extra_intake(horizon, coconut, diet_cut)
    kw = {"extra_meq": extra, "binder_meq_day": 15.0 if binder else 0.0}
    paths = twin.forecast(np.array([i]), horizon, sessions=[[(s1 - base, 4), (s2 - base, 4)]],
                          n_samples=n, rng=np.random.default_rng(11), **kw)[0]
    end = s2 - base
    danger = sum(((paths[:, :end] >= k).sum(axis=1)).mean() for k in (C.K_THRESHOLD, C.K_SEVERE, C.K_CRITICAL))
    f = build_features(t_now, np.array([i]), np.array([s1]), ehr, twin, record, n_samples=n, twin_kw=kw)
    p1 = float(risk_model.predict_proba(f)[0])
    seg2 = paths[:, s1 - base + 4:end]
    p2 = float((seg2.max(axis=1) >= C.K_THRESHOLD).mean()) if seg2.size else 0.0
    return {"s1": s1, "s2": s2, "paths": paths, "p1": p1, "p2": p2, "danger": float(danger), "features": f}


def k_chart(i: int, sc: dict, base_sc: dict | None, show_truth: bool) -> go.Figure:
    fig = go.Figure()
    lo_h = t_now - 6 * 24
    hours = np.arange(lo_h, t_now + 1)
    xs = lambda hh: [(h - t_now) / 24 for h in hh]  # noqa: E731  days relative to now
    m = twin.hist_mean[i, lo_h:t_now + 1]
    sd = twin.hist_sd[i, lo_h:t_now + 1]
    for s, dur in record.sessions[i]:
        if s + dur >= lo_h:
            fig.add_vrect(x0=(s - t_now) / 24, x1=(s + dur - t_now) / 24, fillcolor="#99F6E4", opacity=0.35, line_width=0)
    for key, colr in (("s1", TEAL), ("s2", TEAL)):
        s = sc[key]
        fig.add_vrect(x0=(s - t_now) / 24, x1=(s + 4 - t_now) / 24, fillcolor=colr, opacity=0.12, line_width=0)
    fig.add_hrect(y0=C.K_THRESHOLD, y1=10, fillcolor="#FEE2E2", opacity=0.35, line_width=0)
    fig.add_trace(go.Scatter(x=xs(hours), y=m + 2 * sd, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs(hours), y=m - 2 * sd, fill="tonexty", fillcolor="rgba(15,118,110,0.15)",
                             line=dict(width=0), name="Twin ±2 sd", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs(hours), y=m, line=dict(color=TEAL, width=2.5), name="Twin estimate"))
    pts = [(h, k, s) for (ii, h, k, s) in twin.ecg_points if ii == i and h >= lo_h]
    if pts:
        fig.add_trace(go.Scatter(x=xs([p[0] for p in pts]), y=[p[1] for p in pts], mode="markers",
                                 marker=dict(color="#7C3AED", size=7, symbol="circle-open"),
                                 error_y=dict(array=[p[2] for p in pts], thickness=1, color="#C4B5FD"),
                                 name="Smartwatch ECG estimate"))
    labs = [(h, v) for h, v in record.labs[i] if h >= lo_h]
    if labs:
        fig.add_trace(go.Scatter(x=xs([h for h, _ in labs]), y=[v for _, v in labs], mode="markers",
                                 marker=dict(color=INK, size=10, symbol="diamond"), name="Lab potassium"))
    if show_truth:
        fig.add_trace(go.Scatter(x=xs(hours), y=snap["true_K"][i, lo_h:t_now + 1], line=dict(color=MUTED, dash="dot"),
                                 name="Simulated truth (demo only)"))

    def fan(s, colr, name, dash=None):
        p = s["paths"]
        fh = np.arange(t_now + 1, t_now + 1 + p.shape[1])
        q10, q50, q90 = np.percentile(p, [10, 50, 90], axis=0)
        rgba = "rgba(15,118,110,0.18)" if colr == TEAL else "rgba(120,113,108,0.15)"
        fig.add_trace(go.Scatter(x=xs(fh), y=q90, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xs(fh), y=q10, fill="tonexty", fillcolor=rgba, line=dict(width=0),
                                 name=f"{name}: 80% band", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xs(fh), y=q50, line=dict(color=colr, width=2, dash=dash), name=f"{name}: median"))

    if base_sc is not None:
        fan(base_sc, MUTED, "Current plan", dash="dash")
    fan(sc, TEAL, "Scenario" if base_sc is not None else "Forecast")
    fig.add_hline(y=C.K_THRESHOLD, line=dict(color=RED, width=1, dash="dash"),
                  annotation_text="6.0 danger", annotation_position="top left")
    fig.add_vline(x=0, line=dict(color=INK, width=1), annotation_text="now", annotation_position="top")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white",
                      yaxis=dict(title="Serum potassium (mEq/L)", range=[2.5, max(8.0, float(np.nanmax(m)) + 0.5)],
                                 gridcolor="#F0EFE9"),
                      xaxis=dict(title="Days from now", gridcolor="#F0EFE9"),
                      legend=dict(orientation="h", y=-0.22), hovermode="x unified")
    return fig


def ecg_chart(i: int) -> go.Figure | None:
    recs = [e for e in snap["ecg_log"] if e["idx"] == i and "params" in e]
    post = [e for e in recs if e["post_dialysis"]]
    now = [e for e in recs if not e["post_dialysis"]]
    if not post or not now:
        return None
    b, l = post[-3], now[-1]          # a recent personal post-dialysis reference and tonight's beat
    pb = {k: np.array([v]) for k, v in b["params"].items()}
    pl = {k: np.array([v]) for k, v in l["params"].items()}
    yb, yl = synth_beats(pb)[0], synth_beats(pl)[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ECG_T * 1000, y=yb, line=dict(color=MUTED, width=2),
                             name=f"Personal baseline (after dialysis, {when(b['hour'])})"))
    fig.add_trace(go.Scatter(x=ECG_T * 1000, y=yl, line=dict(color="#7C3AED", width=2.5),
                             name=f"Latest ({when(l['hour'])})"))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
                      xaxis=dict(title="ms from R peak", gridcolor="#FCE7F3", dtick=100),
                      yaxis=dict(title="mV", gridcolor="#FCE7F3"), legend=dict(orientation="h", y=-0.35))
    return fig


def why_chart(i: int, f: pd.DataFrame) -> go.Figure:
    sh = risk_model.explain(f).iloc[0]
    top = sh.reindex(sh.abs().sort_values(ascending=False).index[:7])[::-1]
    labels = [FEATURE_LABELS.get(k, k.replace("_", " ")) for k in top.index]
    vals = [f.iloc[0][k] for k in top.index]
    txt = [f"{v:.2f}" if isinstance(v, (float, np.floating)) and not np.isnan(v) else "—" for v in vals]
    fig = go.Figure(go.Bar(x=top.values, y=labels, orientation="h", text=txt, textposition="auto",
                           marker_color=[RED if v > 0 else GREEN for v in top.values]))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
                      xaxis=dict(title="← lowers risk · raises risk →", gridcolor="#F0EFE9"))
    return fig


def patient_view() -> None:
    order = feats["risk"].sort_values(ascending=False).index.tolist()
    labels = {i: f"{ehr.at[i, 'name']} — {feats.at[i, 'risk']:.0%}" for i in order}
    default = order.index(snap["story_idx"]) if snap["story_idx"] in order else 0
    i = st.selectbox("Patient", order, index=default, format_func=lambda k: labels[k])
    e = ehr.loc[i]
    tags = [f"{e['age']} y · {e['sex']}", f"Dry weight {e['dry_weight_kg']:.0f} kg", e["cause"],
            f"{e['pattern']} ({e['hd_freq']}×/week)", f"Vintage {e['vintage_months']} months",
            f"Urine {e['urine_ml_day']:.0f} ml/day"]
    if e["diabetes"]:
        tags.append("Diabetes")
    if e["ace_arb"]:
        tags.append("ACE-i / ARB")
    if e["beta_blocker"]:
        tags.append("β-blocker")
    st.markdown(f"### {e['name']}")
    st.markdown(" ".join(f'<span class="ant-pill" style="background:#F5F5F4;color:#44403C">{x}</span>' for x in tags),
                unsafe_allow_html=True)

    opts = session_options(i)
    with st.expander("What-if: try a different plan", expanded=True):
        a, b, c, d = st.columns([1.4, 1, 1, 1])
        choice = a.selectbox("Next session", list(opts), index=0)
        coconut = b.slider("Tender coconut waters tomorrow", 0, 3, 0)
        binder = c.toggle("Potassium binder", help="Assumed to remove ~15 mEq of potassium per day in the gut. "
                                                    "An assumption for exploration, not a dosing recommendation.")
        diet = d.toggle("Dietitian call", help="Assumes potassium-rich foods are cut by a quarter for 3 days.")
    s1 = opts[choice]
    usual_s1 = int(snap["next_planned_start"][i])
    base_sc = scenario(i, usual_s1, 0, False, False)
    changed = (s1 != usual_s1) or coconut or binder or diet
    sc = scenario(i, s1, coconut, binder, diet) if changed else base_sc

    k1, k2, k3, k4 = st.columns(4)
    lab, cls = band(sc["p1"])
    k1.metric(f"Risk of K ≥ 6.0 before {when(sc['s1'])}", f"{sc['p1']:.0%}",
              f"{(sc['p1'] - base_sc['p1']) * 100:+.0f} pts vs plan" if changed else lab, delta_color="inverse")
    k2.metric(f"Risk in the following gap (to {when(sc['s2'])})", f"{sc['p2']:.0%}",
              f"{(sc['p2'] - base_sc['p2']) * 100:+.0f} pts" if changed else None, delta_color="inverse")
    k3.metric("Expected danger-hours, both gaps", f"{sc['danger']:.0f}",
              f"{sc['danger'] - base_sc['danger']:+.0f}" if changed else None, delta_color="inverse")
    k4.metric("Twin potassium now", f"{twin.x[i, 0]:.2f} ± {np.sqrt(twin.P[i, 0, 0]):.2f}",
              f"drift {twin.x[i, 1] * 24:+.2f}/day", delta_color="off")

    show_truth = st.toggle("Show simulated ground truth (only possible in a simulation)", value=False)
    st.plotly_chart(k_chart(i, sc, base_sc if changed else None, show_truth), width="stretch")
    st.caption("Shaded teal bands: dialysis sessions (past and planned). Purple circles: potassium estimated from "
               "the smartwatch ECG, with its uncertainty. Diamonds: lab results. The twin blends all of them.")

    l, r = st.columns(2)
    with l:
        st.markdown("**Why this risk** · contributions to the risk model's score")
        st.plotly_chart(why_chart(i, sc["features"]), width="stretch")
    with r:
        st.markdown("**Smartwatch ECG** · tonight vs her own post-dialysis beat")
        fig = ecg_chart(i)
        if fig is None:
            st.info("Not enough ECG recordings yet.")
        else:
            st.plotly_chart(fig, width="stretch")
            le = twin.last_ecg.get(i)
            if le:
                st.caption(f"T-wave height {np.exp(le['d_log_t_amp']):.1f}× baseline · T-wave width "
                           f"{le['d_t_fwhm']:+.0f} ms · PR {le['d_pr_ms']:+.0f} ms · QRS {le['d_qrs_ms']:+.0f} ms. "
                           f"ECG potassium estimate {le['k_hat']:.1f} ± {le['sd']:.1f}.")

    l, r = st.columns(2)
    with l:
        st.markdown("**Weight (smart scale)**")
        w = [(h, kg) for h, kg in record.weights[i] if h >= t_now - 14 * 24]
        if w:
            fig = go.Figure(go.Scatter(x=[(h - t_now) / 24 for h, _ in w], y=[kg for _, kg in w], mode="lines+markers",
                                       line=dict(color=TEAL)))
            fig.add_hline(y=e["dry_weight_kg"], line=dict(color=MUTED, dash="dot"), annotation_text="dry weight")
            fig.update_layout(height=230, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
                              xaxis=dict(title="Days from now"), yaxis=dict(title="kg"))
            st.plotly_chart(fig, width="stretch")
    with r:
        st.markdown("**Food diary · potassium-rich items (last 7 days)**")
        diet_rows = [(when(h), item, f"≈{meq * C.MG_PER_MEQ_K:.0f} mg") for h, item, meq in record.diet[i] if h >= t_now - 7 * 24]
        if diet_rows:
            st.dataframe(pd.DataFrame(diet_rows[::-1], columns=["When", "Item", "Potassium"]), hide_index=True,
                         width="stretch", height=230)
        else:
            st.caption("Nothing logged.")


# ============================================================================ EVIDENCE
def evidence_view() -> None:
    st.markdown("### Does it work? Evidence from the synthetic world")
    st.caption("All numbers come from the scripts in this repository and are regenerated by `make all`. "
               "They show the method works on realistic synthetic data; they are not clinical validation.")
    mm = reports["model_metrics"]
    ab = reports["ablation_csv"]
    if mm and ab is not None:
        fin = mm["final"]
        c = st.columns(4)
        c[0].metric("AUROC (unseen patients)", f"{fin['auroc']:.3f}", f"95% CI {fin['auroc_95ci'][0]:.3f}–{fin['auroc_95ci'][1]:.3f}", delta_color="off")
        lab = ab[ab.model.str.startswith("Last monthly")].iloc[0]
        c[1].metric("Events caught at 90% specificity", f"{fin['sensitivity_at_90_specificity']:.0%}",
                    f"{(fin['sensitivity_at_90_specificity'] - lab['sensitivity_at_90_specificity']) * 100:+.0f} pts vs last lab")
        c[2].metric("Week-to-week AUROC within a patient", f"{fin['within_patient_auroc']:.2f}",
                    f"last lab: {lab['within_patient_auroc']:.2f}", delta_color="off")
        c[3].metric("Test set", f"{mm['test_patients']} patients", f"{mm['test_decisions']:,} nights", delta_color="off")
        st.markdown("#### Each data stream alone vs fused")
        show = ab[["model", "auroc", "auprc", "sensitivity_at_90_specificity", "within_patient_auroc", "brier"]].copy()
        show.columns = ["Approach", "AUROC", "AUPRC", "Sensitivity @ 90% spec.", "Within-patient AUROC", "Brier"]
        st.dataframe(show.round(3).astype(object).where(show.notna(), "—"), hide_index=True, width="stretch")
        if reports["roc"]:
            fig = go.Figure()
            pick = ["Last monthly potassium lab (today's practice)", "Physics-only twin (EHR, no wearables)",
                    "Antaral twin forecast (EHR + wearables, Kalman fusion)", "Antaral twin + risk model (final)"]
            colors = [MUTED, AMBER, "#7C3AED", TEAL]
            for name, colr in zip(pick, colors):
                if name in reports["roc"]:
                    rc = reports["roc"][name]
                    fig.add_trace(go.Scatter(x=rc["fpr"], y=rc["tpr"], name=name, line=dict(color=colr, width=2.5)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color="#D6D3D1", dash="dot"), showlegend=False))
            fig.update_layout(height=380, plot_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10),
                              xaxis=dict(title="False alarm rate"), yaxis=dict(title="Events caught"),
                              legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig, width="stretch")
    us = reports["unit_simulation_csv"]
    if us is not None:
        st.markdown("#### Re-timing sessions in a 60-patient unit (8 weeks × replicate units)")
        metrics = {"pre_dialysis_k_ge_6": "Patients arriving with K ≥ 6.0", "hyperk_episodes": "Episodes of K ≥ 6.0",
                   "patient_hours_k_ge_6_5": "Hours at K ≥ 6.5", "patient_hours_k_ge_7": "Hours at K ≥ 7.0"}
        g = us.groupby("policy")[list(metrics) + ["sessions", "reschedules_per_week"]].mean()
        fixed = g.loc["Fixed weekly pattern"]
        fig = go.Figure()
        for pol, colr in zip([p for p in g.index if p != "Fixed weekly pattern"], [AMBER, TEAL]):
            red = [100 * (1 - g.loc[pol, m] / fixed[m]) for m in metrics]
            fig.add_trace(go.Bar(x=list(metrics.values()), y=red, name=pol, marker_color=colr,
                                 text=[f"{v:.0f}%" for v in red], textposition="outside"))
        fig.update_layout(height=340, plot_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10), barmode="group",
                          yaxis=dict(title="Reduction vs fixed schedule (%)"), legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Same patients, same meals and illnesses, same number of sessions per patient "
                   f"({fixed['sessions']:.0f} sessions per unit in both arms). Reschedules per week: "
                   + ", ".join(f"{p}: {g.loc[p, 'reschedules_per_week']:.0f}" for p in g.index if p != "Fixed weekly pattern"))
    ee = reports["ecg_estimator"]
    if ee:
        st.markdown("#### Smartwatch ECG as a potassium sensor")
        st.write(f"Potassium estimated from a single-lead median beat: mean absolute error **{ee['mae']:.2f} mEq/L**, "
                 f"correlation **{ee['corr']:.2f}** on {ee['val_patients']} held-out calibration patients. "
                 "Deliberately noisy: the twin treats it as a weak sensor and trusts labs more.")


def about_view() -> None:
    st.markdown("""
### Antaral · अंतराल · "the interval"
**The problem.** Many dialysis patients in India get two sessions a week instead of three because of cost
and machine shortages. Between sessions, potassium builds up silently. Too much potassium can stop the heart
without warning, and sudden cardiac death after the long gap between sessions is a well-documented pattern in
haemodialysis.

**What Antaral does.**
1. **Patient twin** — a mechanistic potassium model personalised from the EHR, continuously corrected by a
   Kalman filter using monthly labs and potassium estimated from a smartwatch ECG.
2. **Risk layer** — a calibrated LightGBM model on top of the twin, with per-patient explanations.
3. **Unit twin** — every evening, an integer programme re-plans tomorrow's chairs so the patients heading
   for danger are dialysed first, without adding sessions.

**What it is not.** It is not a medical device and does not tell patients what to do. Every suggestion goes to
a clinician for approval. All data in this demo is synthetic.
""")


{"Unit tonight": unit_view, "Patient twin": patient_view, "Evidence": evidence_view, "About": about_view}[page]()
