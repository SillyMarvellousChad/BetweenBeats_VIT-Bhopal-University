"""Smartwatch single-lead ECG: synthesis and feature extraction.

Synthesis
    Each recording is summarised as a *median beat* (what consumer ECG apps compute by
    averaging the beats in a 30 s strip). The beat is a sum of Gaussian waves (P, Q, R,
    S, T), in the spirit of the ECGSYN model, whose shape responds to serum potassium
    the way the clinical literature describes:

    * T wave grows taller, narrower and more symmetric ("peaked T")
    * QT shortens slightly
    * above ~5.5 mEq/L the P wave flattens and the PR interval lengthens
    * above ~6.0 mEq/L the QRS widens

    How strongly a patient's T wave responds (``ecg_gamma``) and their baseline beat
    shape differ from person to person, which is why Antaral personalises: it compares
    each new beat with the patient's own post-dialysis beat.

Extraction
    Features are measured from the waveform itself (peak finding, widths, slopes), so
    the pipeline would run unchanged on a real exported median beat.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FS = 250                                   # Hz
T = np.arange(-0.40, 0.60, 1.0 / FS)       # seconds, R peak at 0; 250 samples

RAW_FEATURES = ["r_amp", "t_amp", "t_r_ratio", "t_fwhm", "t_upslope", "t_downslope",
                "t_symmetry", "qrs_ms", "pr_ms", "p_amp", "qt_ms"]
DELTA_FEATURES = ["d_log_t_amp", "d_log_t_r", "d_t_fwhm", "d_log_downslope",
                  "d_qrs_ms", "d_pr_ms", "d_log_p_amp", "d_qt_ms"]


def beat_params(k: np.ndarray, r_amp0, t_amp0, t_sigma0, gamma, rng: np.random.Generator) -> dict:
    """Wave parameters for median beats recorded at true potassium ``k`` (arrays)."""
    k = np.asarray(k, dtype=float)
    n = k.shape[0]
    dk = k - 4.5
    ex55 = np.maximum(0.0, k - 5.5)
    ex60 = np.maximum(0.0, k - 6.0)
    ln = lambda s: rng.lognormal(0.0, s, n)  # noqa: E731  day-to-day measurement variability
    return {
        "p_amp": 0.11 * np.exp(-0.55 * ex55) * ln(0.25),
        "p_t": -0.165 - 0.028 * ex55 + rng.normal(0, 0.008, n),
        "q_amp": -0.08 * r_amp0 * ln(0.15),
        "r_amp": r_amp0 * ln(0.12),
        "s_amp": -0.22 * r_amp0 * ln(0.15),
        "qrs_sigma": 0.0105 * (1 + 0.22 * ex60 ** 1.2) * ln(0.09),
        "t_amp": t_amp0 * np.exp(gamma * dk) * ln(0.15),
        "t_sigma": t_sigma0 * np.clip(1 - 0.075 * dk, 0.5, 1.3) * ln(0.11),
        "t_center": 0.275 - 0.012 * dk + rng.normal(0, 0.014, n),
        "wander_amp": rng.uniform(0.0, 0.035, n),
        "wander_phase": rng.uniform(0, 2 * np.pi, n),
        "noise_seed": rng.integers(0, 2**31 - 1, n),
    }


def synth_beats(p: dict) -> np.ndarray:
    """Render median beats (n x len(T), millivolts) from ``beat_params`` output."""
    def col(name):
        return np.asarray(p[name], dtype=float)[:, None]

    t = T[None, :]
    g = lambda amp, mu, sd: amp * np.exp(-0.5 * ((t - mu) / sd) ** 2)  # noqa: E731
    qs = col("qrs_sigma")
    beat = (g(col("p_amp"), col("p_t"), 0.022)
            + g(col("q_amp"), -0.028 * qs / 0.0105, qs * 0.9)
            + g(col("r_amp"), 0.0, qs)
            + g(col("s_amp"), 0.030 * qs / 0.0105, qs)
            + g(col("t_amp"), col("t_center"), col("t_sigma")))
    beat += col("wander_amp") * np.sin(2 * np.pi * 0.35 * t + col("wander_phase"))
    seeds = np.asarray(p["noise_seed"])
    noise = np.stack([np.random.default_rng(int(s)).normal(0, 0.02, T.size) for s in seeds])
    return beat + noise


def _smooth(x: np.ndarray, w: int = 5) -> np.ndarray:
    kernel = np.ones(w) / w
    return np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="same"), 1, x)


def _idx(t0: float, t1: float) -> slice:
    return slice(int(np.searchsorted(T, t0)), int(np.searchsorted(T, t1)))


def extract_features(beats: np.ndarray) -> pd.DataFrame:
    """Measure potassium-sensitive features from median beats (n x len(T))."""
    beats = np.atleast_2d(beats)
    n = beats.shape[0]
    x = _smooth(beats)
    # isoelectric reference: PR segment just before the QRS (robust to baseline wander)
    x = x - np.median(x[:, _idx(-0.075, -0.045)], axis=1, keepdims=True)
    dt = 1.0 / FS
    rows = np.arange(n)

    r_win = _idx(-0.04, 0.04)
    r_i = r_win.start + np.argmax(x[:, r_win], axis=1)
    r_amp = x[rows, r_i]

    t_win = _idx(0.14, 0.50)
    t_i = t_win.start + np.argmax(x[:, t_win], axis=1)
    t_amp = np.maximum(x[rows, t_i], 1e-3)

    # T full width at half maximum, walking out from the peak
    half = t_amp / 2
    left = t_i.copy()
    right = t_i.copy()
    for r in range(n):
        while left[r] > t_win.start and x[r, left[r]] > half[r]:
            left[r] -= 1
        while right[r] < T.size - 1 and x[r, right[r]] > half[r]:
            right[r] += 1
    t_fwhm = (right - left) * dt

    d = np.gradient(x, dt, axis=1)
    up = np.array([d[r, max(t_i[r] - 30, 0):t_i[r] + 1].max() for r in range(n)])
    down = np.array([-d[r, t_i[r]:min(t_i[r] + 30, T.size)].min() for r in range(n)])
    t_up_ms = (t_i - left) * dt
    t_down_ms = (right - t_i) * dt
    symmetry = t_up_ms / np.maximum(t_down_ms, dt)

    # QRS: span around R where the signal deviates beyond 12% of R amplitude
    qrs_ms = np.empty(n)
    q_win = _idx(-0.09, 0.11)
    seg = np.abs(x[:, q_win])
    for r in range(n):
        above = np.flatnonzero(seg[r] > 0.12 * abs(r_amp[r]))
        qrs_ms[r] = (above[-1] - above[0] + 1) * dt * 1000 if above.size else np.nan

    p_win = _idx(-0.32, -0.09)
    p_i = p_win.start + np.argmax(x[:, p_win], axis=1)
    p_amp = np.maximum(x[rows, p_i], 1e-3)
    pr_ms = (T[r_i] - T[p_i]) * 1000

    # QT proxy: QRS onset to the T-wave end (where it falls below 10% of its peak)
    t_end = t_i.copy()
    for r in range(n):
        while t_end[r] < T.size - 1 and x[r, t_end[r]] > 0.1 * t_amp[r]:
            t_end[r] += 1
    qt_ms = (T[t_end] - (T[r_i] - qrs_ms / 2000)) * 1000

    return pd.DataFrame({
        "r_amp": r_amp, "t_amp": t_amp, "t_r_ratio": t_amp / np.maximum(r_amp, 0.05),
        "t_fwhm": t_fwhm * 1000, "t_upslope": up, "t_downslope": down,
        "t_symmetry": symmetry, "qrs_ms": qrs_ms, "pr_ms": pr_ms, "p_amp": p_amp,
        "qt_ms": qt_ms,
    })


def delta_features(raw: pd.DataFrame, baseline: pd.DataFrame | None) -> pd.DataFrame:
    """Change of each feature relative to the patient's own post-dialysis beat."""
    out = pd.DataFrame(index=raw.index)
    if baseline is None:
        for c in DELTA_FEATURES:
            out[c] = np.nan
        return out
    b = baseline.reindex(raw.index) if isinstance(baseline, pd.DataFrame) and len(baseline) == len(raw) else baseline
    out["d_log_t_amp"] = np.log(raw["t_amp"].values / b["t_amp"].values)
    out["d_log_t_r"] = np.log(raw["t_r_ratio"].values / b["t_r_ratio"].values)
    out["d_t_fwhm"] = raw["t_fwhm"].values - b["t_fwhm"].values
    out["d_log_downslope"] = np.log(np.maximum(raw["t_downslope"].values, 1e-3)
                                    / np.maximum(b["t_downslope"].values, 1e-3))
    out["d_qrs_ms"] = raw["qrs_ms"].values - b["qrs_ms"].values
    out["d_pr_ms"] = raw["pr_ms"].values - b["pr_ms"].values
    out["d_log_p_amp"] = np.log(raw["p_amp"].values / b["p_amp"].values)
    out["d_qt_ms"] = raw["qt_ms"].values - b["qt_ms"].values
    return out
