"""Signal layer: one-shot feature computation over a bar-dict.

Takes the output of `build()` (scripts/goog_minute_regime.py:63) or `build_snaps()`
(scripts/adaptive_wh_long.py:41) -- both return the identical bar-dict schema
{bb, ba, flow, ts, mid, spread, day_ix} -- and produces every raw/processed signal
downstream layers need, computed once and read from one place (`build_ctx`).

Every signal here is CAUSAL (computed only from data at-or-before the bar it's
attached to) -- none of this file makes train/test decisions itself; that's the
caller's job (thresholds are frozen by the tool/regime layers, not here).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_lib"))
from flow import zflow
from kinematics import poly_derivs

from l2_strategy import variance_ratio, autocorr1

DEFAULT_FLOW_LW = 5
DEFAULT_KIN_W = 40
DEFAULT_VR_W, DEFAULT_VR_K = 100, 8
DEFAULT_AC_W = 100
DEFAULT_VOL_FAST, DEFAULT_VOL_SLOW = 30, 300
DEFAULT_SPREAD_W, DEFAULT_SPREAD_WARM = 2000, 300


def flow_zscore(bars: dict, lw: int = DEFAULT_FLOW_LW) -> np.ndarray:
    """Thin wrapper over flow.zflow -- kept as a named signal for build_ctx/registry use."""
    return zflow(bars["flow"], len(bars["mid"]), lw)


def kinematic_z(bars: dict, W: int = DEFAULT_KIN_W):
    """Thin wrapper over kinematics.poly_derivs -- returns (z1, z2)."""
    return poly_derivs(bars["mid"], W)


def variance_ratio_series(mid: np.ndarray, W: int = DEFAULT_VR_W,
                          vr_k: int = DEFAULT_VR_K) -> np.ndarray:
    """Rolling variance-ratio, exposed standalone (l2_strategy.compute_regime only
    ever computes this once per bar inside its own loop -- not reusable there)."""
    r = np.diff(mid, prepend=mid[0])
    N = len(mid)
    out = np.ones(N)
    for t in range(W, N):
        out[t] = variance_ratio(r[t - W:t], vr_k)
    return out


def autocorr_series(x: np.ndarray, W: int = DEFAULT_AC_W) -> np.ndarray:
    """Rolling lag-1 autocorrelation of `x` (e.g. mid returns, or signed flow, for
    the flow-persistence check compute_regime does internally)."""
    N = len(x)
    out = np.zeros(N)
    for t in range(W, N):
        out[t] = autocorr1(x[t - W:t])
    return out


def _roll_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Causal rolling mean, current-bar-inclusive (matches l2_hmm.py's private
    helper of the same name/behavior: out[i] = mean(x[i-w+1 : i+1]))."""
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.full(len(x), np.nan)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def vol_zscore(mid: np.ndarray, fast: int = DEFAULT_VOL_FAST,
              slow: int = DEFAULT_VOL_SLOW) -> np.ndarray:
    """Log ratio of fast/slow realized vol -- the vol-regime auxiliary axis.

    Extracted from the vf/vs/lvr block inside l2_hmm.py:31 features() (that
    function is left untouched; Phase 2's features3 will call THIS instead of
    reimplementing it a third time). NaN (pre-warmup) -> 0.0.
    """
    r = np.diff(mid, prepend=mid[0])
    vf = np.sqrt(_roll_mean(r * r, fast))
    vs = np.sqrt(_roll_mean(r * r, slow))
    lvr = np.log((vf + 1e-12) / (vs + 1e-12))
    return np.nan_to_num(lvr, nan=0.0)


def spread_zscore(spread: np.ndarray, W: int = DEFAULT_SPREAD_W,
                  warm: int = DEFAULT_SPREAD_WARM) -> np.ndarray:
    """Rolling z-score of spread vs its own trailing distribution -- the
    liquidity-regime auxiliary axis. Genuinely new (not extracted from anywhere).
    Causal, current-bar-inclusive rolling window (same convention as _roll_mean).
    Before `warm` bars, returns 0.0 (neutral).
    """
    mu = _roll_mean(spread, W)
    var = _roll_mean(spread * spread, W) - mu ** 2
    sd = np.sqrt(np.maximum(var, 1e-18))
    z = (spread - mu) / (sd + 1e-9)
    z = np.nan_to_num(z, nan=0.0)
    z[:warm] = 0.0
    return z


def build_ctx(bars: dict, wall: int) -> dict:
    """One-shot feature computation. Returns passthrough bar-dict fields plus every
    processed signal below, all (N,) arrays aligned to `bars["mid"]`/`bars["ts"]`.
    `wall` is stored for downstream convenience (train/test split index) but this
    function makes no train/test decisions itself -- every signal here is computed
    over the FULL series, causally, the same way regardless of where the wall is.
    """
    mid, spread = bars["mid"], bars["spread"]
    z1, z2 = kinematic_z(bars)
    ctx = dict(
        bb=bars["bb"], ba=bars["ba"], mid=mid, spread=spread, flow=bars["flow"],
        ts=bars["ts"], day_ix=bars["day_ix"], wall=wall,
        flow_z=flow_zscore(bars),
        kin_z1=z1, kin_z2=z2,
        vr=variance_ratio_series(mid),
        ac=autocorr_series(np.diff(mid, prepend=mid[0])),
        vol_z=vol_zscore(mid),
        spread_z=spread_zscore(spread),
    )
    return ctx
