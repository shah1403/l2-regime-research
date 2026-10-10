"""Signed trade-flow z-score -- the one validated offensive signal in this program.

Extracted verbatim (same math, no behavior change) from the `zflow` closure inside
`goog_flow()` in scripts/curated_nfty_goog.py:113-120, which was re-typed a second
time in scripts/goog_flow_gated.py:40-46. This is the single source of truth going
forward; those two files are left untouched (regression-safety for their published
numbers) but should eventually import from here.

GOOG frozen result reproduced by regime_lab/phase0_regression_check.py:
  Lw=5m H=5m  -> TEST +864.3 bps, 106 trades, t=+1.53  (auto-selected on train)
  Lw=5m H=30m -> TEST +839.6 bps,  58 trades, t=+1.61  (manual second look)
"""
from __future__ import annotations

import numpy as np

VAR_WINDOW = 390   # ~1 trading day of 1-min bars, the per-bar flow-variance estimator


def zflow(flow: np.ndarray, N: int, lw: int, var_window: int = VAR_WINDOW) -> np.ndarray:
    """Z-score of the trailing `lw`-bar sum of signed flow, vs its own history.

    s[t]  = sum of `flow` over the `lw` bars ending at t (causal, excludes bar t
            itself per the original off-by-one: c[lw+1:]-c[1:N-lw+1]).
    v[t]  = rolling mean of flow^2 over `var_window` bars -- single-bar variance
            estimate, scaled by `lw` (iid-bars assumption) to variance of `s`.
    Returns 0.0 wherever v is undefined (first `var_window-1` bars).
    """
    c = np.cumsum(np.insert(flow, 0, 0.0))
    s = np.zeros(N)
    s[lw:] = c[lw + 1:] - c[1:N - lw + 1]
    c2 = np.cumsum(np.insert(flow * flow, 0, 0.0))
    v = np.full(N, np.nan)
    v[var_window - 1:] = (c2[var_window:] - c2[:-var_window]) / var_window
    sd = np.sqrt(np.maximum(v * lw, 1e-12))
    return np.where(np.isnan(v), 0.0, s / (sd + 1e-9))


def frozen_flow_threshold(z: np.ndarray, lo: int, hi: int, q: float = 0.95) -> float:
    """Train-frozen quantile threshold on |z|, computed on z[lo:hi] only.

    Matches curated_nfty_goog.py:140 exactly: thr = quantile(|z[420:wall]|, 0.95).
    Returned value is a plain float meant to be reused unchanged on the test slice.
    """
    return float(np.quantile(np.abs(z[lo:hi]), q))
