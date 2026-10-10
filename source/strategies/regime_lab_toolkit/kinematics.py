"""Kinematic (velocity/acceleration) shape signals from a weighted polynomial fit.

`poly_derivs` reuses `_proj` from scripts/l2_strategy.py:55 (the exponentially-weighted
degree-d polynomial projection matrix). Same math as scripts/l2_mmdir.py:53 and the
duplicate in scripts/goog_minute_regime.py:78 -- consolidated here as the single source
of truth. Those files are left untouched (regression-safety).

`kinematic_state` consolidates the two existing z1/z2 -> decision mappings:
  * scripts/l2_mmdir.py:66 dir_target()      -- constant thresholds (Z1_MIN etc.)
  * scripts/goog_minute_regime.py:90 targets() -- quantile thresholds, tunable

...to the quantile-threshold style (matches this project's "nothing hand-tuned"
principle: thresholds are computed by the caller from train-slice quantiles, not
hardcoded module constants). Crucially, unlike both originals, this returns a STATE
DICT describing the shape, not a signed position target -- this repo has proven
(scripts/goog_prediction_viz.py: RMSE ratio 1.68 vs no-change forecast, corr -0.05)
that using kinematic sign as a standalone direction predictor is dead. Nothing here
should ever be read as "the direction to trade" -- see tools.kinematic_modifier for
the only sanctioned use (scaling an externally-supplied direction's confidence).
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_lib"))
from l2_strategy import _proj

DEFAULT_VOL_WIN = 100


def poly_derivs(mid: np.ndarray, W: int, vol_win: int = DEFAULT_VOL_WIN):
    """Vol-normalized velocity (z1) and acceleration (z2) z-scores, each (N,).

    Fits a weighted degree-2 polynomial over each trailing W-bar window (weight
    decay lam=W/2, same convention as every other caller of _proj in this repo).
    First W-1 entries are 0 (post nan_to_num) -- not real signal, warmup.
    """
    N = len(mid)
    C = swv(mid, W) @ _proj(W, 2, W / 2).T
    d1 = np.full(N, np.nan)
    d2 = np.full(N, np.nan)
    d1[W - 1:] = C[:, 1]
    d2[W - 1:] = 2 * C[:, 2]
    r = np.diff(mid, prepend=mid[0])
    var = np.convolve(r * r, np.ones(vol_win) / vol_win, "full")[:N]
    vol = np.sqrt(np.maximum(var, 1e-18))
    z1 = np.nan_to_num(d1 / (vol + 1e-12))
    z2 = np.nan_to_num(d2 / (vol / np.sqrt(W) + 1e-12))
    return z1, z2


def poly_predict(mid: np.ndarray, W: int, H: int, O: int = 2) -> np.ndarray:
    """Predicted mid-price LEVEL at t+H, from a weighted degree-O polynomial
    fit over the trailing W bars extrapolated to tau=H. Same fit machinery as
    poly_derivs (same _proj convention). Returns (N,) with NaN before warmup
    (t < W-1); caller compares predicted[t] against the REALIZED mid[t+H].

    This is the actual kinematic FORECAST (not just derivatives) -- the
    literal thing scripts/goog_prediction_viz.py measured as having negative
    skill (RMSE ratio 1.68 vs no-change) on GOOG calendar-minute bars. Re-run
    on a different clock (trade-count bars) here because that earlier result
    was specific to that clock, not necessarily universal.
    """
    N = len(mid)
    C = swv(mid, W) @ _proj(W, O, W / 2).T
    fut = np.array([float(H) ** k for k in range(O + 1)])
    predicted = np.full(N, np.nan)
    predicted[W - 1:] = C @ fut
    return predicted


def kinematic_state(z1: np.ndarray, z2: np.ndarray, q1: float, q2c: float,
                    q2p: float) -> dict:
    """Classify each bar's kinematic shape, self-referentially (z1's own sign is
    the shape's implied direction -- z2 is judged against IT, not against any
    externally supplied trade direction; that comparison is the caller's job).

    q1  : |z1| threshold for "velocity is significant at all"
    q2c : |z2| threshold (with z2 agreeing in sign with z1) for "accelerating"
    q2p : |z2| threshold (with z2 opposing z1's sign) for "peak/reversal warning"

    Returns a dict of mutually-exclusive (N,) boolean masks: accel, steady, peak,
    none -- plus the raw z1 array (callers that need the shape's own implied
    direction, e.g. to compare against an external signal, read z1 directly).
    """
    sig = np.abs(z1) > q1
    agree = np.sign(z2) == np.sign(z1)
    peak = sig & ~agree & (np.abs(z2) > q2p)
    accel = sig & agree & (np.abs(z2) > q2c) & ~peak
    steady = sig & ~accel & ~peak
    none = ~sig
    return dict(accel=accel, steady=steady, peak=peak, none=none, z1=z1, z2=z2)
