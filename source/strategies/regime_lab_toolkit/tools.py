"""Minimal toolkit: one SIGNAL tool (votes on direction) + one MODIFIER tool
(scales an externally-supplied direction's confidence, never produces its own).

flow_momentum is the one validated offensive signal in this program (GOOG,
+864bps/106tr/t+1.53 frozen test -- see regime_lab/phase0_regression_check.py).
kinematic_modifier repurposes the polynomial-derivative math that has been
proven DEAD as a standalone forecaster (RMSE ratio 1.68 vs no-change, corr
-0.05, scripts/goog_prediction_viz.py) into a narrower, untested claim: does
kinematic shape agreement/disagreement usefully scale someone else's direction?
See phase3_kinematic_modifier_test.py -- this may legitimately fail.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flow import zflow, frozen_flow_threshold
from kinematics import poly_derivs, kinematic_state
from thresholds import dynamic_quantile_thr

DEFAULT_KIN_W = 40
KIN_WARMUP = 150   # matches scripts/goog_minute_regime.py's WARMB convention


def flow_momentum(ctx: dict, lw: int = 5, q: float = 0.95, hold: int = 5,
                  wall: int = None) -> dict:
    """SIGNAL tool. direction (N,) int{-1,0,1}, raw_size (N,) float (constant 1.0
    where firing, matching the proven binary-entry spec), confidence (N,) float
    (|z|/thr, how far past the gate). If `wall` is given, freezes the threshold
    on z[420:wall] q=0.95 EXACTLY as the proven spec (curated_nfty_goog.py:140);
    if None, falls back to a walk-forward dynamic threshold (asset-agnostic
    default for future multi-symbol use, not the reproduced-number path).
    """
    N = len(ctx["mid"])
    z = zflow(ctx["flow"], N, lw)
    if wall is not None:
        thr = frozen_flow_threshold(z, 420, wall, q)
        thr_arr = np.full(N, thr)
    else:
        thr_arr = dynamic_quantile_thr(z, q=q)

    fires = np.abs(z) > thr_arr
    direction = np.where(fires, np.sign(z), 0).astype(int)
    raw_size = np.where(fires, 1.0, 0.0)
    confidence = np.clip(np.abs(z) / np.maximum(thr_arr, 1e-9), 0, 5)
    return dict(direction=direction, raw_size=raw_size, confidence=confidence,
                hold=hold, category="directional", kind="signal")


def kinematic_modifier(ctx: dict, direction: np.ndarray, W: int = DEFAULT_KIN_W,
                       q1_frac: float = 0.70, q2c_frac: float = 0.60,
                       q2p_frac: float = 0.85, wall: int = None,
                       warmup: int = KIN_WARMUP) -> np.ndarray:
    """MODIFIER tool. Takes an EXTERNALLY-supplied `direction` (e.g. from
    flow_momentum) and returns a multiplicative size_mult (N,). Never decides
    direction itself -- that use of this math is the one proven dead in this
    repo. `agree_ext[t] = sign(z1[t]) == direction[t]` (does the kinematic
    shape's own implied direction match the externally-supplied one?):
      agree_ext & accel (self-referential z1/z2 acceleration, matching)  -> 2.0
      ~agree_ext & peak  (self-referential z1/z2 opposition, reversal warn) -> 0.0
      else                                                                -> 1.0 (neutral)

    q1_frac/q2c_frac/q2p_frac are QUANTILE FRACTIONS (0-1), converted to concrete
    |z1|/|z2| thresholds via a train-slice quantile (frozen if `wall` given, else
    computed over the whole series -- exploratory/non-walled fallback).
    """
    mid = ctx["mid"]
    z1, z2 = poly_derivs(mid, W)
    lo = max(W, warmup)
    hi = wall if wall is not None else len(mid)
    q1 = float(np.quantile(np.abs(z1[lo:hi]), q1_frac))
    q2c = float(np.quantile(np.abs(z2[lo:hi]), q2c_frac))
    q2p = float(np.quantile(np.abs(z2[lo:hi]), q2p_frac))

    state = kinematic_state(z1, z2, q1, q2c, q2p)
    agree_ext = np.sign(z1) == direction
    boost = agree_ext & state["accel"]
    shrink = (~agree_ext) & state["peak"]
    mult = np.where(boost, 2.0, np.where(shrink, 0.0, 1.0))
    return mult


TOOLS = {
    "flow_momentum": dict(kind="signal", category="directional", fn=flow_momentum),
}
MODIFIERS = {
    "kinematic_ladder": dict(kind="modifier", category="directional", fn=kinematic_modifier),
}
