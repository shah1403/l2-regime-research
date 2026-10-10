"""Regime classifier: soft posterior over {DIR-up, DIR-down, MR, FLAT} + confidence,
plus continuous vol/liquidity auxiliary axes -- the probabilistic upgrade to the
hard VR/autocorrelation label in scripts/l2_strategy.py's compute_regime().

Built on the already-causal, already-walk-forward GaussHMM in scripts/l2_hmm.py
(fit day k / .filter() decode day k+1, never Viterbi/smoothed -- that leaks the
future). l2_hmm.py is left UNTOUCHED; its 2D features() is the Phase 2 baseline
this module's 3D extension must beat (or honestly fail to beat) in
phase2_forward_informativeness.py.

The one deliberate design constraint: GaussHMM itself is not modified. Verified by
reading l2_hmm.py directly -- `_emis` uses X.shape[1] (dimension-generic),
`fit` derives (n,d)=X.shape and allocates mu/sig as (k,d) -- so passing a 3D
observation matrix in requires zero changes to the model class. This keeps
"did adding flow help" the only variable that changed vs. the tested 2D baseline.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_lib"))
from flow import zflow
from signals import vol_zscore, spread_zscore, _roll_mean

import l2_hmm
from l2_hmm import GaussHMM, label_states, reg_hmm, DRIFT_W, VOL_FAST, VOL_SLOW, K

FLOW_LW = 5


def _drift_z(mid: np.ndarray, drift_w: int = DRIFT_W) -> np.ndarray:
    """Exact drift_z computation from l2_hmm.py:31 features(), pulled out standalone
    so features3 can reuse it without touching l2_hmm.py itself."""
    r = np.diff(mid, prepend=mid[0])
    mu = _roll_mean(r, drift_w)
    v = _roll_mean(r * r, drift_w)
    sd = np.sqrt(np.maximum(v - mu ** 2, 1e-18))
    drift_z = mu * np.sqrt(drift_w) / (sd + 1e-12)
    return np.nan_to_num(drift_z, nan=0.0)


def features3(mid: np.ndarray, flow: np.ndarray, drift_w: int = DRIFT_W,
             vol_fast: int = VOL_FAST, vol_slow: int = VOL_SLOW,
             flow_lw: int = FLOW_LW):
    """[drift_z, log_vol_ratio, flow_z] -- l2_hmm.py's 2D features() extended with
    flow as a third, continuous, magnitude-aware dimension. `drift_w`/`vol_fast`/
    `vol_slow` default to l2_hmm's own constants so the first two columns are
    IDENTICAL to the 2D baseline -- flow_z is the only new information.
    """
    drift_z = _drift_z(mid, drift_w)
    lvr = vol_zscore(mid, vol_fast, vol_slow)
    fz = zflow(flow, len(mid), flow_lw)
    r = np.diff(mid, prepend=mid[0])
    X = np.column_stack([drift_z, lvr, fz])
    X[~np.isfinite(X)] = 0.0
    return X, r


def label_states3(model, X_fit, r_fit) -> dict:
    """Thin wrapper on l2_hmm.label_states -- unchanged logic still works since
    drift_z stays column 0 (model.mu[:,0]) regardless of X's total width."""
    return label_states(model, X_fit, r_fit)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def dir_up_down_split(P: np.ndarray, dir_state: int, drift_z: np.ndarray,
                      k: float = 2.0):
    """Split the single DIR state's posterior mass into up/down by the SIGN and
    MAGNITUDE of drift_z, rather than refitting the HMM with K=4 -- a K=4 refit
    would confound two changes (more states AND flow input) into one experiment.
    """
    p_dir = P[:, dir_state]
    frac_up = _sigmoid(k * drift_z)
    return p_dir * frac_up, p_dir * (1.0 - frac_up)


def _walkforward(bars: dict, decode_fn) -> dict:
    """Shared day-pair walk-forward loop: for each consecutive day pair (k,k+1),
    fit on day k, decode day k+1 via `decode_fn(mid_fit, flow_fit, mid_decode,
    flow_decode) -> (posterior4 (m,4), hard_reg (m,), conf (m,))`. Day 0 (no prior
    day to fit on) is filled with a neutral FLAT/zero-confidence placeholder.
    """
    di = bars["day_ix"]
    N = len(bars["mid"])
    posterior = np.zeros((N, 4))
    posterior[:di[1], 2] = 1.0                      # day 0 default: FLAT (index 2)
    hard_reg = np.zeros(N, dtype=int)
    conf = np.zeros(N)
    for k in range(len(di) - 2):
        lo_fit, hi_fit = di[k], di[k + 1]
        lo_dec, hi_dec = di[k + 1], di[k + 2]
        mid_fit, flow_fit = bars["mid"][lo_fit:hi_fit], bars["flow"][lo_fit:hi_fit]
        mid_dec, flow_dec = bars["mid"][lo_dec:hi_dec], bars["flow"][lo_dec:hi_dec]
        post, hard, c = decode_fn(mid_fit, flow_fit, mid_dec, flow_dec)
        posterior[lo_dec:hi_dec] = post
        hard_reg[lo_dec:hi_dec] = hard
        conf[lo_dec:hi_dec] = c
    return dict(posterior=posterior, hard_reg=hard_reg, confidence=conf)


def walkforward_regime3(bars: dict, wall: int = None, p_min: float = 0.6,
                        flow_lw: int = FLOW_LW) -> dict:
    """Full walk-forward 3D (drift_z, log_vol_ratio, flow_z) HMM regime posterior.
    Columns of `posterior`: [DIR-up, DIR-down, MR, FLAT]. `wall` is accepted for
    interface parity with the rest of the pipeline but unused here -- fitting is
    walk-forward day-by-day regardless of where any train/test split sits.
    """
    def decode(mid_fit, flow_fit, mid_dec, flow_dec):
        Xf, rf = features3(mid_fit, flow_fit, flow_lw=flow_lw)
        Xd, _ = features3(mid_dec, flow_dec, flow_lw=flow_lw)
        m = GaussHMM(k=K).fit(Xf)
        lab = label_states3(m, Xf, rf)
        P = m.filter(Xd)
        hard = P.argmax(1)
        c = P[np.arange(len(P)), hard]
        dir_state = next(s for s, v in lab.items() if v == 1)
        mr_state = next(s for s, v in lab.items() if v == -1)
        flat_states = [s for s, v in lab.items() if v == 0]
        drift_z_dec = Xd[:, 0]
        p_up, p_down = dir_up_down_split(P, dir_state, drift_z_dec)
        p_mr = P[:, mr_state]
        p_flat = P[:, flat_states].sum(1) if flat_states else np.zeros(len(P))
        post4 = np.column_stack([p_up, p_down, p_mr, p_flat])
        reg = np.where(c < p_min, 0, np.where(hard == dir_state, 1,
                       np.where(hard == mr_state, -1, 0)))
        return post4, reg, c

    out = _walkforward(bars, decode)
    out["vol_z"] = vol_zscore(bars["mid"])
    out["spread_z"] = spread_zscore(bars["spread"])
    return out


def walkforward_regime2(bars: dict, wall: int = None, p_min: float = 0.6) -> dict:
    """Baseline comparator: pure walk-forward wrapper around UNMODIFIED
    l2_hmm.reg_hmm (2D features, no flow) -- same day-pair loop, same output shape
    (posterior column 0 = P(DIR)*sign-split, kept comparable to walkforward_regime3
    via the same dir_up_down_split so the two are apples-to-apples).
    """
    def decode(mid_fit, flow_fit, mid_dec, flow_dec):
        reg, P, lab = reg_hmm(mid_fit, mid_dec, p_min=p_min)
        dir_state = next(s for s, v in lab.items() if v == 1)
        mr_state = next(s for s, v in lab.items() if v == -1)
        flat_states = [s for s, v in lab.items() if v == 0]
        Xd, _ = l2_hmm.features(mid_dec)
        drift_z_dec = Xd[:, 0]
        p_up, p_down = dir_up_down_split(P, dir_state, drift_z_dec)
        p_mr = P[:, mr_state]
        p_flat = P[:, flat_states].sum(1) if flat_states else np.zeros(len(P))
        post4 = np.column_stack([p_up, p_down, p_mr, p_flat])
        c = P[np.arange(len(P)), P.argmax(1)]
        return post4, reg, c

    return _walkforward(bars, decode)
