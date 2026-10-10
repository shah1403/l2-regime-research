"""Phase 0 gate: the extracted functions must reproduce ALREADY-KNOWN numbers.

This is a regression check, not a new experiment. Any deviation means the
extraction has a bug -- fix regime_lab/flow.py or regime_lab/kinematics.py, do not
adjust the target numbers.

1. Reproduces scripts/curated_nfty_goog.py's goog_flow() train-select/test-freeze
   procedure using the extracted flow.zflow / flow.frozen_flow_threshold, on GOOG
   1-min bars. Known targets (verified in-session against the live run of
   curated_nfty_goog.py):
     Lw=5m H=5m  (auto-selected by train t-stat) -> TEST +864.3 bps, 106 tr, t+1.53
     Lw=5m H=30m (manual second look)            -> TEST +839.6 bps,  58 tr, t+1.61
2. Diffs kinematics.poly_derivs / kinematics.kinematic_state against
   scripts/l2_mmdir.py's poly_derivs / dir_target on one GOOG day, bit-identical.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_lib"))
from flow import zflow, frozen_flow_threshold
from kinematics import poly_derivs, kinematic_state

from goog_minute_regime import build, TRAIN_DAYS
import l2_mmdir

TOL_BPS = 0.5       # float/order-of-summation tolerance
TOL_TRADES = 0      # trade counts must match exactly (deterministic control flow)
TOL_T = 0.02


def stat(mv):
    mv = np.asarray(mv, float)
    if len(mv) == 0:
        return "0 trades", 0.0, 0, 0.0
    sh = mv.mean() / mv.std() if mv.std() > 0 else 0.0
    tstat = sh * np.sqrt(len(mv))
    line = (f"{mv.sum():+9.1f} bps | {len(mv):4d} tr, {mv.mean():+7.2f}/tr, "
            f"win {100 * np.mean(mv > 0):4.1f}%, t {tstat:+5.2f}")
    return line, float(mv.sum()), len(mv), float(tstat)


def run(z, thr, h, lo, hi, mid, bb, ba, fee_bps):
    """Identical math to the `run()` closure in curated_nfty_goog.py:122-134."""
    mv = []
    t = lo
    while t < hi - h - 2:
        if abs(z[t]) > thr:
            side = np.sign(z[t])
            px_in = ba[t + 1] if side > 0 else bb[t + 1]
            e = t + 1 + h
            px_out = bb[e] if side > 0 else ba[e]
            m = side * (px_out - px_in) / mid[t] * 1e4 - 2 * fee_bps
            mv.append(m)
            t = e + 1
        else:
            t += 1
    return np.array(mv)


def check_flow_regression():
    print("=" * 78)
    print("1. FLOW STRATEGY REGRESSION CHECK")
    print("=" * 78)
    FEE_BPS = 0.10
    bars = build(1)
    N = len(bars["mid"])
    wall = bars["day_ix"][TRAIN_DAYS]
    mid, bb, ba, fl = bars["mid"], bars["bb"], bars["ba"], bars["flow"]

    best, best_t = None, -np.inf
    train_rows = []
    for lw in (5, 15, 30):
        z = zflow(fl, N, lw)
        for h in (5, 15, 30):
            thr = frozen_flow_threshold(z, 420, wall, 0.95)
            mv_tr = run(z, thr, h, 420, wall, mid, bb, ba, FEE_BPS)
            line, total, n, t = stat(mv_tr)
            train_rows.append((lw, h, thr, total, n, t))
            print(f"  TRAIN Lw={lw:>2}m H={h:>2}m: {line}")
            if n >= 15 and t > best_t:
                best_t, best = t, (lw, h, thr)

    ok = True
    lw, h, thr = best
    z = zflow(fl, N, lw)
    mv_te = run(z, thr, h, wall, N, mid, bb, ba, FEE_BPS)
    line, total, n, t = stat(mv_te)
    print(f"\n  FROZEN Lw={lw}m H={h}m -> TEST : {line}")
    exp_total, exp_n, exp_t = 864.3, 106, 1.53
    match = (abs(total - exp_total) < TOL_BPS and n == exp_n and abs(t - exp_t) < TOL_T)
    print(f"  expected: +{exp_total} bps, {exp_n} tr, t+{exp_t}  ->  "
          f"{'PASS' if match else 'FAIL'}")
    ok &= match

    lw2, h2 = 5, 30
    z2 = zflow(fl, N, lw2)
    thr2 = frozen_flow_threshold(z2, 420, wall, 0.95)
    mv_te2 = run(z2, thr2, h2, wall, N, mid, bb, ba, FEE_BPS)
    line2, total2, n2, t2 = stat(mv_te2)
    print(f"\n  SECOND LOOK Lw={lw2}m H={h2}m -> TEST : {line2}")
    exp_total2, exp_n2, exp_t2 = 839.6, 58, 1.61
    match2 = (abs(total2 - exp_total2) < TOL_BPS and n2 == exp_n2
              and abs(t2 - exp_t2) < TOL_T)
    print(f"  expected: +{exp_total2} bps, {exp_n2} tr, t+{exp_t2}  ->  "
          f"{'PASS' if match2 else 'FAIL'}")
    ok &= match2
    return ok


def check_kinematics_regression():
    print("\n" + "=" * 78)
    print("2. KINEMATICS REGRESSION CHECK (vs scripts/l2_mmdir.py)")
    print("=" * 78)
    bars = build(1)
    # one day's worth of bars is plenty for a bit-identical math check
    d0, d1 = bars["day_ix"][6], bars["day_ix"][7]
    mid_day = bars["mid"][d0:d1]

    z1_ref, z2_ref = l2_mmdir.poly_derivs(mid_day, W=l2_mmdir.POLY_W)
    z1_new, z2_new = poly_derivs(mid_day, W=l2_mmdir.POLY_W, vol_win=l2_mmdir.VOL_WIN)
    z1_match = np.allclose(z1_ref, z1_new, atol=1e-9)
    z2_match = np.allclose(z2_ref, z2_new, atol=1e-9)
    print(f"  poly_derivs z1 bit-identical: {'PASS' if z1_match else 'FAIL'}")
    print(f"  poly_derivs z2 bit-identical: {'PASS' if z2_match else 'FAIL'}")

    ladder_ref = l2_mmdir.dir_target(z1_ref, z2_ref)
    state = kinematic_state(z1_new, z2_new, q1=l2_mmdir.Z1_MIN,
                            q2c=l2_mmdir.Z2_CONF, q2p=l2_mmdir.Z2_PEAK)
    size = np.where(state["accel"], 2.0,
                    np.where(state["peak"], 0.0,
                             np.where(state["steady"], 1.0, 0.0)))
    s = np.sign(z1_new) * (np.abs(z1_new) > l2_mmdir.Z1_MIN)
    ladder_new = s * size
    ladder_match = np.allclose(ladder_ref, ladder_new, atol=1e-9)
    print(f"  dir_target ladder reconstructed from kinematic_state bit-identical: "
          f"{'PASS' if ladder_match else 'FAIL'}")

    return z1_match and z2_match and ladder_match


if __name__ == "__main__":
    ok1 = check_flow_regression()
    ok2 = check_kinematics_regression()
    print("\n" + "=" * 78)
    print(f"PHASE 0 GATE: {'PASS -- proceed to Phase 1' if (ok1 and ok2) else 'FAIL -- fix extraction before proceeding'}")
    print("=" * 78)
    sys.exit(0 if (ok1 and ok2) else 1)
