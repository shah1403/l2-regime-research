"""Phase 1 gate: every signal is finite outside its documented warmup, and
signals.py's wrappers are pure orchestration (bit-identical to calling the
Phase 0 functions directly) -- not a silent reimplementation.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_lib"))
from signals import (build_ctx, flow_zscore, kinematic_z, variance_ratio_series,
                     autocorr_series, vol_zscore, spread_zscore,
                     DEFAULT_FLOW_LW, DEFAULT_KIN_W, DEFAULT_SPREAD_WARM)
from flow import zflow
from kinematics import poly_derivs

from goog_minute_regime import build, TRAIN_DAYS, WARMB
from l2_strategy import autocorr1


def check_no_nan_inf(name, arr, warm):
    bad = ~np.isfinite(arr[warm:])
    ok = not bad.any()
    print(f"  {name:<12} finite past warmup={warm:>4}: {'PASS' if ok else 'FAIL'}"
          f"{'' if ok else f'  ({bad.sum()} bad values)'}")
    return ok


def check_orchestration(name, a, b):
    ok = np.array_equal(a, b, equal_nan=False) if np.issubdtype(a.dtype, np.floating) \
        else np.array_equal(a, b)
    ok = np.allclose(a, b, atol=0, rtol=0, equal_nan=True)
    print(f"  {name:<12} bit-identical to direct call: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    bars = build(1)
    wall = bars["day_ix"][TRAIN_DAYS]
    N = len(bars["mid"])

    print("=" * 78)
    print("PHASE 1a: build_ctx over full GOOG bars")
    print("=" * 78)
    ctx = build_ctx(bars, wall)

    ok = True
    ok &= check_no_nan_inf("flow_z", ctx["flow_z"], 390)          # var_window
    ok &= check_no_nan_inf("kin_z1", ctx["kin_z1"], DEFAULT_KIN_W)
    ok &= check_no_nan_inf("kin_z2", ctx["kin_z2"], DEFAULT_KIN_W)
    ok &= check_no_nan_inf("vr", ctx["vr"], 100)
    ok &= check_no_nan_inf("ac", ctx["ac"], 100)
    ok &= check_no_nan_inf("vol_z", ctx["vol_z"], 300)
    ok &= check_no_nan_inf("spread_z", ctx["spread_z"], DEFAULT_SPREAD_WARM)

    print("\n" + "=" * 78)
    print("PHASE 1b: orchestration check (signals.py wrappers vs direct Phase 0 calls)")
    print("=" * 78)
    z_direct = zflow(bars["flow"], N, DEFAULT_FLOW_LW)
    ok &= check_orchestration("flow_z", ctx["flow_z"], z_direct)
    z1_direct, z2_direct = poly_derivs(bars["mid"], DEFAULT_KIN_W)
    ok &= check_orchestration("kin_z1", ctx["kin_z1"], z1_direct)
    ok &= check_orchestration("kin_z2", ctx["kin_z2"], z2_direct)

    print("\n" + "=" * 78)
    print("PHASE 1c: human sanity check -- per-signal summary stats (test window only)")
    print("=" * 78)
    print(f"  {'signal':<12}{'mean':>10}{'std':>10}{'lag1-ac':>10}")
    for name in ("flow_z", "kin_z1", "kin_z2", "vr", "ac", "vol_z", "spread_z"):
        x = ctx[name][wall:]
        lag1 = autocorr1(x)
        print(f"  {name:<12}{x.mean():>10.3f}{x.std():>10.3f}{lag1:>10.3f}")

    print("\n" + "=" * 78)
    print(f"PHASE 1 GATE: {'PASS -- proceed to Phase 2' if ok else 'FAIL -- fix signals.py'}")
    print("=" * 78)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
