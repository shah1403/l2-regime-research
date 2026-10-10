"""Phase 4 gate: unit tests for combine_signals (synthetic 2-tool stub, both
modes) + the real two-tool minimal case (flow_momentum signal + kinematic_ladder
modifier) on GOOG, same wall, single test pass -- compared against Phase 3's
flow_momentum-alone baseline. Must not be a regression, or the cause must be
diagnosed and disclosed. Prints the Stage-1 weight trajectory for a human
sanity check before Phase 5 work begins.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_lib"))
from signals import build_ctx
from tools import TOOLS, MODIFIERS, flow_momentum
from regime import walkforward_regime3
from assembler import combine_signals, run_assembler
from backtest import run_backtest, stat_line, FEE_BPS

from goog_minute_regime import build, TRAIN_DAYS


def unit_test_combine_signals():
    print("=" * 78)
    print("UNIT TESTS: combine_signals synthetic 2-tool stub")
    print("=" * 78)
    ok = True

    # Case 1: soft blend, both agree, different weights/sizes
    d, s = combine_signals({"A": (1, 1.0), "B": (1, 2.0)},
                           {"A": 0.6, "B": 0.4}, mode="soft")
    match = (d == 1 and abs(s - 1.4) < 1e-9)
    print(f"  soft/agree:    dir={d} size={s:.3f}  expected dir=1 size=1.400  "
          f"{'PASS' if match else 'FAIL'}")
    ok &= match

    # Case 2: soft blend, exact opposing cancellation -> abstain
    d, s = combine_signals({"A": (1, 1.0), "B": (-1, 1.0)},
                           {"A": 0.5, "B": 0.5}, mode="soft")
    match = (d == 0 and s == 0.0)
    print(f"  soft/cancel:   dir={d} size={s:.3f}  expected dir=0 size=0.000  "
          f"{'PASS' if match else 'FAIL'}")
    ok &= match

    # Case 3: hard mode, min_confirm=1, one tool active -> takes it
    d, s = combine_signals({"A": (1, 1.0), "B": (-1, 1.0)},
                           {"A": 0.6, "B": 0.0}, mode="hard", min_confirm=1)
    match = (d == 1 and abs(s - 1.0) < 1e-9)
    print(f"  hard/1active:  dir={d} size={s:.3f}  expected dir=1 size=1.000  "
          f"{'PASS' if match else 'FAIL'}")
    ok &= match

    # Case 4: hard mode, min_confirm=2, only one tool active -> abstain
    d, s = combine_signals({"A": (1, 1.0), "B": (0, 0.0)},
                           {"A": 0.6, "B": 0.0}, mode="hard", min_confirm=2)
    match = (d == 0 and s == 0.0)
    print(f"  hard/underconf: dir={d} size={s:.3f}  expected dir=0 size=0.000  "
          f"{'PASS' if match else 'FAIL'}")
    ok &= match

    print(f"\n  UNIT TESTS: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    unit_ok = unit_test_combine_signals()

    print("\n" + "=" * 78)
    print("PHASE 4 GATE: two-tool assembler (flow_momentum + kinematic_ladder) "
          "vs flow_momentum alone")
    print("=" * 78)
    bars = build(1)
    wall = bars["day_ix"][TRAIN_DAYS]
    N = len(bars["mid"])
    ctx = build_ctx(bars, wall)

    print("Fitting walk-forward 3D regime posterior (Phase 2's winning config)...",
          flush=True)
    regime_out = walkforward_regime3(bars)

    tool = flow_momentum(ctx, lw=5, q=0.95, hold=5, wall=wall)
    r_baseline = run_backtest(bars, tool["direction"], tool["raw_size"], wall, N,
                              fee_bps=FEE_BPS, hold=tool["hold"])
    print(f"\n  baseline (flow_momentum alone) : {stat_line(r_baseline)}")

    r_assembled = run_assembler(bars, TOOLS, MODIFIERS, ctx, regime_out, wall,
                                mode="soft")
    print(f"  assembled (2-tool, soft mode)  : {stat_line(r_assembled)}")

    delta_pnl = r_assembled["pnl"] - r_baseline["pnl"]
    delta_t = r_assembled["tstat"] - r_baseline["tstat"]
    print(f"\n  delta: {delta_pnl:+.1f} bps PnL, {delta_t:+.2f} t-stat")
    if delta_pnl >= 0:
        print("  RESULT: assembled system is NOT a regression vs. baseline.")
    else:
        print("  RESULT: assembled system UNDERPERFORMS baseline -- diagnosing cause below.")
        wt = r_assembled["weight_trace"]
        fm_w = wt["flow_momentum"]
        kl_w = wt["kinematic_ladder"]
        fm_valid = fm_w[~np.isnan(fm_w)]
        kl_valid = kl_w[~np.isnan(kl_w)]
        print(f"    flow_momentum weight:  mean {fm_valid.mean():.3f}  "
              f"min {fm_valid.min():.3f}  max {fm_valid.max():.3f}")
        print(f"    kinematic_ladder weight: mean {kl_valid.mean():.3f}  "
              f"min {kl_valid.min():.3f}  max {kl_valid.max():.3f}")
        print("    See assembler.tool_applicability's docstring: if this still "
              "underperforms after that fix,\n     re-diagnose from the weight "
              "trace above rather than assuming cold-start.")

    print("\n" + "=" * 78)
    print("Stage-1 weight trajectory sanity check (test window)")
    print("=" * 78)
    wt = r_assembled["weight_trace"]
    for name, w in wt.items():
        w_valid = w[~np.isnan(w)]
        if len(w_valid) == 0:
            print(f"  {name:<18} no weight samples recorded")
            continue
        n_zero = int(np.mean(w_valid == 0) * 100)
        n_one = int(np.mean(w_valid >= 0.99) * 100)
        print(f"  {name:<18} mean={w_valid.mean():.3f} std={w_valid.std():.3f} "
              f"  {n_zero}% at 0   {n_one}% near 1")

    gate_ok = unit_ok  # PnL delta is diagnostic, not a hard pass/fail per the plan
    print("\n" + "=" * 78)
    print(f"PHASE 4 GATE (unit tests): {'PASS' if gate_ok else 'FAIL'}")
    print("Two-tool vs baseline comparison and weight trajectory are reported "
          "above for human review.")
    print("=" * 78)
    sys.exit(0 if gate_ok else 1)


if __name__ == "__main__":
    main()
