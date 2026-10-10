"""Phase 3 gate 1 (regression): flow_momentum through the new shared run_backtest
engine must reproduce the same known number as Phase 0's direct-loop version --
a second, independent confirmation that the engine's fill/fee convention matches
the proven spec exactly (not just that flow.zflow's math matches).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_lib"))
from signals import build_ctx
from tools import flow_momentum
from backtest import run_backtest, stat_line, FEE_BPS

from goog_minute_regime import build, TRAIN_DAYS

TOL_BPS = 0.5
TOL_T = 0.02


def main():
    bars = build(1)
    wall = bars["day_ix"][TRAIN_DAYS]
    N = len(bars["mid"])
    ctx = build_ctx(bars, wall)

    print("=" * 78)
    print("PHASE 3 GATE 1: flow_momentum via run_backtest (regression)")
    print("=" * 78)
    tool = flow_momentum(ctx, lw=5, q=0.95, hold=5, wall=wall)
    r = run_backtest(bars, tool["direction"], tool["raw_size"], wall, N,
                     fee_bps=FEE_BPS, hold=tool["hold"])
    print(f"  flow_momentum TEST: {stat_line(r)}")

    exp_total, exp_n, exp_t = 864.3, 106, 1.53
    ok = (abs(r["pnl"] - exp_total) < TOL_BPS and r["trades"] == exp_n
          and abs(r["tstat"] - exp_t) < TOL_T)
    print(f"  expected: +{exp_total} bps, {exp_n} tr, t+{exp_t}  ->  "
          f"{'PASS' if ok else 'FAIL'}")

    print("\n" + "=" * 78)
    print(f"PHASE 3 GATE 1: {'PASS -- engine confirmed' if ok else 'FAIL'}")
    print("=" * 78)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
