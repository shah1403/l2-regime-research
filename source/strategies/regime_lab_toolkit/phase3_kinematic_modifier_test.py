"""Phase 3 gate 2 (real experiment): does kinematic_ladder, repurposed as a
confidence MODIFIER on top of flow_momentum's direction, help or hurt out of
sample? Single pass, TEST window only. This is not assumed to work -- the
underlying math has a track record of negative skill as a standalone forecaster
in this repo. If it fails, kinematic_ladder still ships in tools.MODIFIERS for
Phase 4 plumbing, but its default weight should be treated as neutral (1.0),
not helpful.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_lib"))
from signals import build_ctx
from tools import flow_momentum, kinematic_modifier
from backtest import run_backtest, stat_line, FEE_BPS

from goog_minute_regime import build, TRAIN_DAYS


def main():
    bars = build(1)
    wall = bars["day_ix"][TRAIN_DAYS]
    N = len(bars["mid"])
    ctx = build_ctx(bars, wall)

    tool = flow_momentum(ctx, lw=5, q=0.95, hold=5, wall=wall)
    r_alone = run_backtest(bars, tool["direction"], tool["raw_size"], wall, N,
                           fee_bps=FEE_BPS, hold=tool["hold"])

    mult = kinematic_modifier(ctx, tool["direction"], wall=wall)
    size_mod = tool["raw_size"] * mult
    r_mod = run_backtest(bars, tool["direction"], size_mod, wall, N,
                         fee_bps=FEE_BPS, hold=tool["hold"])

    print("=" * 78)
    print("PHASE 3 GATE 2: flow_momentum alone vs flow_momentum * kinematic_ladder")
    print("=" * 78)
    print(f"  flow_momentum alone            : {stat_line(r_alone)}")
    print(f"  flow_momentum * kinematic mult : {stat_line(r_mod)}")

    n_zeroed = int(((tool["raw_size"] > 0) & (mult == 0)).sum())
    n_boosted = int(((tool["raw_size"] > 0) & (mult > 1)).sum())
    n_neutral = int(((tool["raw_size"] > 0) & (mult == 1)).sum())
    print(f"\n  of {n_zeroed + n_boosted + n_neutral} firing bars: "
          f"{n_boosted} boosted (x2), {n_zeroed} zeroed out, {n_neutral} neutral")

    print("\n" + "=" * 78)
    if r_mod["pnl"] > r_alone["pnl"] and r_mod["tstat"] > r_alone["tstat"]:
        verdict = "kinematic_ladder HELPS on this window (both PnL and t-stat improve)"
    elif r_mod["pnl"] < r_alone["pnl"] and r_mod["tstat"] < r_alone["tstat"]:
        verdict = "kinematic_ladder HURTS on this window (both PnL and t-stat worse)"
    else:
        verdict = "kinematic_ladder is MIXED on this window (metrics disagree)"
    print(f"RESULT: {verdict}")
    print("This is a real, disclosed outcome -- kinematic_ladder ships in the "
          "registry either way,\ndefault weight treated as neutral unless this "
          "clearly helps.")
    print("=" * 78)


if __name__ == "__main__":
    main()
