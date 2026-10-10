"""Phase 5: fresh-asset validation -- does flow_momentum (the one OOS-positive
offensive signal in this program, GOOG-only until now) generalize?

Crypto (BTCUSDT/ETHUSDT/SOLUSDT/XRPUSDT): the 7 CONSECUTIVE days 2023-09-01..07
give a genuine walk-forward split (train on days 1-2, test on days 3-7, single
pass, threshold frozen on train exactly as the GOOG spec does). The 3 scattered
extra dates per symbol (2023-06-01, 2023-12-01, 2024-03-01) are NOT part of that
week -- each is tested separately as an out-of-time sanity check: the SAME
train-frozen threshold from the week, applied once, no re-fitting. This is
honest out-of-time validation, not cherry-picked re-tuning.

NFTY: reuses the already-established TRAIN_DAYS=5 / test-15 split on its full
20-day set, run through regime_lab's tools/backtest for the first time (prior
NFTY runs this session used the pre-regime_lab ad hoc scripts).

Frozen spec, UNCHANGED from GOOG (lw=5m, q=0.95, hold=5m taker) -- if any
number differs per asset, it is the asset's own train-frozen threshold value,
never a re-tuned parameter.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_lib"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "adaptive_wh"))
from signals import build_ctx
from tools import flow_momentum
from backtest import run_backtest, stat_line, FEE_BPS

from goog_minute_regime import build as build_goog, TRAIN_DAYS as GOOG_TRAIN_DAYS
from adaptive_wh_long import build_snaps

CRYPTO_SYMS = ["btcusdt", "ethusdt", "solusdt", "xrpusdt"]
CRYPTO_WEEK = [f"2023-09-{d:02d}" for d in range(1, 8)]
CRYPTO_OOT = ["2023-06-01", "2023-12-01", "2024-03-01"]
CRYPTO_TRAIN_DAYS = 2   # 2 of 7 -> 5 test days, closest achievable to GOOG's ratio


def build_snaps_dates(sym: str, bar_min: int, dates: list) -> dict:
    """Same logic as adaptive_wh_long.build_snaps, but for an explicit ordered
    date list instead of globbing every file for the symbol -- needed to
    isolate the 7-day consecutive week from the 3 scattered out-of-time dates.
    """
    parts, day_ix = [], [0]
    for d in dates:
        fp = os.path.join("data", "crypto_data", sym.upper(), d[:4], d[5:7], f"orderbook_{sym}_{d}.npz")
        r = np.load(fp, allow_pickle=True)
        bb = r["bids"][:, 0, 0].astype(float)
        ba = r["asks"][:, 0, 0].astype(float)
        ts = r["ts"].astype(np.int64)
        ok = (bb > 0) & (ba > bb)
        bb, ba, ts_v = bb[ok], ba[ok], ts[ok]
        step = bar_min * 60_000
        bounds = np.arange(ts_v[0] + step, ts_v[-1] + step, step)
        idx = np.searchsorted(ts_v, bounds, side="right") - 1
        good = idx >= 0
        idx, bounds = idx[good], bounds[good]
        tts = ts[r["tr_bookidx"].astype(int)]
        fl = r["tr_side"].astype(float) * r["tr_qty"].astype(float)
        bkt = np.clip(np.searchsorted(bounds, tts, side="left"), 0, len(bounds) - 1)
        flow = np.bincount(bkt, weights=fl, minlength=len(bounds))
        parts.append(dict(bb=bb[idx], ba=ba[idx], flow=flow, ts=bounds))
        day_ix.append(day_ix[-1] + len(idx))
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    cat["mid"] = (cat["bb"] + cat["ba"]) / 2
    cat["spread"] = cat["ba"] - cat["bb"]
    cat["day_ix"] = np.array(day_ix)
    return cat


def eval_flow_momentum(bars: dict, wall: int, lo_ctx: int, hi_ctx: int, label: str):
    ctx = build_ctx(bars, wall)
    med = np.median(bars["mid"][lo_ctx:hi_ctx])
    sp_bps = float(np.median(bars["spread"][lo_ctx:hi_ctx]) / med * 1e4)
    tool = flow_momentum(ctx, lw=5, q=0.95, hold=5, wall=wall)
    r = run_backtest(bars, tool["direction"], tool["raw_size"], lo_ctx, hi_ctx,
                     fee_bps=FEE_BPS, hold=tool["hold"])
    print(f"  {label:<28} spread {sp_bps:6.2f}bps  {stat_line(r)}")
    return r


def eval_oot_day(sym: str, date: str, thr_frozen: float):
    """Apply an ALREADY-FROZEN threshold (from the symbol's train week) to a
    single scattered out-of-time day -- no re-fitting, single pass."""
    bars = build_snaps_dates(sym, 1, [date])
    N = len(bars["mid"])
    med = np.median(bars["mid"])
    sp_bps = float(np.median(bars["spread"]) / med * 1e4)
    from flow import zflow
    z = zflow(bars["flow"], N, 5)
    thr_arr = np.full(N, thr_frozen)
    fires = np.abs(z) > thr_arr
    direction = np.where(fires, np.sign(z), 0).astype(int)
    raw_size = np.where(fires, 1.0, 0.0)
    r = run_backtest(bars, direction, raw_size, 420, N, fee_bps=FEE_BPS, hold=5)
    print(f"    OOT {date}  spread {sp_bps:6.2f}bps  {stat_line(r)}")
    return r


def frozen_threshold_from_week(bars: dict, wall: int) -> float:
    from flow import zflow, frozen_flow_threshold
    z = zflow(bars["flow"], len(bars["mid"]), 5)
    return frozen_flow_threshold(z, 420, wall, 0.95)


def main():
    print("=" * 86)
    print("REFERENCE: GOOG (already established, frozen spec lw=5m q=0.95 hold=5m)")
    print("=" * 86)
    gbars = build_goog(1)
    gwall = gbars["day_ix"][GOOG_TRAIN_DAYS]
    eval_flow_momentum(gbars, gwall, gwall, len(gbars["mid"]), "GOOG (20d, test=15d)")

    print("\n" + "=" * 86)
    print("CRYPTO: 7-consecutive-day week (train=2d, test=5d), SAME frozen spec")
    print("=" * 86)
    crypto_results = {}
    for sym in CRYPTO_SYMS:
        bars = build_snaps_dates(sym, 1, CRYPTO_WEEK)
        wall = bars["day_ix"][CRYPTO_TRAIN_DAYS]
        N = len(bars["mid"])
        r = eval_flow_momentum(bars, wall, wall, N, f"{sym.upper()} (week, test=5d)")
        crypto_results[sym] = r
        thr = frozen_threshold_from_week(bars, wall)
        for d in CRYPTO_OOT:
            eval_oot_day(sym, d, thr)

    print("\n" + "=" * 86)
    print("NFTY: established TRAIN_DAYS=5 / test=15d split, via regime_lab (first time)")
    print("=" * 86)
    nbars = build_snaps("nfty", 1)
    nwall = nbars["day_ix"][GOOG_TRAIN_DAYS]
    r_nfty = eval_flow_momentum(nbars, nwall, nwall, len(nbars["mid"]), "NFTY (20d, test=15d)")

    print("\n" + "=" * 86)
    print("SUMMARY")
    print("=" * 86)
    all_pos = all(r["pnl"] > 0 for r in crypto_results.values())
    print(f"  crypto week-test all positive: {'YES' if all_pos else 'NO'}")
    print(f"  crypto mean t-stat: {np.mean([r['tstat'] for r in crypto_results.values()]):+.2f}")


if __name__ == "__main__":
    main()
