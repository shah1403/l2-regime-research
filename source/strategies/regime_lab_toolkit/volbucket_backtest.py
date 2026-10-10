"""Real backtest (economic gate via fees, real fills, strict wall) for
volume-bucketed order-flow imbalance -- probe_volume_bucket_flow.py showed
consistent positive IC on GOOG/BTC/ETH (and weak/inconsistent on SOL/XRP, and
NFTY structurally too sparse to bucket at all). This promotes the surviving
candidates to a proper taker backtest, tested cross-asset from the same run,
per the corrected process (see feedback-cross-asset-testing memory).

Protocol:
  * bucket size V: calibrated ONCE per asset from TRAIN data only, frozen.
  * entry threshold: |z| > train-frozen q=0.95 quantile (same rule as
    flow_momentum, just on bucket-index instead of bar-index).
  * hold (buckets): a small grid {3,5,10} is searched ONLY on GOOG's train
    slice (the probe's IC curve motivated this range) and the single winner
    is FROZEN and reused identically on every asset's test -- deliberately
    NOT re-tuned per asset, to keep this a genuine "one frozen spec, many
    assets" transfer test rather than quietly re-introducing per-asset fitting.
  * GOOG: standard TRAIN_DAYS=5/test=15 wall on its 20 daily files.
  * crypto (BTC/ETH -- SOL/XRP included for completeness, expected weaker
    per the probe): 7-consecutive-day week, train=2/test=5 wall, PLUS the 3
    scattered out-of-time dates checked separately with everything frozen
    from the week -- identical split to phase5_cross_asset.py so this is
    directly comparable to flow_momentum's cross-asset result.
  * fees: GOOG at its usual 0.10bps; crypto reported BOTH gross (0 fee) and
    net at REAL Binance taker (4.5bps/side) -- the whole point of retesting
    is whether the improved IC survives real costs, not just GOOG's.
  * NFTY: skipped (0 usable buckets in the probe -- structurally too sparse,
    not re-run here to avoid wasting a call on an already-known result).
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_lib"))
from backtest import run_backtest, stat_line

from l2_strategy import setup

TARGET_BUCKETS = 300
TRAIL_Z = 50
WARM_BUCKETS = TRAIL_Z + 5
HOLD_GRID = [3, 5, 10]
Q = 0.95
FEE_GOOG = 0.10
FEE_CRYPTO_GROSS = 0.0
FEE_CRYPTO_NET = 4.5

GOOG_TRAIN_DAYS = 5
CRYPTO_WEEK = [f"2023-09-{d:02d}" for d in range(1, 8)]
CRYPTO_TRAIN_DAYS = 2
CRYPTO_OOT = ["2023-06-01", "2023-12-01", "2024-03-01"]
CRYPTO_SYMS = ["btcusdt", "ethusdt", "solusdt", "xrpusdt"]


def calibrate_V(files, target_buckets=TARGET_BUCKETS):
    """target_buckets is PER FILE/DAY (matches probe_volume_bucket_flow.py's
    density) -- pooling volume across N train files without scaling the target
    by N inflates V by ~N x too coarse (caught via GOOG test producing 0 trades
    despite a healthy train grid search; fixed here, not silently patched)."""
    total_qty, n_trades = 0.0, 0
    for fp in files:
        raw = np.load(fp, allow_pickle=True)
        q = raw["tr_qty"].astype(float)
        total_qty += q.sum()
        n_trades += len(q)
    local_target = min(target_buckets * len(files), max(10, n_trades // 3))
    return total_qty / local_target


def bucket_file(fp, V):
    D = setup(fp)
    raw = np.load(fp, allow_pickle=True)
    tq = raw["tr_qty"].astype(float)
    tsd = raw["tr_side"].astype(int)
    tbi = raw["tr_bookidx"].astype(int)
    order = np.argsort(tbi, kind="stable")
    qty, side, bidx = tq[order], tsd[order], tbi[order]
    cum = np.cumsum(qty)
    n_buckets = int(cum[-1] // V) if len(cum) else 0
    if n_buckets < WARM_BUCKETS + max(HOLD_GRID) + 5:
        return None
    edges = np.clip(np.searchsorted(cum, V * np.arange(1, n_buckets + 1)), 0, len(bidx) - 1)
    flow = np.zeros(n_buckets)
    lo = 0
    for i, e in enumerate(edges):
        flow[i] = (side[lo:e + 1] * qty[lo:e + 1]).sum()
        lo = e + 1
    snap_idx = np.clip(bidx[edges], 0, D["N"] - 1)
    return dict(mid=D["mid"][snap_idx], bb=D["bb"][snap_idx], ba=D["ba"][snap_idx],
               spread=D["spread"][snap_idx], flow=flow)


def concat_files(files, V):
    parts, day_ix = [], [0]
    for fp in files:
        b = bucket_file(fp, V)
        if b is None:
            continue
        parts.append(b)
        day_ix.append(day_ix[-1] + len(b["mid"]))
    if not parts:
        return None
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    cat["day_ix"] = np.array(day_ix)
    return cat


def zscore_causal(flow, trail=TRAIL_Z):
    n = len(flow)
    z = np.zeros(n)
    for t in range(trail, n):
        seg = flow[t - trail:t]
        sd = seg.std()
        z[t] = flow[t] / (sd + 1e-9) if sd > 0 else 0.0
    return z


def signal(bars, wall, q=Q):
    z = zscore_causal(bars["flow"])
    thr = float(np.quantile(np.abs(z[WARM_BUCKETS:wall]), q))
    fires = np.abs(z) > thr
    direction = np.where(fires, np.sign(z), 0).astype(int)
    raw_size = np.where(fires, 1.0, 0.0)
    return direction, raw_size, thr


def freeze_hold_on_goog():
    files = sorted(glob.glob(os.path.join("data", "stock_data", "GOOG", "L2_top", "*", "*", "orderbook_goog_*.npz")))
    train_files = files[:GOOG_TRAIN_DAYS]
    V = calibrate_V(train_files)
    bars = concat_files(files, V)
    wall = bars["day_ix"][GOOG_TRAIN_DAYS]
    d, s, thr = signal(bars, wall)
    print("=" * 90)
    print("FREEZE HOLD: small grid on GOOG train slice only (declared, printed in full)")
    print("=" * 90)
    best, best_t = None, -np.inf
    for h in HOLD_GRID:
        r = run_backtest(bars, d, s, WARM_BUCKETS, wall, fee_bps=FEE_GOOG, hold=h)
        print(f"  hold={h:>2} buckets: {stat_line(r)}")
        if r["trades"] >= 10 and r["tstat"] > best_t:
            best_t, best = r["tstat"], h
    print(f"  FROZEN hold = {best} buckets (train t={best_t:+.2f}) -- used unchanged everywhere below")
    return best, V, bars, wall


def run_goog(hold, V, bars, wall):
    d, s, thr = signal(bars, wall)
    N = len(bars["mid"])
    r = run_backtest(bars, d, s, wall, N, fee_bps=FEE_GOOG, hold=hold)
    print(f"\nGOOG   (test=15d, thr={thr:.2f}, V={V:.1f}): {stat_line(r)}")
    return r


def run_crypto_symbol(sym, hold):
    week_files = [os.path.join("data", "crypto_data", sym.upper(), d[:4], d[5:7], f"orderbook_{sym}_{d}.npz") for d in CRYPTO_WEEK]
    train_files = week_files[:CRYPTO_TRAIN_DAYS]
    V = calibrate_V(train_files)
    bars = concat_files(week_files, V)
    if bars is None:
        print(f"{sym.upper():<8} week: no usable buckets"); return
    wall = bars["day_ix"][CRYPTO_TRAIN_DAYS]
    N = len(bars["mid"])
    d, s, thr = signal(bars, wall)
    rg = run_backtest(bars, d, s, wall, N, fee_bps=FEE_CRYPTO_GROSS, hold=hold)
    rn = run_backtest(bars, d, s, wall, N, fee_bps=FEE_CRYPTO_NET, hold=hold)
    print(f"{sym.upper():<8} week (thr={thr:.2f}, V={V:.1f})")
    print(f"  gross(0bps): {stat_line(rg)}")
    print(f"  net(4.5bps): {stat_line(rn)}")

    for date in CRYPTO_OOT:
        fp = os.path.join("data", "crypto_data", sym.upper(), date[:4], date[5:7], f"orderbook_{sym}_{date}.npz")
        b = bucket_file(fp, V)          # SAME frozen V
        if b is None:
            print(f"    OOT {date}: no usable buckets"); continue
        b["day_ix"] = np.array([0, len(b["mid"])])
        z = zscore_causal(b["flow"])
        fires = np.abs(z) > thr          # SAME frozen threshold
        dO = np.where(fires, np.sign(z), 0).astype(int)
        sO = np.where(fires, 1.0, 0.0)
        rgO = run_backtest(b, dO, sO, WARM_BUCKETS, len(b["mid"]), fee_bps=FEE_CRYPTO_GROSS, hold=hold)
        rnO = run_backtest(b, dO, sO, WARM_BUCKETS, len(b["mid"]), fee_bps=FEE_CRYPTO_NET, hold=hold)
        print(f"    OOT {date}  gross {stat_line(rgO)}  |  net {stat_line(rnO)}")
    return rg, rn


def main():
    hold, gV, gbars, gwall = freeze_hold_on_goog()
    run_goog(hold, gV, gbars, gwall)

    print("\n" + "=" * 90)
    print(f"CRYPTO (hold={hold} buckets frozen from GOOG, week train=2d/test=5d + 3 OOT dates)")
    print("=" * 90)
    for sym in CRYPTO_SYMS:
        run_crypto_symbol(sym, hold)
        print()

    print("=" * 90)
    print("NFTY: skipped -- 0 usable volume buckets in the probe (structurally too sparse)")
    print("=" * 90)


if __name__ == "__main__":
    main()
