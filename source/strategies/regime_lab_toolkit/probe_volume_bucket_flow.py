"""Cheap cross-asset PROBE (correlation only, no backtest, no new tool built yet)
for volume-bucketed order-flow imbalance -- the principled fix motivated by
flow_momentum's cross-asset failure (phase5_cross_asset.py): a fixed 5-CALENDAR-
minute window means wildly different amounts of information on NFTY (~48
trades/day) vs BTCUSDT (thousands/minute). Bucket by cumulative traded VOLUME
instead, so the window self-adapts to each asset's own trading intensity.

Protocol (same discipline as everything else in this program, deliberately
cheap): for each asset, use its FIRST available day to calibrate the bucket
size V (V = that day's total volume / TARGET_BUCKETS) -- frozen afterward, never
re-tuned. Every subsequent day is bucketed with that frozen V and scored purely
as an out-of-sample correlation (IC) between bucket-level signed-flow z-score
and the forward return H buckets later, at several H. No entries, no fees, no
hold logic -- this is a "does the idea have ANY legs" check across ALL assets
at once, per the corrected process: probe cross-asset BEFORE building.

Reuses l2_strategy.setup() (mid/bb/ba/valid) -- already validated, asset-
agnostic. Uses the raw trade tape (tr_price/tr_qty/tr_side/tr_bookidx) directly
for bucketing, not the calendar-bar builders (which would throw away the
intra-bar granularity bucketing needs).
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_lib"))
from l2_strategy import setup

ASSETS = ["goog", "btcusdt", "ethusdt", "solusdt", "xrpusdt", "nfty"]
TARGET_BUCKETS = 300      # buckets/day on the calibration day -> sets V
H_LIST = [1, 3, 5, 10, 20]
TRAIL_Z = 50               # trailing buckets for the z-score's own history
WARM_BUCKETS = TRAIL_Z + 5


def load_day(path):
    D = setup(path)
    raw = np.load(path, allow_pickle=True)
    return D, raw["tr_price"].astype(float), raw["tr_qty"].astype(float), \
        raw["tr_side"].astype(int), raw["tr_bookidx"].astype(int)


def bucket_day(D, tr_qty, tr_side, tr_bookidx, V):
    """Cumulative-volume bucketing of the trade tape into (bucket_mid, bucket_flow)."""
    order = np.argsort(tr_bookidx, kind="stable")
    qty, side, bidx = tr_qty[order], tr_side[order], tr_bookidx[order]
    cum = np.cumsum(qty)
    n_buckets = int(cum[-1] // V) if len(cum) else 0
    if n_buckets < WARM_BUCKETS + max(H_LIST) + 5:
        return None, None
    edges = np.searchsorted(cum, V * np.arange(1, n_buckets + 1))
    edges = np.clip(edges, 0, len(bidx) - 1)
    flow = np.zeros(n_buckets)
    lo = 0
    for i, e in enumerate(edges):
        flow[i] = (side[lo:e + 1] * qty[lo:e + 1]).sum()
        lo = e + 1
    bucket_snap_idx = bidx[edges]
    mid = D["mid"][np.clip(bucket_snap_idx, 0, D["N"] - 1)]
    return mid, flow


def zscore_causal(flow, trail=TRAIL_Z):
    n = len(flow)
    z = np.zeros(n)
    for t in range(trail, n):
        seg = flow[t - trail:t]
        sd = seg.std()
        z[t] = flow[t] / (sd + 1e-9) if sd > 0 else 0.0
    return z


def ic(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30 or a[m].std() == 0 or b[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _asset_dir(asset):
    if asset == "goog":
        return os.path.join("data", "stock_data", "GOOG", "L2_top")
    if asset == "nfty":
        return os.path.join("data", "stock_data", "NFTY_ETF")
    return os.path.join("data", "crypto_data", asset.upper())


def main():
    print(f"{'asset':<10}{'V(calib)':>12}{'days':>6}{'buckets':>9} | "
          + "".join(f"IC@H={h:<4}".rjust(10) for h in H_LIST))
    for asset in ASSETS:
        files = sorted(glob.glob(os.path.join(_asset_dir(asset), "*", "*", f"orderbook_{asset}_*.npz")))
        if len(files) < 2:
            print(f"{asset:<10}  (skipped: need >=2 day files, have {len(files)})")
            continue

        # --- calibrate V on the FIRST day only, freeze it ---
        Dc, _, qc, _, _ = load_day(files[0])
        # sparse instruments (e.g. NFTY, ~50 trades/day) can't support
        # TARGET_BUCKETS meaningfully -- scale down so each bucket averages
        # >=3 trades, reported explicitly rather than silently degenerate
        local_target = min(TARGET_BUCKETS, max(10, len(qc) // 3))
        V = float(qc.sum()) / local_target

        z_all = {h: [] for h in H_LIST}
        fwd_all = {h: [] for h in H_LIST}
        total_buckets = 0
        for fp in files[1:]:
            D, tp, tq, ts_, tbi = load_day(fp)
            mid, flow = bucket_day(D, tq, ts_, tbi, V)
            if mid is None:
                continue
            z = zscore_causal(flow)
            n = len(mid)
            total_buckets += n
            for h in H_LIST:
                fwd = np.full(n, np.nan)
                fwd[:n - h] = (mid[h:] - mid[:n - h]) / mid[:n - h] * 1e4
                lo = WARM_BUCKETS
                z_all[h].append(z[lo:n - h])
                fwd_all[h].append(fwd[lo:n - h])

        ics = []
        for h in H_LIST:
            if not z_all[h]:
                ics.append(float("nan")); continue
            zc = np.concatenate(z_all[h])
            fc = np.concatenate(fwd_all[h])
            ics.append(ic(zc, fc))
        row = f"{asset:<10}{V:>12.2f}{len(files)-1:>6}{total_buckets:>9} | "
        row += "".join(f"{v:>+10.4f}" if np.isfinite(v) else f"{'nan':>10}" for v in ics)
        print(row, flush=True)


if __name__ == "__main__":
    main()
