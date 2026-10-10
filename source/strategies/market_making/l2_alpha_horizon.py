"""The decisive experiment: does top-of-book imbalance predict mid moves at ALL,
and at what horizon -- measured at native tick resolution (every book update).

For each symbol-day, load the FULL Binance futures bookTicker file (millions of
updates), compute imb=(bq-aq)/(bq+aq), and for horizons 0.1s..60s report:

  IC        corr(imb[t], mid move over next h seconds), all updates
  move      mean signal-direction mid move (bps) when |imb| beyond its 95th pct
  taker     net bps of a crossing round trip: fill at NEXT update's touch
            (entry long at ask[i+1], exit at bid[j] after h seconds) -- what the
            current strategy's execution would earn gross at this horizon
  maker     upper bound if entry were a filled passive quote (entry at bid[i],
            exit taker at bid[j]) -- spread capture ceiling, ignores fill risk

Overlapping samples inflate t-stats, so read point estimates, not significance.
Fills at the next update assume ~ms latency (colocation-grade); this is the
best case for the signal, which is exactly what we want to bound.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data_pipeline"))
from binance_l2_fetch import download, BASE, TMP_DIR

HORIZONS_S = [0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60]
GRID = [("BTCUSDT", "2023-09-01"), ("ETHUSDT", "2023-09-01"),
        ("SOLUSDT", "2023-09-01"), ("XRPUSDT", "2023-09-01")]
Q = 0.95                       # signal gate: |imb| beyond its own 95th percentile


def load_full(symbol, date):
    zp = download(f"{BASE}/bookTicker/{symbol}/{symbol}-bookTicker-{date}.zip",
                  os.path.join(TMP_DIR, f"{symbol}-bookTicker-{date}.zip"))
    df = pd.read_csv(zp)
    if "best_bid_price" not in df.columns:
        df = pd.read_csv(zp, header=None,
                         names=["update_id", "best_bid_price", "best_bid_qty",
                                "best_ask_price", "best_ask_qty",
                                "transaction_time", "event_time"])
    bb = df["best_bid_price"].to_numpy(np.float64)
    ba = df["best_ask_price"].to_numpy(np.float64)
    bq = df["best_bid_qty"].to_numpy(np.float64)
    aq = df["best_ask_qty"].to_numpy(np.float64)
    ts = np.maximum.accumulate(df["event_time"].to_numpy(np.int64))
    ok = (ba > bb) & (bq > 0) & (aq > 0)
    return bb[ok], ba[ok], bq[ok], aq[ok], ts[ok]


def study(symbol, date):
    bb, ba, bq, aq, ts = load_full(symbol, date)
    n = len(ts)
    mid = (bb + ba) / 2
    imb = (bq - aq) / (bq + aq)
    med = np.median(mid)
    spread_bps = float(np.median((ba - bb) / mid) * 1e4)
    hi = np.quantile(imb, Q); lo = np.quantile(imb, 1 - Q)
    sig = np.where(imb >= hi, 1, np.where(imb <= lo, -1, 0))
    print(f"\n{symbol} {date}: {n:,} book updates, median spread {spread_bps:.3f} bps, "
          f"imb gates [{lo:+.2f},{hi:+.2f}]  ({np.mean(sig != 0) * 100:.1f}% fire)")
    print(f"  {'h':>6} {'IC':>8} {'move_bps':>9} {'taker_bps':>10} {'maker_bps':>10} {'n_sig':>10}")
    for h in HORIZONS_S:
        j = np.searchsorted(ts, ts + int(h * 1000))
        v = (j < n) & (np.arange(n) + 1 < n)
        jj = np.minimum(j, n - 1)
        fwd = (mid[jj] - mid) / mid * 1e4
        m = v & (sig != 0)
        ic = float(np.corrcoef(imb[v], fwd[v])[0, 1])
        move = float(np.mean(sig[m] * fwd[m]))
        i1 = np.minimum(np.arange(n) + 1, n - 1)          # fill at next update
        long_pnl = (bb[jj] - ba[i1]) / mid * 1e4          # cross in, cross out
        short_pnl = (bb[i1] - ba[jj]) / mid * 1e4
        taker = float(np.mean(np.where(sig[m] > 0, long_pnl[m], short_pnl[m])))
        mlong = (bb[jj] - bb) / mid * 1e4                 # passive entry bound
        mshort = (ba - ba[jj]) / mid * 1e4
        maker = float(np.mean(np.where(sig[m] > 0, mlong[m], mshort[m])))
        print(f"  {h:>5.2f}s {ic:+8.4f} {move:+9.3f} {taker:+10.3f} {maker:+10.3f} "
              f"{int(m.sum()):>10,}")
    return spread_bps


if __name__ == "__main__":
    os.makedirs(TMP_DIR, exist_ok=True)
    grid = GRID if len(sys.argv) < 2 else \
        [(s, sys.argv[2] if len(sys.argv) > 2 else "2023-09-01")
         for s in sys.argv[1].split(",")]
    for sym, date in grid:
        try:
            study(sym, date)
        except Exception as e:
            print(f"!! {sym} {date}: {type(e).__name__}: {e}", flush=True)
