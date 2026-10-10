"""Databento MBP-10 (e.g. Nasdaq ITCH) -> backtest .npz + single-day test battery.

Usage:
    python scripts/databento_mbp.py <file1.mbp-10.csv> [file2 ...]

Each file is one venue-day (possibly one symbol). Per symbol-day we:
  1. convert to the standard .npz (bids/asks (N,10,2), sided trades, ts) under
     data/l2/ -- so the dashboard and every existing tool can use it,
  2. run the snapshot strategy: fixed thr=0.4 vs dynamic (walk-fwd 85th pct),
     gross, with alpha-vs-cost decomposition and imbalance IC,
  3. report the economics the fast variant would see (median spread vs its
     0.35 bps cap) so its abstention/verdict is explicit.

Regular trading hours only (13:30-20:00 UTC); pre/post-market books on thin
ETFs are one-sided or phantom-wide and validate_book would reject them anyway.
Trade side mapping (Databento): side is the aggressor -- B(uy)=+1, A(sk/sell)=-1,
N(one) dropped from signed flow.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "strategies", "_lib"))
from l2_strategy import setup_arrays, backtest, WARM
from l2_diagnose import dynamic_thr, signal_ic, decompose, HOLD

K = 10
FAST_SPREAD_CAP = 0.35


def convert(path):
    df = pd.read_csv(path)
    ts_all = pd.to_datetime(df["ts_event"])
    hm = ts_all.dt.strftime("%H:%M")
    df = df[(hm >= "13:30") & (hm < "20:00")].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("no regular-hours rows")
    sym = str(df["symbol"].iloc[0])
    date = str(pd.to_datetime(df["ts_event"].iloc[0]).date())
    ts = (pd.to_datetime(df["ts_event"]).astype("int64") // 10**6).to_numpy(np.int64)

    N = len(df)
    bids = np.zeros((N, K, 2)); asks = np.zeros((N, K, 2))
    for k in range(K):
        bids[:, k, 0] = pd.to_numeric(df[f"bid_px_{k:02d}"], errors="coerce").fillna(0)
        bids[:, k, 1] = pd.to_numeric(df[f"bid_sz_{k:02d}"], errors="coerce").fillna(0)
        asks[:, k, 0] = pd.to_numeric(df[f"ask_px_{k:02d}"], errors="coerce").fillna(0)
        asks[:, k, 1] = pd.to_numeric(df[f"ask_sz_{k:02d}"], errors="coerce").fillna(0)

    it = df.index[(df["action"] == "T") & (df["side"].isin(["B", "A"]))].to_numpy()
    tr_side = np.where(df.loc[it, "side"].to_numpy() == "B", 1, -1)
    out = dict(bids=bids, asks=asks,
               tr_price=pd.to_numeric(df.loc[it, "price"]).to_numpy(float),
               tr_qty=pd.to_numeric(df.loc[it, "size"]).to_numpy(float),
               tr_side=tr_side.astype(int), tr_bookidx=it.astype(int),
               ts=ts, symbol=f"{sym}-{date}")
    npz = os.path.join("data", "l2", f"orderbook_{sym.lower()}_{date}.npz")
    os.makedirs(os.path.dirname(npz), exist_ok=True)
    np.savez_compressed(npz, **out)
    return out, npz


def battery(d, npz):
    D = setup_arrays(d["bids"], d["asks"], d["tr_price"], d["tr_qty"],
                     d["tr_side"], d["tr_bookidx"], d["symbol"])
    med = np.median(D["mid"]); N = D["N"]
    sp = float(np.median(D["spread"][D["valid"]]) / med * 1e4)
    span_h = (d["ts"][-1] - d["ts"][0]) / 3_600_000
    print(f"\n=== {D['symbol']} ===  ({npz})")
    print(f"  {N:,} RTH book events over {span_h:.1f}h "
          f"(~{3600 * N / (span_h * 3600):.0f} events/h), "
          f"{len(d['tr_price'])} sided trades")
    print(f"  median mid ${med:.2f}  median spread {sp:.1f} bps  "
          f"invalid books {100 * np.mean(~D['valid']):.1f}%")
    rp = {k: 100 * np.mean(D["reg"] == k) for k in (1, -1, 0)}
    print(f"  regime split: DIR {rp[1]:.1f}% | MM {rp[-1]:.1f}% | FLAT {rp[0]:.1f}%")
    ic = signal_ic(D)
    print(f"  imbalance IC (fwd move over {HOLD} events): all {ic['ic_all']:+.3f}  "
          f"DIR-only {ic['ic_dir']:+.3f}  |imb|>0.4 {ic['ic_strong']:+.3f}")

    if N < WARM + 50:
        print(f"  !! too few events for the snapshot strategy (need >{WARM + 50})")
        return
    variants = [("fixed thr=0.4", 0.4), ("dynamic thr", dynamic_thr(D["imb"]))]
    for name, thr in variants:
        r = backtest(D, thr=thr, hold=HOLD, record=True)
        dec = decompose(D, r)
        print(f"  {name:>13}: Sharpe {r['per_trade_sharpe']:+.3f}  t={r['tstat']:+.2f}  "
              f"PnL {r['pnl']:+.3f} ({r['edge_bps']:+.1f} bps/tr)  "
              f"win {r['win_rate']:.0f}%  ({r['trades']} tr)  |  "
              f"alpha {dec['alpha_bps']:+.1f} bps vs cost {dec['cost_bps']:+.1f} bps")
    print(f"  fast variant: median spread {sp:.1f} bps vs {FAST_SPREAD_CAP} bps cap "
          f"-> {'TRADES' if sp <= FAST_SPREAD_CAP else 'abstains (spread > alpha ceiling ~1bp)'}")


if __name__ == "__main__":
    files = sys.argv[1:]
    if not files:
        print("usage: python scripts/databento_mbp.py <file.mbp-10.csv> [...]"); sys.exit(1)
    for f in files:
        try:
            d, npz = convert(f)
            battery(d, npz)
        except Exception as e:
            print(f"!! {f}: {type(e).__name__}: {e}")
