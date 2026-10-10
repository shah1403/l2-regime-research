"""Stream a large (multi-day) Databento MBP-10 CSV into per-day backtest .npz.

Handles files far bigger than RAM by chunked reads of only the needed columns.
Per trading day (RTH 13:30-20:00 UTC): book events sampled to ~TARGET snapshots
(K=1 touch depth) + the FULL sided trade tape mapped to snapshot indices --
trades between kept snapshots are preserved, which the MM fill model needs.

Usage: python scripts/databento_month.py <big.mbp-10.csv>
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

USECOLS = ["ts_event", "action", "side", "price", "size",
           "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00", "symbol"]
TARGET = 25_000
CHUNK = 2_000_000


def flush_day(day, parts, outdir):
    df = pd.concat(parts, ignore_index=True)
    n = len(df)
    if n < 2000:
        print(f"  {day}: only {n} RTH rows, skipped"); return None
    sym = str(df["symbol"].iloc[0]).lower()
    ts_ms = (pd.to_datetime(df["ts_event"], format="ISO8601").astype("int64")
             // 10**6).to_numpy(np.int64)
    k = max(1, n // TARGET)
    keep = np.arange(0, n, k)
    m = len(keep)
    bb = pd.to_numeric(df["bid_px_00"], errors="coerce").fillna(0).to_numpy()
    ba = pd.to_numeric(df["ask_px_00"], errors="coerce").fillna(0).to_numpy()
    bq = pd.to_numeric(df["bid_sz_00"], errors="coerce").fillna(0).to_numpy()
    aq = pd.to_numeric(df["ask_sz_00"], errors="coerce").fillna(0).to_numpy()
    bids = np.stack([bb[keep], bq[keep]], -1)[:, None, :]
    asks = np.stack([ba[keep], aq[keep]], -1)[:, None, :]

    it = np.flatnonzero((df["action"] == "T").to_numpy()
                        & df["side"].isin(["B", "A"]).to_numpy())
    bi = np.searchsorted(keep, it, side="right") - 1
    ok = bi >= 0
    out = os.path.join(outdir, f"orderbook_{sym}_{day}.npz")
    np.savez_compressed(
        out, bids=bids, asks=asks,
        tr_price=pd.to_numeric(df["price"].iloc[it[ok]]).to_numpy(float),
        tr_qty=pd.to_numeric(df["size"].iloc[it[ok]]).to_numpy(float),
        tr_side=np.where(df["side"].iloc[it[ok]].to_numpy() == "B", 1, -1).astype(int),
        tr_bookidx=bi[ok].astype(int), ts=ts_ms[keep],
        symbol=f"{sym.upper()}-{day}")
    print(f"  {day}: {n:,} RTH events -> {m:,} snaps, {int(ok.sum()):,} trades -> {out}",
          flush=True)
    return out


def main(path, outdir=os.path.join("data", "l2")):
    os.makedirs(outdir, exist_ok=True)
    cur_day, parts, made = None, [], []
    for chunk in pd.read_csv(path, usecols=USECOLS, chunksize=CHUNK):
        te = chunk["ts_event"].astype(str)
        hm = te.str[11:16]
        chunk = chunk[(hm >= "13:30") & (hm < "20:00")]
        if len(chunk) == 0:
            continue
        for day, g in chunk.groupby(te.loc[chunk.index].str[:10], sort=False):
            if cur_day is not None and day != cur_day:
                p = flush_day(cur_day, parts, outdir)
                if p: made.append(p)
                parts = []
            cur_day = day
            parts.append(g)
    if parts:
        p = flush_day(cur_day, parts, outdir)
        if p: made.append(p)
    print(f"\n{len(made)} day files written")
    return made


if __name__ == "__main__":
    main(sys.argv[1])
