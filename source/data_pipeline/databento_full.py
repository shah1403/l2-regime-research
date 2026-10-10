"""Extract FULL-resolution top-of-book event streams from a big MBP-10 CSV.

No sampling: every RTH book event's touch (bb,ba,bq,aq,ts) per day, float32
prices/sizes (~60MB/day), written to data/l2full/<sym>_<day>.npz. This is the
input for tick-cadence strategy work (the sampled data/l2 npz stay as-is for
the snapshot stack).

Usage: python scripts/databento_full.py <big.mbp-10.csv>
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

USECOLS = ["ts_event", "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00", "symbol"]
CHUNK = 2_000_000
OUT = os.path.join("data", "l2full")


def flush(day, parts):
    df = pd.concat(parts, ignore_index=True)
    sym = str(df["symbol"].iloc[0]).lower()
    bb = pd.to_numeric(df["bid_px_00"], errors="coerce").fillna(0).to_numpy(np.float64)
    ba = pd.to_numeric(df["ask_px_00"], errors="coerce").fillna(0).to_numpy(np.float64)
    bq = pd.to_numeric(df["bid_sz_00"], errors="coerce").fillna(0).to_numpy(np.float64)
    aq = pd.to_numeric(df["ask_sz_00"], errors="coerce").fillna(0).to_numpy(np.float64)
    ts = (pd.to_datetime(df["ts_event"], format="ISO8601").astype("int64")
          // 10**6).to_numpy(np.int64)
    ok = (bb > 0) & (ba > bb) & (bq > 0) & (aq > 0)
    p = os.path.join(OUT, f"{sym}_{day}.npz")
    np.savez_compressed(p, bb=bb[ok].astype(np.float32), ba=ba[ok].astype(np.float32),
                        bq=bq[ok].astype(np.float32), aq=aq[ok].astype(np.float32),
                        ts=np.maximum.accumulate(ts[ok]))
    print(f"  {day}: {int(ok.sum()):,} valid events -> {p}", flush=True)


def main(path):
    os.makedirs(OUT, exist_ok=True)
    cur, parts = None, []
    for chunk in pd.read_csv(path, usecols=USECOLS, chunksize=CHUNK):
        te = chunk["ts_event"].astype(str)
        hm = te.str[11:16]
        chunk = chunk[(hm >= "13:30") & (hm < "20:00")]
        if len(chunk) == 0:
            continue
        for day, g in chunk.groupby(te.loc[chunk.index].str[:10], sort=False):
            if cur is not None and day != cur:
                flush(cur, parts); parts = []
            cur = day
            parts.append(g)
    if parts:
        flush(cur, parts)


if __name__ == "__main__":
    main(sys.argv[1])
