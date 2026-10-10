"""Fetch real Binance USDT-M futures tick data and convert to backtest .npz.

Sources (free, no API key): https://data.binance.vision
  * bookTicker : every best-bid/ask update (price+size both sides) --
                 ONLY published 2023-05-16 .. 2024-03-30, ~140MB/day zipped
  * aggTrades  : every trade with aggressor flag, 2020 .. now

The strategy is L1-driven (imbalance/validation use only the touch), so
bookTicker + aggTrades reproduce it faithfully with K=1 depth.

Per (symbol, date): stream the bookTicker zip and keep every k-th update so a
day collapses to ~TARGET_SNAPS event-driven snapshots; attach the FULL trade
tape (signed flow uses all prints, mapped to the enclosing snapshot); write
data/l2/orderbook_<symbol>_<date>.npz. Zips are cached in %TEMP% (not OneDrive)
and deleted after conversion.
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd

BASE = "https://data.binance.vision/data/futures/um/daily"
OUT_DIR = os.path.join("data", "l2")
TMP_DIR = os.path.join(tempfile.gettempdir(), "binance_dumps")
TARGET_SNAPS = 20_000

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]
DATES = ["2023-06-01", "2023-09-01", "2023-12-01", "2024-03-01"]  # bookTicker era


def download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        got = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk); got += len(chunk)
        print(f"    {got/1e6:7.1f} MB in {time.time()-t0:5.1f}s  {url.rsplit('/',1)[-1]}",
              flush=True)
        if total and got != total:
            raise IOError(f"short read {got}/{total}")
    os.replace(tmp, dest)
    return dest


def sample_bookticker(zip_path, target=TARGET_SNAPS):
    """Stream the bookTicker CSV, keep every k-th update. Returns dict of arrays."""
    with zipfile.ZipFile(zip_path) as z:
        info = z.infolist()[0]
        stride = max(1, int(info.file_size / 70 / target))
        kept = []
        with z.open(info) as f:
            txt = io.TextIOWrapper(f, encoding="utf-8", newline="")
            first = txt.readline()
            i = 0
            if first[:1].isdigit():          # no header row
                kept.append(first); i = 1
            for line in txt:
                if i % stride == 0:
                    kept.append(line)
                i += 1
    # columns: update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,
    #          transaction_time,event_time
    a = np.array([k.rstrip("\n").split(",") for k in kept])
    return dict(bid_px=a[:, 1].astype(float), bid_qty=a[:, 2].astype(float),
                ask_px=a[:, 3].astype(float), ask_qty=a[:, 4].astype(float),
                ts=a[:, 6].astype(np.int64), n_updates=i)


def load_aggtrades(zip_path):
    df = pd.read_csv(zip_path)
    if "price" not in df.columns:            # header-less variant
        df = pd.read_csv(zip_path, header=None,
                         names=["agg_trade_id", "price", "quantity", "first_trade_id",
                                "last_trade_id", "transact_time", "is_buyer_maker"])
    return df


def convert(symbol, date):
    out = os.path.join(OUT_DIR, f"orderbook_{symbol.lower()}_{date}.npz")
    if os.path.exists(out):
        print(f"  [skip] {out} exists"); return out
    print(f"  {symbol} {date}", flush=True)
    bt_zip = download(f"{BASE}/bookTicker/{symbol}/{symbol}-bookTicker-{date}.zip",
                      os.path.join(TMP_DIR, f"{symbol}-bookTicker-{date}.zip"))
    tr_zip = download(f"{BASE}/aggTrades/{symbol}/{symbol}-aggTrades-{date}.zip",
                      os.path.join(TMP_DIR, f"{symbol}-aggTrades-{date}.zip"))

    bk = sample_bookticker(bt_zip)
    N = len(bk["ts"])
    bids = np.stack([bk["bid_px"], bk["bid_qty"]], -1)[:, None, :]   # (N,1,2)
    asks = np.stack([bk["ask_px"], bk["ask_qty"]], -1)[:, None, :]

    df = load_aggtrades(tr_zip)
    tt = df["transact_time"].to_numpy(np.int64)
    idx = np.searchsorted(bk["ts"], tt, side="right") - 1
    keep = idx >= 0
    tr_side = np.where(df["is_buyer_maker"].to_numpy(bool), -1, 1)  # maker-buy => aggressor sell

    np.savez_compressed(
        out,
        bids=bids, asks=asks,
        tr_price=df["price"].to_numpy(float)[keep],
        tr_qty=df["quantity"].to_numpy(float)[keep],
        tr_side=tr_side[keep].astype(int),
        tr_bookidx=idx[keep].astype(int),
        ts=bk["ts"], symbol=f"{symbol}-{date}",
    )
    print(f"    -> {out}: {N} snaps (from {bk['n_updates']:,} book updates), "
          f"{int(keep.sum()):,} trades", flush=True)
    for p in (bt_zip, tr_zip):               # free the temp space
        try: os.remove(p)
        except OSError: pass
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)
    args = sys.argv[1:]
    symbols = args[0].split(",") if args else SYMBOLS
    dates = args[1].split(",") if len(args) > 1 else DATES
    done, failed = [], []
    for d in dates:
        for s in symbols:
            try:
                done.append(convert(s, d))
            except Exception as e:
                print(f"    !! {s} {d}: {type(e).__name__}: {e}", flush=True)
                failed.append((s, d))
    print(f"\n{len(done)} datasets ready, {len(failed)} failed {failed if failed else ''}")


if __name__ == "__main__":
    main()
