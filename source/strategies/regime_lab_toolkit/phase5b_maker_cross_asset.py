"""Cross-asset check for directed passive quoting, from the start this time
(not NFTY-only-then-extend): same frozen-threshold protocol (train q80 |imb|,
30min hold, touch-fill bound) applied to GOOG and the four crypto symbols
alongside NFTY. Two questions: (1) does it make money anywhere besides NFTY,
and (2) does it naturally stay quiet on tight-spread liquid names (the
assembler's making-category applicability assumes wide-spread-favors-making;
this checks whether that assumption is actually true rather than just
plausible).
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maker import frozen_imbalance_threshold, raw_imbalance
from maker_backtest import run_maker_day, stat_line

GOOG_TRAIN_DAYS = 5
CRYPTO_TRAIN_FILES = 2   # first 2 files (2023-09-01, 02) of each symbol's set
CRYPTO_SYMS = ["btcusdt", "ethusdt", "solusdt", "xrpusdt"]


def median_spread_bps(fp):
    r = np.load(fp, allow_pickle=True)
    valid, bb, ba, bq, aq, imb = raw_imbalance(r["bids"], r["asks"])
    if not valid.any():
        return float("nan")
    mid = (bb + ba) / 2
    return float(np.median((ba[valid] - bb[valid]) / mid[valid]) * 1e4)


def run_asset(name, files, train_n):
    train_files, test_files = files[:train_n], files[train_n:]
    if len(test_files) == 0:
        print(f"{name:<12} no test files"); return
    thr = frozen_imbalance_threshold(train_files, q=0.80)
    sp = np.nanmedian([median_spread_bps(fp) for fp in test_files])
    totals = []
    for fp in test_files:
        r = run_maker_day(fp, thr, hold_ms=30 * 60_000, fee_bps=0.0)
        totals.append(r["total"])
    d = np.array(totals)
    print(f"{name:<12} spread {sp:6.2f}bps  thr={thr:.2f}  {stat_line(d)}")


def main():
    print("=" * 96)
    print("Directed passive quoting, SAME frozen-per-asset protocol, all assets in one pass")
    print("=" * 96)

    goog_files = sorted(glob.glob(os.path.join("data", "stock_data", "GOOG", "L2_top", "*", "*", "orderbook_goog_*.npz")))
    run_asset("GOOG", goog_files, GOOG_TRAIN_DAYS)

    for sym in CRYPTO_SYMS:
        files = sorted(glob.glob(os.path.join("data", "crypto_data", sym.upper(), "*", "*", f"orderbook_{sym}_*.npz")))
        run_asset(sym.upper(), files, CRYPTO_TRAIN_FILES)

    nfty_files = sorted(glob.glob(os.path.join("data", "stock_data", "NFTY_ETF", "*", "*", "orderbook_nfty_*.npz")))
    run_asset("NFTY", nfty_files, GOOG_TRAIN_DAYS)


if __name__ == "__main__":
    main()
