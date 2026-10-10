"""Regression check: the extracted maker engine must reproduce the known NFTY
directed-passive-quoting number from curated_nfty_goog.py's nfty_dpq()
(TRAIN_DAYS=5/test=15, threshold = train q80 |imb|, hold 30min):
  TEST total: +39.0 bps | 2 trades, win 100%, t+2.68
  daily: mean +2.6 bps/day, Sharpe(d) +0.34 (t+1.3), positive days 13%
Small sample (2 trades) on purpose -- this asset fires rarely by design. Any
deviation here means the extraction has a bug, not a new finding.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maker import frozen_imbalance_threshold
from maker_backtest import run_maker_day, stat_line

TRAIN_DAYS = 5
TOL_BPS = 0.5
TOL_TRADES = 0


def main():
    files = sorted(glob.glob(os.path.join("data", "stock_data", "NFTY_ETF", "*", "*", "orderbook_nfty_20*.npz")))
    print(f"{len(files)} NFTY day files (expect 20)")
    train_files, test_files = files[:TRAIN_DAYS], files[TRAIN_DAYS:]
    thr = frozen_imbalance_threshold(train_files, q=0.80)
    print(f"frozen threshold |imb| > {thr:.2f} (train q80)")

    daily_totals = []
    all_trades = []
    for fp in test_files:
        r = run_maker_day(fp, thr, hold_ms=30 * 60_000, fee_bps=0.0)
        day = os.path.basename(fp).split("_")[-1][:10]
        daily_totals.append(r["total"])
        all_trades.extend(r["pnl_bps"].tolist())
        print(f"  {day}: {r['total']:+8.1f} bps  ({r['trades']} round trips)")

    d = np.array(daily_totals)
    all_trades = np.array(all_trades)
    print(f"\n{stat_line(d)}")

    exp_total, exp_n = 39.0, 2
    total = float(all_trades.sum())
    ok = abs(total - exp_total) < TOL_BPS and len(all_trades) == exp_n
    print(f"\nexpected: +{exp_total} bps, {exp_n} trades  ->  {'PASS' if ok else 'FAIL'}")
    print(f"PHASE 5b GATE: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
