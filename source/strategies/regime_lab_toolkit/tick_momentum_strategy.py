"""Turns the conditional_return_dist.py FINDING into an actual traded strategy and
backtests it, no costs. This is the one piece from the whole session's search that
has held up out-of-sample with real statistical significance at the tick level
(GOOG/BTC/ETH survive the day-level SE correction) -- built from what's already been
validated, not a new invented signal.

Rule: position(t) = sign(mid[t] - mid[t-1])  (go the same direction as the tick that
just happened), held for exactly one tick, pnl(t) = position(t) * (mid[t+1] - mid[t]).
Shock ticks excluded via the same frozen sigma_ref as conditional_return_dist.py
(TRAIN-derived, frozen on TEST). Zero-current-tick bars are skipped (no signal).

No fees, no spread, no slippage -- answers "is there positive expected value here at
all before any execution friction," a strictly easier bar than "is it tradeable"
(already answered no, in the spread-comparison work: sub-spread on GOOG/SOL/XRP/NFTY,
plausibly sub-fee on BTC/ETH once real crypto exchange fee schedules are applied).
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from conditional_return_dist import tick_returns, SIGMA_MULT, GOOG_TRAIN_DAYS, CRYPTO_TRAIN_FILES, CRYPTO_WEEK


def sigma_ref_from(files):
    all_r = [tick_returns(fp) for fp in files]
    return float(np.concatenate(all_r).std())


def strategy_pnl_per_day(files, sigma_ref):
    """Per-day PnL arrays -- one array of per-trade pnls for each day-file, day
    boundaries respected (never trade across a day gap)."""
    day_pnls = []
    for fp in files:
        r = tick_returns(fp)
        cur = np.sign(r[:-1])
        nxt = r[1:]
        keep = (cur != 0) & (np.abs(nxt) <= SIGMA_MULT * sigma_ref)
        day_pnls.append(cur[keep] * nxt[keep])
    return day_pnls


def report(label, day_pnls):
    all_pnl = np.concatenate(day_pnls) if day_pnls else np.array([])
    day_means = np.array([p.mean() for p in day_pnls if len(p)])
    n = len(all_pnl)
    if n == 0:
        print(f"  {label}: no trades"); return
    hit = 100 * np.mean(all_pnl > 0)
    cum = all_pnl.sum()
    if len(day_means) >= 2:
        se = day_means.std(ddof=1) / np.sqrt(len(day_means))
        t = day_means.mean() / se if se > 0 else 0.0
        sd = day_means.std(ddof=1)
        sharpe = day_means.mean() / sd * np.sqrt(252) if sd > 0 else 0.0
    else:
        t, sharpe = float("nan"), float("nan")
    print(f"  {label:<22} n_trades={n:>7}  mean/trade={all_pnl.mean():+.5g}  "
          f"cum={cum:+.5g}  hit={hit:5.1f}%  day-level t={t:+.2f} (n_days={len(day_means)})  "
          f"ann.Sharpe~{sharpe:+.2f}")


def run_asset(name, train_files, test_files):
    sigma_ref = sigma_ref_from(train_files)
    train_pnls = strategy_pnl_per_day(train_files, sigma_ref)
    test_pnls = strategy_pnl_per_day(test_files, sigma_ref)
    print(f"\n{name} (sigma_ref={sigma_ref:.5g}, frozen from train):")
    report("TRAIN", train_pnls)
    report("TEST (frozen)", test_pnls)


def main():
    goog_files = sorted(glob.glob(os.path.join("data", "stock_data", "GOOG", "L2_top", "*", "*", "orderbook_goog_*.npz")))
    run_asset("GOOG", goog_files[:GOOG_TRAIN_DAYS], goog_files[GOOG_TRAIN_DAYS:])

    for sym in ("btcusdt", "ethusdt", "solusdt", "xrpusdt"):
        week = [os.path.join("data", "crypto_data", sym.upper(), d[:4], d[5:7], f"orderbook_{sym}_{d}.npz") for d in CRYPTO_WEEK]
        run_asset(sym.upper(), week[:CRYPTO_TRAIN_FILES], week[CRYPTO_TRAIN_FILES:])


if __name__ == "__main__":
    main()
