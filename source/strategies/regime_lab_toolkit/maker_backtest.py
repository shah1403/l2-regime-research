"""Maker (passive-quote) execution engine -- generalizes the per-day loop
inside curated_nfty_goog.py's nfty_dpq() (lines 185-213) into a reusable,
parameterized function. This is a DIFFERENT execution model from
backtest.run_backtest (which always crosses the spread as a taker): entry and
exit only fill when a trade actually prints AT-OR-THROUGH your posted price
(touch-fill bound -- optimistic, since it assumes front-of-queue; the
pessimistic "price must trade through" bound would give zero fills on a book
this sparse, as established earlier in this program).

Operates at raw SNAPSHOT scale (every book update), not calendar bars --
passive fills require checking every individual trade against the posted
price, which calendar-bar aggregation would destroy. This is why it's a
separate engine rather than a variant of run_backtest, and why it doesn't
plug into run_assembler's bar-indexed loop without further work (flagged, not
silently forced).
"""
from __future__ import annotations

import numpy as np

from maker import raw_imbalance, trade_map

DEFAULT_HOLD_MS = 30 * 60_000


def run_maker_day(fp: str, thr: float, hold_ms: int = DEFAULT_HOLD_MS,
                  fee_bps: float = 0.0) -> dict:
    """One day-file, directed passive quoting: join the touch on whichever
    side extreme imbalance favors, hold `hold_ms`, passive exit on the
    opposite touch, day-end taker flatten as a backstop. Returns
    dict(pnl_bps list per round trip, total, day label unset here).
    """
    r = np.load(fp, allow_pickle=True)
    valid, bb, ba, bq, aq, imb = raw_imbalance(r["bids"], r["asks"])
    ts = r["ts"].astype(np.int64)
    tm = trade_map(r["tr_price"], r["tr_side"], r["tr_bookidx"])
    mid = (bb + ba) / 2
    med = np.median(mid[valid]) if valid.any() else np.median(mid)
    n = len(bb)
    fee = 2 * fee_bps

    inv = 0
    entry = 0.0
    entry_ts = 0
    pnl_day = []
    for t in range(n - 1):
        if not valid[t]:
            continue
        trades = tm.get(t, ())
        if inv == 0:
            if imb[t] > thr:
                for tp, sd in trades:
                    if tp <= bb[t] and sd < 0:
                        inv = 1; entry = bb[t]; entry_ts = ts[t]; break
            elif imb[t] < -thr:
                for tp, sd in trades:
                    if tp >= ba[t] and sd > 0:
                        inv = -1; entry = ba[t]; entry_ts = ts[t]; break
        else:
            if ts[t] - entry_ts >= hold_ms:
                if inv > 0:
                    for tp, sd in trades:
                        if tp >= ba[t] and sd > 0:
                            pnl_day.append((ba[t] - entry) / med * 1e4 - fee)
                            inv = 0; break
                else:
                    for tp, sd in trades:
                        if tp <= bb[t] and sd < 0:
                            pnl_day.append((entry - bb[t]) / med * 1e4 - fee)
                            inv = 0; break
    if inv != 0:                                    # day-end taker flatten
        valid_idx = np.flatnonzero(valid)
        px = bb[valid_idx[-1]] if inv > 0 else ba[valid_idx[-1]]
        pnl = (px - entry) / med * 1e4 - fee if inv > 0 else (entry - px) / med * 1e4 - fee
        pnl_day.append(pnl)

    pnl_day = np.array(pnl_day, dtype=float)
    return dict(pnl_bps=pnl_day, total=float(pnl_day.sum()), trades=len(pnl_day),
               win_rate=float(100 * np.mean(pnl_day > 0)) if len(pnl_day) else 0.0)


def stat_line(files_totals: np.ndarray) -> str:
    if len(files_totals) == 0:
        return "0 days"
    sh = files_totals.mean() / files_totals.std() if files_totals.std() > 0 else 0.0
    return (f"{files_totals.sum():+9.1f} bps total | {len(files_totals)} days, "
            f"mean {files_totals.mean():+.2f}/day, Sharpe(d) {sh:+.2f} "
            f"(t={sh * np.sqrt(len(files_totals)):+.2f}), "
            f"pos days {100 * np.mean(files_totals > 0):.0f}%")
