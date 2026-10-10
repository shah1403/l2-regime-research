"""Shared bar-scale execution engine -- the piece that didn't exist yet.

Every current script (goog_minute_regime.py, adaptive_wh_long.py,
curated_nfty_goog.py, goog_flow_gated.py) reimplements its own ad hoc trading
loop. This is the one engine regime_lab code uses going forward; those scripts
are left untouched (regression-safety for their published numbers).

Convention (matches every existing loop in this repo): decide at bar t using
information known through t, fill at bar t+1's touch price, one position at a
time, fee charged proportional to notional*size, PnL in bps of mid at entry.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

FEE_BPS = 0.10   # matches scripts/goog_minute_regime.py's FEE_BPS default


def run_backtest(bars: dict, direction: np.ndarray, size: np.ndarray, lo: int,
                 hi: int, fee_bps: float = FEE_BPS, hold: Optional[int] = None,
                 early_exit: Optional[Callable[[int], bool]] = None) -> dict:
    """direction[t]/size[t] must be decidable from information known through bar
    t. `hold`: fixed hold in bars. `early_exit(t) -> bool`: optional dynamic exit
    check at bar t (e.g. "regime left DIR"); if both given, hold acts as a max.
    At least one of `hold`/`early_exit` is required.

    Returns dict(pnl, trades, per_trade_pnls, win_rate, tstat, equity).
    `equity` is a simple realized-PnL step function (jumps at trade close, not
    a full intra-trade markout) -- same simplification every existing script
    in this repo uses.
    """
    if hold is None and early_exit is None:
        raise ValueError("run_backtest needs `hold` (fixed) and/or `early_exit` (dynamic)")

    mid, bb, ba = bars["mid"], bars["bb"], bars["ba"]
    N = len(mid)
    per_trade = []
    equity = np.full(N, np.nan)
    cum = 0.0
    t = lo
    fixed_only = hold is not None and early_exit is None
    stop = (hi - hold - 2) if fixed_only else (hi - 2)

    while t < stop:
        d = direction[t]
        sz = size[t] if size[t] > 0 else 0.0
        if d != 0 and sz > 0:
            equity[t] = cum                       # mark equity as of entry (pre-trade)
            side = 1.0 if d > 0 else -1.0
            px_in = ba[t + 1] if side > 0 else bb[t + 1]
            if fixed_only:
                e = t + 1 + hold
            else:
                max_e = (t + 1 + hold) if hold is not None else (hi - 1)
                e = t + 1
                while e < max_e and not early_exit(e):
                    e += 1
                e = min(e, hi - 1)
            px_out = bb[e] if side > 0 else ba[e]
            pnl_bps = side * (px_out - px_in) / mid[t] * 1e4 * sz - 2 * fee_bps * sz
            per_trade.append(pnl_bps)
            cum += pnl_bps
            equity[t + 1:e + 1] = cum
            t = e + 1
        else:
            equity[t] = cum
            t += 1
    equity[t:hi] = cum

    per_trade = np.array(per_trade, dtype=float)
    if len(per_trade) == 0:
        return dict(pnl=0.0, trades=0, per_trade_pnls=per_trade, win_rate=0.0,
                    tstat=0.0, equity=equity)
    sh = per_trade.mean() / per_trade.std() if per_trade.std() > 0 else 0.0
    return dict(pnl=float(per_trade.sum()), trades=len(per_trade),
                per_trade_pnls=per_trade, win_rate=float(100 * np.mean(per_trade > 0)),
                tstat=float(sh * np.sqrt(len(per_trade))), equity=equity)


def stat_line(r: dict) -> str:
    """Same print format used throughout this repo's scripts, for easy eyeball
    comparison against known numbers."""
    if r["trades"] == 0:
        return "0 trades"
    return (f"{r['pnl']:+9.1f} bps | {r['trades']:4d} tr, "
            f"{r['pnl']/r['trades']:+7.2f}/tr, win {r['win_rate']:4.1f}%, "
            f"t {r['tstat']:+5.2f}")
