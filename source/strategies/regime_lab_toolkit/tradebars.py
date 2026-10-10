"""Trade-count bars: every N trades = one bar, price = mid at that Nth trade.

A third clock, distinct from both prior attempts: flow_momentum used calendar
time (5 min, wrong unit for assets trading at different speeds), the volume
probe used cumulative traded size (self-adapts to $-activity but still an
indirect proxy). Trade count is the most direct "trading resolution" clock --
N trades is N trades regardless of how much size or how long they took.

N (trades/bar) is calibrated per asset from TRAIN data only (small declared
grid), then frozen -- same discipline as every other time-sensitive parameter
in this program.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_lib"))
from l2_strategy import setup


def bucket_by_trade_count(fp: str, N: int):
    """One day-file -> bar-level mid/bb/ba/spread, one bar per N trades
    (chronological order via tr_bookidx). Returns None if too few trades.
    """
    D = setup(fp)
    raw = np.load(fp, allow_pickle=True)
    tbi = raw["tr_bookidx"].astype(int)
    if len(tbi) < N:
        return None
    order = np.argsort(tbi, kind="stable")
    bidx = tbi[order]
    n_bars = len(bidx) // N
    if n_bars < 1:
        return None
    bar_end_trade = np.arange(1, n_bars + 1) * N - 1     # index of the Nth,2Nth,... trade
    snap_idx = np.clip(bidx[bar_end_trade], 0, D["N"] - 1)
    return dict(mid=D["mid"][snap_idx], bb=D["bb"][snap_idx], ba=D["ba"][snap_idx],
               spread=D["spread"][snap_idx])


def concat_files(files: list, N: int):
    parts, day_ix = [], [0]
    for fp in files:
        b = bucket_by_trade_count(fp, N)
        if b is None:
            continue
        parts.append(b)
        day_ix.append(day_ix[-1] + len(b["mid"]))
    if not parts:
        return None
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    cat["day_ix"] = np.array(day_ix)
    return cat
