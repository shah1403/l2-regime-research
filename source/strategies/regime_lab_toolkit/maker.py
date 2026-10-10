"""Raw snapshot-scale primitives for directed passive quoting -- extracted from
the `nfty_dpq()` closure in scripts/curated_nfty_goog.py:155-224 (never
exported/reusable there). Deliberately operates on RAW top-of-book imbalance
(no forward-fill through invalid snapshots), matching the original bit-for-bit
-- this is NOT the same `imb` field l2_strategy.setup_arrays() returns (that
one IS forward-filled). Kept distinct on purpose so the regression check can
prove the extraction is faithful before anything is built on top of it.
"""
from __future__ import annotations

import numpy as np


def raw_imbalance(bids: np.ndarray, asks: np.ndarray):
    """Top-of-book imbalance straight from the raw arrays, valid mask requires
    a two-sided, non-crossed, nonzero-depth touch -- no ffill. Matches
    curated_nfty_goog.py:163-165,175-182 exactly."""
    bb = bids[:, 0, 0]
    ba = asks[:, 0, 0]
    bq = bids[:, 0, 1]
    aq = asks[:, 0, 1]
    valid = (bb > 0) & (ba > bb) & (bq > 0) & (aq > 0)
    imb = (bq - aq) / (bq + aq + 1e-9)
    return valid, bb, ba, bq, aq, imb


def trade_map(tr_price: np.ndarray, tr_side: np.ndarray, tr_bookidx: np.ndarray) -> dict:
    """snapshot index -> list of (price, side) trades printed in that snapshot."""
    tm = {}
    for p, s, b in zip(tr_price, tr_side, tr_bookidx):
        tm.setdefault(int(b), []).append((float(p), int(s)))
    return tm


def frozen_imbalance_threshold(files: list, q: float = 0.80) -> float:
    """Pooled |imb| quantile across the given (train) files -- frozen, reused
    unchanged on test. Matches curated_nfty_goog.py:160-166."""
    pooled = []
    for fp in files:
        r = np.load(fp, allow_pickle=True)
        valid, bb, ba, bq, aq, imb = raw_imbalance(r["bids"], r["asks"])
        pooled.append(np.abs(imb[valid]))
    return float(np.quantile(np.concatenate(pooled), q))
