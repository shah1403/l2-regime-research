"""Walk-forward dynamic-quantile threshold -- the template for "nothing hand-tuned".

Generalization of `dynamic_thr` in scripts/l2_diagnose.py:31 (there hardcoded to
`|imb|`) to any signal array. Same algorithm, unchanged: block-updated every `step`
bars, each update using only the trailing `W` bars of STRICTLY PAST data -- causal
by construction, safe to use inside a test window without leaking future data.
"""
from __future__ import annotations

import numpy as np


def dynamic_quantile_thr(x: np.ndarray, W: int = 2000, q: float = 0.85, step: int = 25,
                         warm: int = 300, fallback: float = 0.4,
                         clip: tuple = (0.05, 0.95)) -> np.ndarray:
    """Rolling quantile of |x|, recomputed every `step` bars from the trailing `W`
    bars ending at the update point (never includes the current or future bars).
    Before `warm`, holds `fallback`. Returns an (n,) array, clipped to `clip`.
    """
    a = np.abs(x)
    n = len(a)
    th = np.full(n, fallback)
    for t in range(warm, n, step):
        s = max(0, t - W)
        th[t:t + step] = np.quantile(a[s:t], q)
    return np.clip(th, *clip)
