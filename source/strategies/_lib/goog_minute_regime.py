"""Regime + kinematic DIR strategy on GOOG at MINUTE scale -- where the moves are.

The prior failures traded seconds-scale signals against bps-scale costs. This
runs the SAME architecture (VR/autocorr regime detector with its economic gate,
polynomial d1/d2 conviction ladder) on 1/5-minute bars, where GOOG moves
50-300+ bps and the ~2 bps spread is amortized over hours-long holds.

STRICT protocol (no overfitting):
  * train = first 5 trading days, test = remaining 15. Hard wall.
  * ALL free parameters chosen on train only: the small declared grid
    (bar {1,5}min x kinematic window {30,60} x z1 gate quantile {0.6,0.8} x
    regime cost_mult {0.5,1.0}) AND the z-score thresholds themselves
    (quantiles of |z1|,|z2| computed over train bars only, frozen).
  * one single pass over test with the frozen config. Every train-grid row is
    printed so the selection is auditable.
  * baselines on the same test window: buy&hold, and d1-sign-only (no regime).

Execution: decide at bar t close, fill at bar t+1 touch (buy at ask, sell at
bid -- spread paid naturally), positions may carry overnight (you hold shares),
0.10 bps/side fee. PnL in bps of median mid.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view as swv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_lib"))
from l2_strategy import compute_regime, _proj

TRAIN_DAYS = 5
FEE_BPS = 0.10
REG_W = 100
WARMB = 150
GRID = dict(bar=[1, 5], W=[30, 60], q1=[0.6, 0.8], cm=[0.5, 1.0])
MIN_TRAIN_TRADES = 8


# ------------------------- bars -------------------------
def day_bars(fp_full, fp_snap, bar_min):
    r = np.load(fp_full)
    bb, ba, ts = r["bb"].astype(np.float64), r["ba"].astype(np.float64), r["ts"]
    step = bar_min * 60_000
    bounds = np.arange(ts[0] + step, ts[-1] + step, step)
    idx = np.searchsorted(ts, bounds, side="right") - 1
    keep = np.concatenate([[True], np.diff(idx) > 0]) & (idx >= 0)
    idx, bounds = idx[keep], bounds[keep]
    s = np.load(fp_snap, allow_pickle=True)
    tts = s["ts"][s["tr_bookidx"].astype(int)]
    fl = s["tr_side"].astype(float) * s["tr_qty"].astype(float)
    bkt = np.searchsorted(bounds, tts, side="left")
    flow = np.bincount(np.clip(bkt, 0, len(bounds) - 1), weights=fl,
                       minlength=len(bounds))
    return dict(bb=bb[idx], ba=ba[idx], flow=flow, ts=bounds)


def build(bar_min):
    fulls = sorted(glob.glob(os.path.join("data", "stock_data", "GOOG", "L2_full", "*", "*", "goog_*.npz")))
    snaps = sorted(glob.glob(os.path.join("data", "stock_data", "GOOG", "L2_top", "*", "*", "orderbook_goog_*.npz")))
    parts, day_ix = [], [0]
    for ff, fs in zip(fulls, snaps):
        parts.append(day_bars(ff, fs, bar_min))
        day_ix.append(day_ix[-1] + len(parts[-1]["bb"]))
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    cat["mid"] = (cat["bb"] + cat["ba"]) / 2
    cat["spread"] = cat["ba"] - cat["bb"]
    cat["day_ix"] = np.array(day_ix)          # bar index where each day starts
    return cat


# ------------------------- signals -------------------------
def kinematics(mid, W):
    N = len(mid)
    C = swv(mid, W) @ _proj(W, 2, W / 2).T
    d1 = np.full(N, np.nan); d2 = np.full(N, np.nan)
    d1[W - 1:] = C[:, 1]; d2[W - 1:] = 2 * C[:, 2]
    r = np.diff(mid, prepend=mid[0])
    var = np.convolve(r * r, np.ones(100) / 100, "full")[:N]
    vol = np.sqrt(np.maximum(var, 1e-18))
    return np.nan_to_num(d1 / (vol + 1e-12)), \
           np.nan_to_num(d2 / (vol / np.sqrt(W) + 1e-12))


def targets(z1, z2, thr):
    t1, t2c, t2p = thr
    s = np.sign(z1) * (np.abs(z1) > t1)
    agree = np.sign(z2) == np.sign(z1)
    peak = (s != 0) & ~agree & (np.abs(z2) > t2p)
    accel = (s != 0) & agree & (np.abs(z2) > t2c) & ~peak
    size = np.where(accel, 2.0, np.where(peak, 0.0, 1.0))
    return s * size


def run(bars, reg, tgt, lo, hi, use_regime=True):
    """Trade bars [lo,hi); returns equity (bps), per-episode pnls (bps)."""
    mid, bb, ba = bars["mid"], bars["bb"], bars["ba"]
    med = np.median(mid[lo:hi])
    cash = 0.0; pos = 0.0; eq = np.zeros(hi - lo); eps = []; entry_eq = 0.0
    for t in range(lo, hi - 1):
        want = tgt[t] if (reg[t] == 1 or not use_regime) else 0.0
        if want != pos:
            px = ba[t + 1] if want > pos else bb[t + 1]
            cash -= (want - pos) * px + abs(want - pos) * px * FEE_BPS / 1e4
            cur_eq = cash + want * mid[t + 1]
            if pos == 0:
                entry_eq = cur_eq
            elif want == 0 or np.sign(want) != np.sign(pos):
                eps.append((cur_eq - entry_eq) / med * 1e4); entry_eq = cur_eq
            pos = want
        eq[t - lo] = (cash + pos * mid[t]) / med * 1e4
    if pos != 0:                                  # mark final open position
        cash += pos * (bb[hi - 1] if pos > 0 else ba[hi - 1])
        eps.append((cash - entry_eq) / med * 1e4)
        pos = 0
    eq[-1] = cash / med * 1e4
    return eq, np.array(eps)


def stat(eps, total):
    if len(eps) == 0:
        return "0 trades"
    sh = eps.mean() / eps.std() if eps.std() > 0 else 0
    return (f"{total:+8.1f} bps | {len(eps):3d} tr, {eps.mean():+7.1f}/tr, "
            f"win {100 * np.mean(eps > 0):4.1f}%, t {sh * np.sqrt(len(eps)):+5.2f}")


def main():
    results = []
    cache = {}
    for bar in GRID["bar"]:
        bars = build(bar); cache[bar] = bars
        train_end = bars["day_ix"][TRAIN_DAYS]
        for cm in GRID["cm"]:
            reg = compute_regime(bars["mid"], bars["spread"], bars["flow"],
                                 W=REG_W, cost_mult=cm)
            for W in GRID["W"]:
                z1, z2 = kinematics(bars["mid"], W)
                for q1 in GRID["q1"]:
                    tr_sl = slice(WARMB, train_end)
                    thr = (np.quantile(np.abs(z1[tr_sl]), q1),
                           np.quantile(np.abs(z2[tr_sl]), 0.60),
                           np.quantile(np.abs(z2[tr_sl]), 0.85))
                    tgt = targets(z1, z2, thr)
                    eq, eps = run(bars, reg, tgt, WARMB, train_end)
                    results.append(dict(bar=bar, cm=cm, W=W, q1=q1, thr=thr,
                                        total=eq[-1], n=len(eps), eps=eps))
                    print(f"  TRAIN bar={bar}m cm={cm} W={W} q1={q1}: "
                          f"{stat(eps, eq[-1])}", flush=True)

    ok = [r for r in results if r["n"] >= MIN_TRAIN_TRADES]
    if not ok:
        print("no config reached the minimum trade count on train"); return
    best = max(ok, key=lambda r: r["total"])
    print(f"\n=== FROZEN: bar={best['bar']}m cost_mult={best['cm']} "
          f"W={best['W']} q1={best['q1']} (train {best['total']:+.1f} bps) ===\n")

    bars = cache[best["bar"]]
    train_end = bars["day_ix"][TRAIN_DAYS]; N = len(bars["mid"])
    reg = compute_regime(bars["mid"], bars["spread"], bars["flow"],
                         W=REG_W, cost_mult=best["cm"])
    z1, z2 = kinematics(bars["mid"], best["W"])
    tgt = targets(z1, z2, best["thr"])            # thresholds frozen from train
    eq, eps = run(bars, reg, tgt, train_end, N)
    print(f"TEST (15 days, one pass) : {stat(eps, eq[-1])}")
    eq_b, eps_b = run(bars, reg, tgt, train_end, N, use_regime=False)
    print(f"  baseline d1-only        : {stat(eps_b, eq_b[-1])}")
    med = np.median(bars["mid"][train_end:])
    bh = (bars["mid"][N - 1] - bars["mid"][train_end]) / med * 1e4
    print(f"  baseline buy & hold     : {bh:+.1f} bps")
    dirpct = 100 * np.mean(reg[train_end:] == 1)
    print(f"  DIR occupancy on test   : {dirpct:.1f}%")

    # ---- plot: mid + position shading, equity below ----
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True,
                                 height_ratios=[2, 1])
    x = np.arange(N)
    a1.plot(x, bars["mid"], lw=0.7, color="#1f77b4")
    pos_sh = np.zeros(N)
    posv = 0.0
    for t in range(WARMB, N - 1):
        want = tgt[t] if reg[t] == 1 else 0.0
        if want != posv: posv = want
        pos_sh[t + 1] = posv
    a1.fill_between(x, bars["mid"].min(), bars["mid"].max(),
                    where=pos_sh > 0, color="green", alpha=0.10)
    a1.fill_between(x, bars["mid"].min(), bars["mid"].max(),
                    where=pos_sh < 0, color="red", alpha=0.10)
    a1.axvline(train_end, color="k", lw=1.5, ls="--")
    a1.set_title(f"GOOG {best['bar']}-min bars — mid + position shading "
                 f"(green long / red short); dashed line = train/test wall")
    a1.set_ylabel("$")
    full_eq = np.concatenate([np.zeros(train_end), eq])
    a2.plot(x, full_eq, color="#111", lw=1.0, label="strategy (test only)")
    mid_reb = (bars["mid"] - bars["mid"][train_end]) / med * 1e4
    a2.plot(x, mid_reb, color="#1f77b4", lw=0.7, alpha=0.7, label="buy & hold")
    a2.axvline(train_end, color="k", lw=1.5, ls="--")
    a2.axhline(0, color="gray", lw=0.5)
    a2.set_ylabel("bps"); a2.legend(fontsize=8)
    for d in bars["day_ix"]:
        a1.axvline(d, color="gray", lw=0.3, alpha=0.5)
    fig.tight_layout()
    out = os.path.join("reports", "l2_backtest", "GOOG_minute_regime.png")
    fig.savefig(out, dpi=110)
    print(f"\nplot -> {out}")


if __name__ == "__main__":
    main()
