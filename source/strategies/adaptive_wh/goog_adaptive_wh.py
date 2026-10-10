"""Adaptive (W,O,H) drift strategy on GOOG 1-min bars -- the user's original
edge-selected kinematic machine at the minute scale.

  fit windows  W : {10, 40, 90, 120} minutes (weighted polyfit, lambda=W/2)
  orders       O : {0, 1, 2}   (0 = reversion-to-weighted-mean, 1 = linear
                                trend, 2 = curvature extrapolation)
  horizons     H : {1, 5, 10, 25} minutes
  predicted move over H = poly(tau=H) - CURRENT mid (uniform for all orders)
  selector: per bar, the combo with best trailing realized edge over L=150
            bars, net of median spread (walk-forward by construction -- an
            edge for combo (.,.,H) is only known H bars after its prediction)

Trading (all declared, nothing fitted on test):
  * regime gate: compute_regime, cost_mult=0.5 (frozen from the prior
    train-day selection), default detector params, W=100 bars
  * entry: reg==DIR and |predicted move| > current spread (economic gate)
  * hold: the SELECTED combo's own horizon H (adaptive), early exit on
    regime leaving DIR; fills at next bar's touch; 0.10 bps/side fee
  * size 1, one position at a time

Strict wall: days 1-5 = train report only, days 6-20 = single-pass test.
Baselines: buy&hold, ungated variant. Also prints the selector's (W,O,H)
usage histogram so you can see what it chose to be.
"""
from __future__ import annotations

import os
import sys
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view as swv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_lib"))
from goog_minute_regime import build, TRAIN_DAYS, REG_W, FEE_BPS
from l2_strategy import compute_regime, _proj

W_LIST = [10, 40, 90, 120]
O_LIST = [0, 1, 2]
H_LIST = [1, 5, 10, 25]
L = 150
COST_MULT = 0.5
WARM = max(W_LIST) + max(H_LIST) + L + 10


def adaptive_drift(mid, spread):
    """Returns disp (predicted move, $), H_sel, sel index, combos list."""
    N = len(mid); cost = np.median(spread)
    combos, disps = [], []
    for W in W_LIST:
        wv = swv(mid, W)
        for O in O_LIST:
            C = np.full((N, O + 1), np.nan)
            C[W - 1:] = wv @ _proj(W, O, W / 2).T
            for H in H_LIST:
                fut = np.array([float(H) ** k for k in range(O + 1)])
                pred = C @ fut
                combos.append((W, O, H))
                disps.append(np.nan_to_num(pred - mid))
    D = np.vstack(disps)
    recent = np.full((len(combos), N), -1e9)
    for i, (W, O, H) in enumerate(combos):
        rz = np.full(N, np.nan); rz[:N - H] = mid[H:] - mid[:N - H]
        e = np.nan_to_num(np.sign(D[i]) * rz - cost)
        cs = np.cumsum(e)
        recent[i, H + L:] = cs[L:N - H] - cs[:N - H - L]
    sel = np.argmax(recent, axis=0)
    ar = np.arange(N)
    return D[sel, ar], np.array([combos[s][2] for s in sel]), sel, combos


def trade(bars, reg, disp, Hs, lo, hi, use_regime=True):
    mid, bb, ba, sp = bars["mid"], bars["bb"], bars["ba"], bars["spread"]
    med = np.median(mid[lo:hi])
    cash = 0.0; pos = 0.0; hold_until = -1
    eq = np.full(len(mid), np.nan); eps = []; entries = []; exits = []
    entry_eq = 0.0
    for t in range(lo, hi - 1):
        if pos == 0:
            gate = (reg[t] == 1 or not use_regime)
            if gate and abs(disp[t]) > sp[t]:
                side = np.sign(disp[t])
                px = ba[t + 1] if side > 0 else bb[t + 1]
                cash -= side * px + px * FEE_BPS / 1e4
                pos = side; hold_until = t + 1 + int(Hs[t])
                entry_eq = cash + pos * mid[t + 1]
                entries.append((t + 1, px, side))
        else:
            if t >= hold_until or (reg[t] != 1 and use_regime):
                px = bb[t + 1] if pos > 0 else ba[t + 1]
                cash += pos * px - px * FEE_BPS / 1e4
                pnl = (cash - entry_eq) / med * 1e4
                eps.append(pnl); exits.append((t + 1, px, pnl)); pos = 0
        eq[t] = (cash + pos * mid[t]) / med * 1e4
    if pos != 0:
        px = bb[hi - 1] if pos > 0 else ba[hi - 1]
        cash += pos * px
        eps.append((cash - entry_eq) / med * 1e4)
    eq[hi - 1] = cash / med * 1e4
    return eq, np.array(eps), entries, exits


def stat(eps, total):
    if len(eps) == 0: return "0 trades"
    sh = eps.mean() / eps.std() if eps.std() > 0 else 0
    return (f"{total:+8.1f} bps | {len(eps):3d} tr, {eps.mean():+6.1f}/tr, "
            f"win {100 * np.mean(eps > 0):4.1f}%, t {sh * np.sqrt(len(eps)):+5.2f}")


def main():
    bars = build(1)
    N = len(bars["mid"]); wall = bars["day_ix"][TRAIN_DAYS]
    reg = compute_regime(bars["mid"], bars["spread"], bars["flow"],
                         W=REG_W, cost_mult=COST_MULT)
    disp, Hs, sel, combos = adaptive_drift(bars["mid"], bars["spread"])

    eq_tr, eps_tr, _, _ = trade(bars, reg, disp, Hs, WARM, wall)
    print(f"TRAIN (days 1-5)         : {stat(eps_tr, eq_tr[wall - 1])}")
    eq, eps, entries, exits = trade(bars, reg, disp, Hs, wall, N)
    print(f"TEST  (days 6-20, frozen): {stat(eps, eq[N - 1])}")
    eq_u, eps_u, _, _ = trade(bars, reg, disp, Hs, wall, N, use_regime=False)
    print(f"  ungated baseline        : {stat(eps_u, eq_u[N - 1])}")
    med = np.median(bars["mid"][wall:])
    bh = (bars["mid"][N - 1] - bars["mid"][wall]) / med * 1e4
    print(f"  buy & hold              : {bh:+.1f} bps")
    print(f"  DIR occupancy test      : {100 * np.mean(reg[wall:] == 1):.1f}%")

    cnt = Counter(combos[s] for s in sel[wall:])
    print("\nselector usage on test (W,O,H -> % of bars):")
    for (w, o, h), c in cnt.most_common(10):
        print(f"  W={w:>3} O={o} H={h:>2} : {100 * c / (N - wall):5.1f}%")

    # ---- dashboard ----
    x = np.arange(N)
    fig = plt.figure(figsize=(17, 13))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.3, 1.3, 1.0], hspace=0.32)
    aA = fig.add_subplot(gs[0]); aB = fig.add_subplot(gs[1], sharex=aA)
    aD = fig.add_subplot(gs[2])
    ylim = (bars["mid"].min() - 1, bars["mid"].max() + 1)
    aA.fill_between(x, *ylim, where=reg == 1, color="#2ca02c", alpha=0.15)
    aA.fill_between(x, *ylim, where=reg == -1, color="#d62728", alpha=0.08)
    aA.plot(x, bars["mid"], lw=0.6, color="#1f77b4")
    eL = [(t, p) for t, p, s in entries if s > 0]
    eS = [(t, p) for t, p, s in entries if s < 0]
    if eL: aA.scatter(*zip(*eL), marker="^", s=70, color="#2ca02c",
                      edgecolors="k", zorder=5, label=f"long ({len(eL)})")
    if eS: aA.scatter(*zip(*eS), marker="v", s=70, color="#d62728",
                      edgecolors="k", zorder=5, label=f"short ({len(eS)})")
    if exits: aA.scatter([t for t, _, _ in exits], [p for _, p, _ in exits],
                         marker="x", s=45, color="k", zorder=5,
                         label=f"exit ({len(exits)})")
    aA.axvline(wall, color="k", ls="--", lw=1.8)
    aA.set_ylim(*ylim); aA.set_ylabel("$"); aA.legend(fontsize=8)
    aA.set_title("GOOG 1-min — adaptive (W,O,H) drift: W{10,40,90,120}m x O{0,1,2} x "
                 "H{1,5,10,25}m, edge-selected | green=DIR red=MR | dashed=wall")
    aB.plot(x, np.where(np.isnan(eq), None, eq), color="#111", lw=1.2,
            label=f"gated, test ({eq[N-1]:+.0f} bps)")
    aB.plot(x, np.where(np.isnan(eq_u), None, eq_u), color="#d62728", lw=0.8,
            alpha=0.8, label=f"ungated ({eq_u[N-1]:+.0f} bps)")
    aB.plot(x[wall:], bh_series(bars, wall, med), color="#1f77b4", lw=0.7,
            alpha=0.8, label=f"buy&hold ({bh:+.0f} bps)")
    aB.axvline(wall, color="k", ls="--", lw=1.8); aB.axhline(0, color="gray", lw=0.5)
    aB.set_ylabel("bps"); aB.legend(fontsize=9)
    if len(eps):
        aD.bar(range(1, len(eps) + 1),
               eps, color=["#2ca02c" if p > 0 else "#d62728" for p in eps])
        aD.axhline(0, color="k", lw=0.6)
        aD.set_title(f"test trades ({len(eps)}; win {100*np.mean(eps>0):.0f}%, "
                     f"mean {eps.mean():+.1f} bps)")
        aD.set_xlabel("trade #"); aD.set_ylabel("bps")
    out = os.path.join("reports", "l2_backtest", "GOOG_adaptive_WOH.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\nplot -> {out}")


def bh_series(bars, wall, med):
    return (bars["mid"][wall:] - bars["mid"][wall]) / med * 1e4


if __name__ == "__main__":
    main()
