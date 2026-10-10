"""Adaptive (W,O,H) drift, long-horizon edition -- the selector asked for it.

Changes vs goog_adaptive_wh (all declared before running):
  * horizons extended to {25, 60, 120, 390} MINUTES (390 = full session);
    holds match the selected horizon and may span days (you hold shares)
  * open-gap guard: no NEW entries in the first 30 minutes of a session
    (fits spanning the overnight gap produced the two big June-22 losers)
  * run on BOTH symbols: GOOG (1-min bars from the full event stream) and
    NFTY (5-min forward-filled bars from the snapshot files; window/horizon
    menus converted to bars, min fit window 6 bars)

Everything else unchanged: windows {10,40,90,120}m, orders {0,1,2}, L=150-bar
trailing-edge selection (causal), regime gate cost_mult=0.5, entry needs
|predicted move| > spread, wall after day 5, single test pass, baselines.
"""
from __future__ import annotations

import glob
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
from goog_minute_regime import build as build_goog, TRAIN_DAYS, REG_W, FEE_BPS
from l2_strategy import compute_regime, _proj

W_MIN = [10, 40, 90, 120]
H_MIN = [25, 60, 120, 390]
O_LIST = [0, 1, 2]
L = 150
COST_MULT = 0.5
OPEN_GUARD_MIN = 30


def build_snaps(sym, bar_min):
    files = sorted(glob.glob(os.path.join("data", "stock_data", "NFTY_ETF", "*", "*",
                                          f"orderbook_{sym}_20*.npz")))
    parts, day_ix = [], [0]
    for fp in files:
        r = np.load(fp, allow_pickle=True)
        bb = r["bids"][:, 0, 0].astype(float); ba = r["asks"][:, 0, 0].astype(float)
        ts = r["ts"].astype(np.int64)
        ok = (bb > 0) & (ba > bb)
        bb, ba, ts_v = bb[ok], ba[ok], ts[ok]
        step = bar_min * 60_000
        bounds = np.arange(ts_v[0] + step, ts_v[-1] + step, step)
        idx = np.searchsorted(ts_v, bounds, side="right") - 1
        good = idx >= 0
        idx, bounds = idx[good], bounds[good]
        tts = ts[r["tr_bookidx"].astype(int)]
        fl = r["tr_side"].astype(float) * r["tr_qty"].astype(float)
        bkt = np.clip(np.searchsorted(bounds, tts, side="left"), 0, len(bounds) - 1)
        flow = np.bincount(bkt, weights=fl, minlength=len(bounds))
        parts.append(dict(bb=bb[idx], ba=ba[idx], flow=flow, ts=bounds))
        day_ix.append(day_ix[-1] + len(idx))
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    cat["mid"] = (cat["bb"] + cat["ba"]) / 2
    cat["spread"] = cat["ba"] - cat["bb"]
    cat["day_ix"] = np.array(day_ix)
    return cat


def bars_since_open(bars):
    N = len(bars["mid"]); out = np.zeros(N, int)
    di = bars["day_ix"]
    for i in range(len(di) - 1):
        out[di[i]:di[i + 1]] = np.arange(di[i + 1] - di[i])
    return out


def adaptive_drift(mid, spread, Wb, Hb):
    N = len(mid); cost = np.median(spread)
    combos, disps = [], []
    for W in Wb:
        wv = swv(mid, W)
        for O in O_LIST:
            C = np.full((N, O + 1), np.nan)
            C[W - 1:] = wv @ _proj(W, O, W / 2).T
            for H in Hb:
                fut = np.array([float(H) ** k for k in range(O + 1)])
                combos.append((W, O, H))
                disps.append(np.nan_to_num(C @ fut - mid))
    D = np.vstack(disps)
    recent = np.full((len(combos), N), -1e9)
    for i, (W, O, H) in enumerate(combos):
        rz = np.full(N, np.nan); rz[:N - H] = mid[H:] - mid[:N - H]
        e = np.nan_to_num(np.sign(D[i]) * rz - cost)
        cs = np.cumsum(e)
        if N > H + L:
            recent[i, H + L:] = cs[L:N - H] - cs[:N - H - L]
    sel = np.argmax(recent, axis=0)
    ar = np.arange(N)
    return D[sel, ar], np.array([combos[s][2] for s in sel]), sel, combos


def trade(bars, reg, disp, Hs, bso, guard, lo, hi, use_regime=True):
    mid, bb, ba, sp = bars["mid"], bars["bb"], bars["ba"], bars["spread"]
    med = np.median(mid[lo:hi])
    cash = 0.0; pos = 0.0; hold_until = -1; entry_eq = 0.0
    eq = np.full(len(mid), np.nan); eps, entries, exits = [], [], []
    for t in range(lo, hi - 1):
        if pos == 0:
            if ((reg[t] == 1 or not use_regime) and abs(disp[t]) > sp[t]
                    and bso[t] >= guard):
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
        cash += pos * (bb[hi - 1] if pos > 0 else ba[hi - 1])
        eps.append((cash - entry_eq) / med * 1e4)
    eq[hi - 1] = cash / med * 1e4
    return eq, np.array(eps), entries, exits


def stat(eps, total):
    if len(eps) == 0: return "0 trades"
    sh = eps.mean() / eps.std() if eps.std() > 0 else 0
    return (f"{total:+8.1f} bps | {len(eps):3d} tr, {eps.mean():+6.1f}/tr, "
            f"win {100 * np.mean(eps > 0):4.1f}%, t {sh * np.sqrt(len(eps)):+5.2f}")


def run_symbol(name, bars, bar_min):
    N = len(bars["mid"]); wall = bars["day_ix"][TRAIN_DAYS]
    Wb = sorted({max(6, round(m / bar_min)) for m in W_MIN})
    Hb = sorted({max(1, round(m / bar_min)) for m in H_MIN})
    guard = max(1, round(OPEN_GUARD_MIN / bar_min))
    warm = max(Wb) + max(Hb) + L + 10
    print(f"\n=== {name.upper()} ({bar_min}-min bars, N={N}, wall={wall}, "
          f"W(bars)={Wb}, H(bars)={Hb}, guard={guard}) ===")
    reg = compute_regime(bars["mid"], bars["spread"], bars["flow"],
                         W=REG_W, cost_mult=COST_MULT)
    disp, Hs, sel, combos = adaptive_drift(bars["mid"], bars["spread"], Wb, Hb)
    bso = bars_since_open(bars)
    eq_tr, eps_tr, _, _ = trade(bars, reg, disp, Hs, bso, guard, warm, wall)
    print(f"TRAIN : {stat(eps_tr, eq_tr[wall - 1])}")
    eq, eps, entries, exits = trade(bars, reg, disp, Hs, bso, guard, wall, N)
    print(f"TEST  : {stat(eps, eq[N - 1])}")
    eq_u, eps_u, _, _ = trade(bars, reg, disp, Hs, bso, guard, wall, N,
                              use_regime=False)
    print(f"  ungated    : {stat(eps_u, eq_u[N - 1])}")
    med = np.median(bars["mid"][wall:])
    bh = (bars["mid"][N - 1] - bars["mid"][wall]) / med * 1e4
    print(f"  buy & hold : {bh:+.1f} bps  |  DIR occupancy "
          f"{100 * np.mean(reg[wall:] == 1):.1f}%")
    cnt = Counter(combos[s] for s in sel[wall:])
    tot = N - wall
    print("  selector top-5: " + "  ".join(
        f"W{w}/O{o}/H{h}:{100 * c / tot:.0f}%" for (w, o, h), c in cnt.most_common(5)))

    # dashboard
    x = np.arange(N)
    fig, (aA, aB) = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
                                 height_ratios=[2, 1])
    ylim = (bars["mid"].min() * 0.999, bars["mid"].max() * 1.001)
    aA.fill_between(x, *ylim, where=reg == 1, color="#2ca02c", alpha=0.15)
    aA.fill_between(x, *ylim, where=reg == -1, color="#d62728", alpha=0.08)
    aA.plot(x, bars["mid"], lw=0.6, color="#1f77b4")
    eL = [(t, p) for t, p, s in entries if s > 0]
    eS = [(t, p) for t, p, s in entries if s < 0]
    if eL: aA.scatter(*zip(*eL), marker="^", s=80, color="#2ca02c",
                      edgecolors="k", zorder=5, label=f"long ({len(eL)})")
    if eS: aA.scatter(*zip(*eS), marker="v", s=80, color="#d62728",
                      edgecolors="k", zorder=5, label=f"short ({len(eS)})")
    if exits: aA.scatter([t for t, _, _ in exits], [p for _, p, _ in exits],
                         marker="x", s=50, color="k", zorder=5,
                         label=f"exit ({len(exits)})")
    aA.axvline(wall, color="k", ls="--", lw=1.8)
    aA.set_ylim(*ylim); aA.set_ylabel("$"); aA.legend(fontsize=8)
    aA.set_title(f"{name.upper()} {bar_min}m — adaptive W/O/H with H up to full "
                 f"session, 30-min open guard | dashed = wall")
    aB.plot(x, eq, color="#111", lw=1.2, label=f"gated test ({eq[N-1]:+.0f})")
    aB.plot(x, eq_u, color="#d62728", lw=0.8, alpha=0.8,
            label=f"ungated ({eq_u[N-1]:+.0f})")
    aB.plot(x[wall:], (bars["mid"][wall:] - bars["mid"][wall]) / med * 1e4,
            color="#1f77b4", lw=0.7, alpha=0.8, label=f"buy&hold ({bh:+.0f})")
    aB.axvline(wall, color="k", ls="--", lw=1.8)
    aB.axhline(0, color="gray", lw=0.5); aB.set_ylabel("bps"); aB.legend(fontsize=8)
    out = os.path.join("reports", "l2_backtest", f"{name}_adaptive_long.png")
    fig.tight_layout(); fig.savefig(out, dpi=110)
    print(f"  plot -> {out}")


def main():
    run_symbol("goog", build_goog(1), 1)
    run_symbol("nfty", build_snaps("nfty", 5), 5)


if __name__ == "__main__":
    main()
