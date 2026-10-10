"""Operating-scale retune for GOOG at full event resolution.

Protocol (per the user's rule -- calibrate on ONE day, test the rest frozen):
  day 1 (2026-06-11):
    a) signal-horizon study at event cadence (IC / conditional move / taker RT)
    b) grid the OPERATING SCALE: hold {0.5,1,2,5,15,60}s x gate quantile
       {0.90,0.95,0.99} x spread cap {1.5, 2.5, inf} bps
       -> pick by gross total PnL (min 100 trades)
  days 2..20: run the frozen config, report per-day + pooled, gross and net
       (Nasdaq-ish taker ~0.10 bps/side on a $355 stock).

Entry: taker at the NEXT book event's touch when |imb| >= trailing-30min
q-quantile gate AND spread <= cap; tanh conviction size; exit taker after the
fixed hold; one position at a time. Same machine as the crypto fast variant.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GATE_WIN_MS = 30 * 60_000
GATE_STEP_MS = 60_000
SIZE_K, SIZE_MAX = 1.5, 2.0
FEE_BPS = 0.10                       # per side, Nasdaq-ish on a $355 stock
HOLDS = [0.5, 1.0, 2.0, 5.0, 15.0, 60.0]
QS = [0.90, 0.95, 0.99]
CAPS = [1.5, 2.5, np.inf]
MIN_TRADES = 100
HSTUDY = [0.1, 0.25, 0.5, 1, 2, 5, 15, 60, 300]


def load(p):
    r = np.load(p)
    return (r["bb"].astype(np.float64), r["ba"].astype(np.float64),
            r["bq"].astype(np.float64), r["aq"].astype(np.float64), r["ts"])


def horizon_study(bb, ba, bq, aq, ts):
    n = len(ts); mid = (bb + ba) / 2
    imb = (bq - aq) / (bq + aq)
    spread_bps = float(np.median((ba - bb) / mid) * 1e4)
    hi = np.quantile(imb, 0.95); lo = np.quantile(imb, 0.05)
    sig = np.where(imb >= hi, 1, np.where(imb <= lo, -1, 0))
    print(f"  {n:,} events, median spread {spread_bps:.2f} bps, "
          f"gates [{lo:+.2f},{hi:+.2f}]")
    print(f"  {'h':>7} {'IC':>8} {'move_bps':>9} {'taker_bps':>10}")
    for h in HSTUDY:
        j = np.minimum(np.searchsorted(ts, ts + int(h * 1000)), n - 1)
        v = (j < n - 1) & (np.arange(n) + 1 < n)
        fwdb = (mid[j] - mid) / mid * 1e4
        m = v & (sig != 0)
        ic = float(np.corrcoef(imb[v], fwdb[v])[0, 1])
        move = float(np.mean(sig[m] * fwdb[m]))
        i1 = np.minimum(np.arange(n) + 1, n - 1)
        lp = (bb[j] - ba[i1]) / mid * 1e4
        sp_ = (bb[i1] - ba[j]) / mid * 1e4
        taker = float(np.mean(np.where(sig[m] > 0, lp[m], sp_[m])))
        print(f"  {h:>6.2f}s {ic:+8.4f} {move:+9.3f} {taker:+10.3f}")


def rolling_gate(ts, imb, q):
    a = np.abs(imb)
    grid = np.arange(ts[0] + GATE_WIN_MS, ts[-1] + GATE_STEP_MS, GATE_STEP_MS)
    gi = np.searchsorted(ts, grid)
    vals = np.empty(len(grid))
    for k in range(len(grid)):
        s = np.searchsorted(ts, grid[k] - GATE_WIN_MS)
        seg = a[s:gi[k]]
        vals[k] = np.quantile(seg, q) if len(seg) > 1000 else 1.01
    pos = np.searchsorted(grid, ts, side="right") - 1
    return np.where(pos >= 0, vals[np.clip(pos, 0, len(vals) - 1)], 1.01)


def simulate(bb, ba, ts, imb, gate, spread_bps, hold_s, cap_bps):
    n = len(ts); mid = (bb + ba) / 2
    sig_idx = np.flatnonzero((np.abs(imb) >= gate) & (spread_bps <= cap_bps))
    moves, sizes = [], []
    k = 0; hold_ms = int(hold_s * 1000)
    while k < len(sig_idx):
        i = sig_idx[k]
        fi = i + 1
        ei = np.searchsorted(ts, ts[min(fi, n - 1)] + hold_ms)
        if ei >= n or fi >= n: break
        side = 1 if imb[i] > 0 else -1
        g = gate[i]
        z = SIZE_K * (abs(imb[i]) - g) / max(1 - g, 1e-9)
        m = min(SIZE_MAX * np.tanh(max(z, 0.0)), SIZE_MAX)
        if m > 0:
            px_in = ba[fi] if side > 0 else bb[fi]
            px_out = bb[ei] if side > 0 else ba[ei]
            moves.append(side * (px_out - px_in) / mid[i] * 1e4)
            sizes.append(m)
        k = np.searchsorted(sig_idx, ei + 1)
    return np.array(moves), np.array(sizes)


def stats(moves, sizes, fee=0.0):
    if len(moves) == 0:
        return dict(trades=0, total=0.0, mean=0.0, sharpe=0.0, t=0.0, win=0.0)
    pnl = (moves - 2 * fee) * sizes
    sh = pnl.mean() / pnl.std() if pnl.std() > 0 else 0.0
    return dict(trades=len(pnl), total=float(pnl.sum()), mean=float(pnl.mean()),
                sharpe=float(sh), t=float(sh * np.sqrt(len(pnl))),
                win=float(100 * np.mean(pnl > 0)))


def main():
    files = sorted(glob.glob(os.path.join("data", "stock_data", "GOOG", "L2_full", "*", "*", "goog_*.npz")))
    if len(files) < 2:
        print(f"need full-res day files, found {len(files)}"); return
    tune_fp, test_fps = files[0], files[1:]
    day = os.path.basename(tune_fp).split("_", 1)[1].replace(".npz", "")
    print(f"=== TUNE DAY {day} — signal horizon study ===")
    bb, ba, bq, aq, ts = load(tune_fp)
    horizon_study(bb, ba, bq, aq, ts)

    mid = (bb + ba) / 2
    imb = (bq - aq) / (bq + aq)
    spread_bps = (ba - bb) / mid * 1e4
    gates = {q: rolling_gate(ts, imb, q) for q in QS}
    best, best_total = None, -np.inf
    print(f"\n=== TUNE DAY grid (gross totals, sized bps) ===")
    for q in QS:
        for hold in HOLDS:
            for cap in CAPS:
                mv, sz = simulate(bb, ba, ts, imb, gates[q], spread_bps, hold, cap)
                st = stats(mv, sz)
                tag = f"q{q:.2f} h{hold:>5.1f}s cap{cap if np.isfinite(cap) else 'inf':>4}"
                print(f"  {tag}: total {st['total']:>+9.1f}  "
                      f"{st['mean']:+.3f}/tr  ({st['trades']:>5} tr)", flush=True)
                if st["trades"] >= MIN_TRADES and st["total"] > best_total:
                    best_total, best = st["total"], (q, hold, cap)
    if best is None:
        print("\nno config met the minimum-trades bar on the tuning day"); return
    q, hold, cap = best
    print(f"\n=== FROZEN CONFIG: gate q={q}, hold={hold}s, spread cap={cap} bps "
          f"(day-1 total {best_total:+.1f}) ===")
    print(f"{'day':<14}{'trades':>8}{'gross':>10}{'bps/tr':>8}{'win%':>7}"
          f"{'t':>7}{'net(0.2rt)':>11}")
    pooled_mv, pooled_sz, daily = [], [], []
    for fp in test_fps:
        d = os.path.basename(fp).split("_", 1)[1].replace(".npz", "")
        bb, ba, bq, aq, ts = load(fp)
        mid = (bb + ba) / 2
        imb = (bq - aq) / (bq + aq)
        sp = (ba - bb) / mid * 1e4
        mv, sz = simulate(bb, ba, ts, imb, rolling_gate(ts, imb, q), sp, hold, cap)
        g = stats(mv, sz); nf = stats(mv, sz, FEE_BPS)
        print(f"{d:<14}{g['trades']:>8,}{g['total']:>+10.1f}{g['mean']:>+8.3f}"
              f"{g['win']:>7.1f}{g['t']:>+7.1f}{nf['total']:>+11.1f}", flush=True)
        pooled_mv.append(mv); pooled_sz.append(sz); daily.append(g["total"])
    mv = np.concatenate(pooled_mv); sz = np.concatenate(pooled_sz)
    g = stats(mv, sz); nf = stats(mv, sz, FEE_BPS)
    dl = np.array(daily)
    dsh = dl.mean() / dl.std() if dl.std() > 0 else 0
    print(f"\n== pooled OOS (19 days) ==")
    print(f"  gross: {g['total']:+.1f} bps total, {g['mean']:+.3f}/tr, "
          f"win {g['win']:.1f}%, per-trade t {g['t']:+.1f}  ({g['trades']:,} trades)")
    print(f"  net  : {nf['total']:+.1f} bps total, {nf['mean']:+.3f}/tr")
    print(f"  daily: mean {dl.mean():+.1f} bps/day, Sharpe {dsh:+.2f} "
          f"(t={dsh*np.sqrt(len(dl)):+.1f}), positive days {100*np.mean(dl>0):.0f}%")


if __name__ == "__main__":
    main()
