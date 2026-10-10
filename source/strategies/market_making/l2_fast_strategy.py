"""Fast (tick-cadence) variant of the imbalance strategy, on FULL bookTicker data.

Re-architecture motivated by l2_alpha_horizon: the signal's IC peaks at 0.1-1s
and decays 5x by 60s, so decisions/holds move from minutes to seconds.

Rules (fixed a priori from the horizon study -- no tuning on these runs):
  * gate    : |imb| >= trailing 95th pct of |imb| over the past 30 min
              (walk-forward, refreshed every 60s of data) -- the dynamic
              threshold, so the trigger self-calibrates per asset
  * spread  : enter only when spread <= 0.35 bps (alpha saturates ~1 bp, so a
              wider spread cannot be paid; wide-spread assets go quiet BY DESIGN)
  * entry   : taker at the NEXT book update's touch (also a +250ms stress leg)
  * size    : tanh conviction, same constants as the slow strategy
  * exit    : taker after a fixed hold (0.25s / 1s / 5s), one position at a time
  * fees    : reported gross (Prosperity mode), net taker 4.5bps/side, and the
              breakeven fee per side implied by the gross edge
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data_pipeline"))
from l2_alpha_horizon import load_full

Q = 0.95
GATE_WIN_MS = 30 * 60_000
GATE_STEP_MS = 60_000
SPREAD_CAP_BPS = 0.35
HOLDS_S = [0.25, 1.0, 5.0]
DELAYS = [("next", None), ("250ms", 250)]
SIZE_K, SIZE_MAX = 1.5, 2.0
FEE_TAKER_BPS = 4.5

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
DATES = ["2023-09-01", "2023-12-01", "2024-03-01"]


def rolling_gate(ts, imb):
    """Walk-forward 95th pct of |imb| over the trailing 30 min, stepped every 60s."""
    a = np.abs(imb)
    grid = np.arange(ts[0] + GATE_WIN_MS, ts[-1] + GATE_STEP_MS, GATE_STEP_MS)
    gi = np.searchsorted(ts, grid)
    vals = np.empty(len(grid))
    for k in range(len(grid)):
        s = np.searchsorted(ts, grid[k] - GATE_WIN_MS)
        seg = a[s:gi[k]]
        vals[k] = np.quantile(seg, Q) if len(seg) > 1000 else 1.01
    pos = np.searchsorted(grid, ts, side="right") - 1        # strictly past grid pt
    gate = np.where(pos >= 0, vals[np.clip(pos, 0, len(vals) - 1)], 1.01)
    return gate


def simulate(bb, ba, ts, imb, gate, spread_bps, hold_s, delay_ms):
    n = len(ts)
    mid = (bb + ba) / 2
    sig_idx = np.flatnonzero((np.abs(imb) >= gate) & (spread_bps <= SPREAD_CAP_BPS))
    moves, sizes, sides, t_exit = [], [], [], []
    k = 0; hold_ms = int(hold_s * 1000)
    while k < len(sig_idx):
        i = sig_idx[k]
        fi = i + 1 if delay_ms is None else np.searchsorted(ts, ts[i] + delay_ms)
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
            sizes.append(m); sides.append(side); t_exit.append(ts[ei])
        k = np.searchsorted(sig_idx, ei + 1)                 # one position at a time
    return (np.array(moves), np.array(sizes), np.array(sides),
            np.array(t_exit, dtype=np.int64))


def metrics(moves, sizes, fee_bps=0.0):
    if len(moves) == 0:
        return dict(trades=0, total=0.0, mean=0.0, sharpe=0.0, t=0.0, win=0.0, be=0.0)
    pnl = (moves - 2 * fee_bps) * sizes
    sh = pnl.mean() / pnl.std() if pnl.std() > 0 else 0.0
    return dict(trades=len(pnl), total=float(pnl.sum()), mean=float(pnl.mean()),
                sharpe=float(sh), t=float(sh * np.sqrt(len(pnl))),
                win=float(100 * np.mean(pnl > 0)),
                be=float((moves * sizes).sum() / (2 * sizes.sum())))


def main():
    os.makedirs(os.path.join("reports", "l2_backtest"), exist_ok=True)
    fig, axes = plt.subplots(len(SYMBOLS), len(DATES),
                             figsize=(5.2 * len(DATES), 3.0 * len(SYMBOLS)),
                             squeeze=False)
    for r, sym in enumerate(SYMBOLS):
        for c, date in enumerate(DATES):
            try:
                bb, ba, bq, aq, ts = load_full(sym, date)
            except Exception as e:
                print(f"!! {sym} {date}: {type(e).__name__}: {e}", flush=True)
                axes[r][c].set_visible(False); continue
            mid = (bb + ba) / 2
            imb = (bq - aq) / (bq + aq)
            spread_bps = (ba - bb) / mid * 1e4
            gate = rolling_gate(ts, imb)
            tradable = float(100 * np.mean(spread_bps <= SPREAD_CAP_BPS))
            print(f"\n{sym} {date}: {len(ts):,} updates, "
                  f"spread<= {SPREAD_CAP_BPS}bps on {tradable:.0f}% of book", flush=True)
            print(f"  {'hold':>6} {'delay':>6} {'trades':>7} {'grossSum':>10} "
                  f"{'bps/tr':>8} {'Sharpe':>8} {'t':>7} {'win%':>6} "
                  f"{'beFee':>7} {'net(4.5)':>10}")
            for hold in HOLDS_S:
                for dname, dms in DELAYS:
                    mv, sz, sd, tx = simulate(bb, ba, ts, imb, gate,
                                              spread_bps, hold, dms)
                    g = metrics(mv, sz); nf = metrics(mv, sz, FEE_TAKER_BPS)
                    print(f"  {hold:>5.2f}s {dname:>6} {g['trades']:>7,} "
                          f"{g['total']:>+10.1f} {g['mean']:>+8.3f} "
                          f"{g['sharpe']:>+8.3f} {g['t']:>+7.1f} {g['win']:>6.1f} "
                          f"{g['be']:>7.3f} {nf['total']:>+10.1f}", flush=True)
                    if hold == 1.0 and dname == "next":
                        ax = axes[r][c]
                        if g["trades"]:
                            eq = np.cumsum(mv * sz)
                            x = (tx - ts[0]) / 3_600_000
                            ax.plot(x, eq, lw=0.9, color="tab:blue")
                            ax.axhline(0, color="gray", lw=0.5)
                            ax.set_title(f"{sym} {date}  h=1s gross "
                                         f"{eq[-1]:+.0f}bps  Sh {g['sharpe']:+.2f} "
                                         f"t {g['t']:+.0f} ({g['trades']:,}tr)",
                                         fontsize=8)
                        else:
                            ax.text(0.5, 0.5, f"{sym} {date}\nno tradable spread",
                                    ha="center", va="center", fontsize=9,
                                    transform=ax.transAxes)
                        ax.tick_params(labelsize=7)
            del bb, ba, bq, aq, ts, mid, imb, spread_bps, gate
    fig.suptitle("Fast imbalance strategy — cumulative gross PnL (bps of mid), "
                 "hold 1s, next-update fills, dynamic 95th-pct gate", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join("reports", "l2_backtest", "fast_strategy_summary.png")
    fig.savefig(out, dpi=110)
    print(f"\nplot -> {out}")


if __name__ == "__main__":
    main()
