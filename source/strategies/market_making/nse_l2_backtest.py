"""Run the regime-switching L2 strategy (tanh sizing) on NSE order-book .npz files
and render the diagnostic dashboard per instrument:

    1. midprice with buy/sell entry arrows + exits, regime-shaded background
    2. mark-to-market PnL (equity) with regime shading
    3. signed position (tanh conviction size)
    4. top-of-book imbalance vs +/-thr, with the drift-confirm signal
    5. equity drawdown
    6. per-trade PnL histogram + regime split bar

Usage:  python scripts/nse_l2_backtest.py [file1.npz file2.npz ...]
        (no args -> simulate the default NSE universe first)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_lib"))
from l2_strategy import setup, backtest, WARM

REG_COLORS = {1: ("#2ca02c", 0.13), -1: ("#d62728", 0.10), 0: ("#7f7f7f", 0.04)}
REG_NAMES = {1: "DIR (trending)", -1: "MM (mean-rev)", 0: "FLAT (no edge)"}


def shade_regimes(ax, reg):
    """Background shading per detected regime (green=DIR, red=MM, gray=FLAT)."""
    N = len(reg); t0 = 0
    for t in range(1, N + 1):
        if t == N or reg[t] != reg[t0]:
            c, a = REG_COLORS[int(reg[t0])]
            ax.axvspan(t0, t, color=c, alpha=a, lw=0)
            t0 = t


def dashboard(D, R, thr, out_png):
    N = D["N"]; x = np.arange(N)
    fig = plt.figure(figsize=(15, 18))
    gs = fig.add_gridspec(6, 2, height_ratios=[2.2, 1.6, 1.1, 1.3, 1.0, 1.3],
                          hspace=0.42, wspace=0.25)
    ax_mid = fig.add_subplot(gs[0, :])
    ax_eq  = fig.add_subplot(gs[1, :], sharex=ax_mid)
    ax_pos = fig.add_subplot(gs[2, :], sharex=ax_mid)
    ax_imb = fig.add_subplot(gs[3, :], sharex=ax_mid)
    ax_dd  = fig.add_subplot(gs[4, :], sharex=ax_mid)
    ax_hist = fig.add_subplot(gs[5, 0])
    ax_reg  = fig.add_subplot(gs[5, 1])

    # -- 1. mid + trades ------------------------------------------------
    shade_regimes(ax_mid, D["reg"])
    ax_mid.plot(x, D["mid"], color="#1f77b4", lw=0.7, label="mid")
    be = [(t, p) for t, s, p in R["entries"] if s > 0]
    se = [(t, p) for t, s, p in R["entries"] if s < 0]
    if be: ax_mid.scatter(*zip(*be), marker="^", s=55, color="#2ca02c",
                          edgecolors="k", lw=0.4, zorder=5, label=f"long entry ({len(be)})")
    if se: ax_mid.scatter(*zip(*se), marker="v", s=55, color="#d62728",
                          edgecolors="k", lw=0.4, zorder=5, label=f"short entry ({len(se)})")
    ex = [(t, p) for t, s, p, pnl, et in R["exits"]]
    if ex: ax_mid.scatter(*zip(*ex), marker="x", s=34, color="k", zorder=5,
                          label=f"exit ({len(ex)})")
    ax_mid.set_title(f"{D['symbol']} — midprice, trades, regime background "
                     f"(green=DIR, red=MM, gray=FLAT)")
    ax_mid.set_ylabel("price (Rs)"); ax_mid.legend(loc="best", fontsize=8)
    ax_mid.axvline(WARM, color="k", ls=":", lw=0.8)

    # -- 2. equity -------------------------------------------------------
    shade_regimes(ax_eq, D["reg"])
    ax_eq.plot(x, R["equity"], color="#111111", lw=0.9)
    ax_eq.axhline(0, color="gray", lw=0.6)
    ax_eq.set_title(f"mark-to-market PnL (tanh sizing, gross)  |  final {R['pnl']:+.2f} Rs/unit  "
                    f"Sharpe(per-trade) {R['per_trade_sharpe']:+.3f}  t={R['tstat']:+.2f}")
    ax_eq.set_ylabel("PnL (Rs)")

    # -- 3. position -----------------------------------------------------
    ax_pos.fill_between(x, R["position"], step="post",
                        where=R["position"] >= 0, color="#2ca02c", alpha=0.7)
    ax_pos.fill_between(x, R["position"], step="post",
                        where=R["position"] < 0, color="#d62728", alpha=0.7)
    ax_pos.axhline(0, color="gray", lw=0.6)
    ax_pos.set_title("signed position (tanh conviction size, max 2.0)")
    ax_pos.set_ylabel("units")

    # -- 4. imbalance ----------------------------------------------------
    ax_imb.plot(x, D["imb"], color="#9467bd", lw=0.5)
    for y in (thr, -thr): ax_imb.axhline(y, color="k", ls="--", lw=0.7)
    ax_imb.plot(x, 0.25 * D["drift"], color="#ff7f0e", lw=0.6, alpha=0.7,
                label="drift sign (scaled)")
    ax_imb.set_title(f"top-of-book imbalance vs entry threshold |imb|>{thr} + drift confirm")
    ax_imb.set_ylabel("imbalance"); ax_imb.set_ylim(-1.05, 1.05)
    ax_imb.legend(loc="best", fontsize=8)

    # -- 5. drawdown -------------------------------------------------------
    dd = R["equity"] - np.maximum.accumulate(R["equity"])
    ax_dd.fill_between(x, dd, 0, color="#d62728", alpha=0.55)
    ax_dd.set_title(f"equity drawdown  |  max {dd.min():+.2f} Rs")
    ax_dd.set_ylabel("Rs"); ax_dd.set_xlabel("snapshot index (tick-to-tick)")

    # -- 6a. trade PnL histogram ------------------------------------------
    ax_hist.hist(R["trade_pnls"], bins=40, color="#1f77b4", edgecolor="white", lw=0.3)
    ax_hist.axvline(0, color="k", lw=0.8)
    ax_hist.set_title(f"per-trade PnL ({R['trades']} trades, win rate {R['win_rate']:.1f}%)")
    ax_hist.set_xlabel("Rs per trade")

    # -- 6b. regime split --------------------------------------------------
    fr = {k: 100 * np.mean(D["reg"] == k) for k in (1, -1, 0)}
    bars = ax_reg.bar([REG_NAMES[k] for k in (1, -1, 0)],
                      [fr[k] for k in (1, -1, 0)],
                      color=[REG_COLORS[k][0] for k in (1, -1, 0)], alpha=0.75)
    for b, k in zip(bars, (1, -1, 0)):
        ax_reg.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                    f"{fr[k]:.1f}%", ha="center", fontsize=10)
    ax_reg.set_ylim(0, 100); ax_reg.set_title("regime classification (% of snapshots)")
    ax_reg.tick_params(axis="x", labelsize=8)

    fig.suptitle(f"{D['symbol']}  |  N={N} L2 snapshots, K={D['K']} levels  |  "
                 f"invalid {100 * np.mean(~D['valid']):.1f}%", y=0.995, fontsize=13)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    files = sys.argv[1:]
    if not files:
        import nse_l2_sim
        files = nse_l2_sim.main()
    outdir = os.path.join("reports", "l2_backtest")
    os.makedirs(outdir, exist_ok=True)

    for fp in files:
        D = setup(fp)
        reg_pct = {k: 100 * np.mean(D["reg"] == k) for k in (1, -1, 0)}
        print(f"\n=== {D['symbol']} ({fp}) ===")
        print(f"  snapshots {D['N']}  invalid {100 * np.mean(~D['valid']):.1f}%")
        print(f"  regime split: DIR {reg_pct[1]:.1f}%  |  MM {reg_pct[-1]:.1f}%  |  "
              f"FLAT {reg_pct[0]:.1f}%")
        for fee in (0.0, 2.5, 5.0):
            r = backtest(D, sizing="tanh", fee_bps=fee, record=(fee == 0.0))
            tag = "gross" if fee == 0 else f"net {2 * fee:.0f}bps rt"
            print(f"  tanh [{tag:>12}]: Sharpe {r['per_trade_sharpe']:+.3f}  "
                  f"t={r['tstat']:+.2f}  PnL {r['pnl']:+.2f}  "
                  f"edge {r['edge_bps']:+.3f}bps  win {r['win_rate']:.1f}%  ({r['trades']} tr)")
            if fee == 0.0:
                R = r
        png = os.path.join(outdir, f"{D['symbol']}_dashboard.png")
        dashboard(D, R, thr=0.4, out_png=png)
        print(f"  plot -> {png}")


if __name__ == "__main__":
    main()
