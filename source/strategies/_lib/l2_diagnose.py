"""Cross-asset diagnosis of the regime-switching imbalance strategy (gross, no fees).

Per dataset in data/l2:
  variants : A fixed thr=0.4 (baseline)    B dynamic thr (walk-fwd |imb| quantile)
             C dynamic thr, no regime gate D dynamic thr, no drift confirm
  diagnosis: signal IC (imb -> forward mid move over the hold horizon, overall /
             DIR-only / above-threshold), firing rates, and per-trade PnL split
             into ALPHA (mid-to-mid move captured) vs COST (spread paid crossing
             the book): pnl = alpha - cost. Alpha tells you if the idea predicts;
             cost tells you what execution eats.

Dynamic threshold: thr[t] = q-quantile of |imb| over the trailing 2000 snaps
(updated every 25, strictly past data) -> the entry gate adapts to each asset's
imbalance distribution instead of assuming 0.4 means the same thing everywhere.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l2_strategy import setup, backtest, WARM, LAG

HOLD = 15
DYN_W, DYN_Q, DYN_STEP, DYN_WARMN = 2000, 0.85, 25, 300


def dynamic_thr(imb, W=DYN_W, q=DYN_Q, step=DYN_STEP, warm=DYN_WARMN, fallback=0.4):
    a = np.abs(imb); n = len(a); th = np.full(n, fallback)
    for t in range(warm, n, step):
        s = max(0, t - W)
        th[t:t + step] = np.quantile(a[s:t], q)          # strictly past data
    return np.clip(th, 0.05, 0.95)


def signal_ic(D, hold=HOLD):
    """Corr of imb[t] with the forward mid move the trade would ride (t+1 .. t+1+hold)."""
    mid, imb, reg = D["mid"], D["imb"], D["reg"]
    n = D["N"]; h = hold
    fwd = np.full(n, np.nan)
    fwd[:n - 1 - h] = mid[1 + h:] - mid[1:n - h]
    ok = ~np.isnan(fwd)
    def cc(m):
        m = m & ok
        if m.sum() < 50: return np.nan
        a, b = imb[m], fwd[m]
        return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan
    base = np.ones(n, bool); base[:WARM] = False
    return dict(ic_all=cc(base), ic_dir=cc(base & (reg == 1)),
                ic_strong=cc(base & (np.abs(imb) > 0.4)))


def decompose(D, R):
    """pnl = alpha (mid move captured) - cost (execution vs mid). Returns bps of mid."""
    mid = D["mid"]; med = np.median(mid)
    alphas, costs = [], []
    for (ee, s, epx), (xe, s2, xpx, pnl, et) in zip(R["entries"], R["exits"]):
        psz = abs(R["position"][ee])
        alpha = (mid[xe] - mid[ee]) * s * psz
        alphas.append(alpha); costs.append(alpha - pnl)
    if not alphas: return dict(alpha_bps=np.nan, cost_bps=np.nan)
    return dict(alpha_bps=float(np.mean(alphas) / med * 1e4),
                cost_bps=float(np.mean(costs) / med * 1e4))


def one(path):
    D = setup(path)
    med = np.median(D["mid"])
    spread_bps = float(np.median(D["spread"][D["valid"]]) / med * 1e4)
    thr_d = dynamic_thr(D["imb"])
    runs = {
        "A_fix": backtest(D, thr=0.4, hold=HOLD, record=True),
        "B_dyn": backtest(D, thr=thr_d, hold=HOLD, record=True),
        "C_dyn_noreg": backtest(D, thr=thr_d, hold=HOLD, use_regime=False),
        "D_dyn_nodrift": backtest(D, thr=thr_d, hold=HOLD, use_drift_confirm=False),
    }
    row = dict(symbol=D["symbol"], n=D["N"],
               real=("USDT" in D["symbol"].upper()),
               spread_bps=spread_bps,
               dir_pct=float(100 * np.mean(D["reg"] == 1)),
               fire04=float(100 * np.mean(np.abs(D["imb"]) > 0.4)),
               thr_dyn_med=float(np.median(thr_d[DYN_WARMN:])),
               **signal_ic(D))
    for k, r in runs.items():
        row[k] = dict(sharpe=r["per_trade_sharpe"], t=r["tstat"], trades=r["trades"],
                      edge_bps=r["edge_bps"], pnl=r["pnl"], win=r["win_rate"])
    row["decomp_A"] = decompose(D, runs["A_fix"])
    row["decomp_B"] = decompose(D, runs["B_dyn"])
    return row


def fmt(v, w=7, d=3):
    return f"{v:+{w}.{d}f}" if v == v else " " * (w - 3) + "nan"


def main():
    files = sys.argv[1:] or sorted(glob.glob(os.path.join("data", "**", "*.npz"), recursive=True))
    rows = []
    for fp in files:
        try:
            r = one(fp); rows.append(r)
            print(f"[{len(rows):2d}/{len(files)}] {r['symbol']:<26} "
                  f"A {fmt(r['A_fix']['sharpe'])} ({r['A_fix']['trades']:3d}tr)  "
                  f"B {fmt(r['B_dyn']['sharpe'])} ({r['B_dyn']['trades']:3d}tr)  "
                  f"ICdir {fmt(r['ic_dir'], 6)}", flush=True)
        except Exception as e:
            print(f"!! {fp}: {type(e).__name__}: {e}", flush=True)

    for grp, name in ((True, "REAL (Binance futures)"), (False, "SYNTHETIC (NSE sim)")):
        sub = [r for r in rows if r["real"] == grp]
        if not sub: continue
        print(f"\n{'='*118}\n{name} -- {len(sub)} datasets, gross (no fees), hold={HOLD}\n{'='*118}")
        print(f"{'symbol':<26}{'sprd':>5}{'DIR%':>6}{'|i|>.4%':>8}{'thrDyn':>7}"
              f"{'IC_dir':>8} | {'A fix':>7}{'B dyn':>7}{'C noreg':>8}{'D nodrf':>8} |"
              f"{'alphaB':>8}{'costB':>7}{'edgeB':>7}{'trB':>5}")
        for r in sub:
            print(f"{r['symbol']:<26}{r['spread_bps']:5.1f}{r['dir_pct']:6.1f}"
                  f"{r['fire04']:8.1f}{r['thr_dyn_med']:7.2f}{fmt(r['ic_dir'],8)} |"
                  f"{fmt(r['A_fix']['sharpe'])}{fmt(r['B_dyn']['sharpe'])}"
                  f"{fmt(r['C_dyn_noreg']['sharpe'],8)}{fmt(r['D_dyn_nodrift']['sharpe'],8)} |"
                  f"{fmt(r['decomp_B']['alpha_bps'],8,2)}{fmt(r['decomp_B']['cost_bps'],7,2)}"
                  f"{fmt(r['B_dyn']['edge_bps'],7,2)}{r['B_dyn']['trades']:5d}")
        for v in ("A_fix", "B_dyn", "C_dyn_noreg", "D_dyn_nodrift"):
            sh = np.array([r[v]["sharpe"] for r in sub])
            ts = np.array([r[v]["t"] for r in sub])
            print(f"  {v:<14} mean Sharpe {sh.mean():+.3f}  median {np.median(sh):+.3f}  "
                  f"positive {100*np.mean(sh>0):.0f}%  |t|>2 pos {100*np.mean(ts>2):.0f}%  "
                  f"|t|>2 neg {100*np.mean(ts<-2):.0f}%")
        a = np.array([r["decomp_B"]["alpha_bps"] for r in sub])
        c = np.array([r["decomp_B"]["cost_bps"] for r in sub])
        ic = np.array([r["ic_dir"] for r in sub])
        print(f"  decomp (B): mean alpha {np.nanmean(a):+.2f}bps vs mean cost {np.nanmean(c):+.2f}bps"
              f"  |  mean IC_dir {np.nanmean(ic):+.4f}  IC_dir>0 on {100*np.nanmean(ic>0):.0f}%")


if __name__ == "__main__":
    main()
