"""Head-to-head: VR/autocorr regime detector vs walk-forward Gaussian HMM,
both gating the identical MM+DIR engine on per-day L2 files.

HMM protocol: fit on day k-1, causally filter day k (no same-day fitting, no
smoothing/Viterbi). Day 1 has no prior day and is skipped for both arms so the
comparison is apples-to-apples.

Usage: python scripts/l2_regime_compare.py [glob-or-files...]
       (default: data/l2/orderbook_goog_*.npz)
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_lib"))
from l2_strategy import setup
from l2_mmdir import run_day, trades_map
from l2_hmm import reg_hmm

SCEN = ("nasdaq", 0.10, 0.15)     # maker rebate bps, taker fee bps


def main():
    pats = sys.argv[1:] or [os.path.join("data", "stock_data", "GOOG", "L2_top", "*", "*", "orderbook_goog_*.npz")]
    files = sorted(sum([glob.glob(p) if any(c in p for c in "*?") else [p]
                        for p in pats], []))
    if len(files) < 2:
        print(f"need >=2 day files, got {len(files)}"); return
    print(f"{len(files)} day files; comparing on days 2..{len(files)}")
    hdr = (f"{'day':<20}{'arm':>6}{'DIR%':>6}{'MM%':>6} | "
           f"{'MMfil':>6}{'MMpnl':>8}{'DIRtr':>6}{'DIRpnl':>8}"
           f"{'tot-gross':>10}{'tot-net':>9}")
    print(hdr)
    agg = {"vrac": {"tot": [], "net": []}, "hmm": {"tot": [], "net": []}}
    D_prev = None
    for i, fp in enumerate(files):
        D = setup(fp)
        tm = trades_map(fp, D["N"])
        if D_prev is not None:
            for arm in ("vrac", "hmm"):
                Da = D
                if arm == "hmm":
                    reg, P, lab = reg_hmm(D_prev["mid"], D["mid"])
                    Da = dict(D); Da["reg"] = reg
                g = run_day(Da, 0, 0, "strict", tm)
                n = run_day(Da, SCEN[1], SCEN[2], "strict", tm)
                dirp = 100 * np.mean(Da["reg"] == 1)
                mmp = 100 * np.mean(Da["reg"] != 1)
                print(f"{D['symbol']:<20}{arm:>6}{dirp:>6.1f}{mmp:>6.1f} | "
                      f"{g['mm_fills']:>6}{g['mm_pnl']:>+8.2f}{g['dir_trades']:>6}"
                      f"{g['dir_pnl']:>+8.2f}{g['total']:>+10.2f}{n['total']:>+9.2f}",
                      flush=True)
                agg[arm]["tot"].append(g["total"]); agg[arm]["net"].append(n["total"])
        D_prev = D
    print()
    for arm in ("vrac", "hmm"):
        for tag in ("tot", "net"):
            xs = np.array(agg[arm][tag])
            if len(xs) == 0: continue
            sh = xs.mean() / xs.std() if xs.std() > 0 else 0.0
            print(f"  {arm:<5} {tag:<4}: mean {xs.mean():+8.2f} bps/day  "
                  f"Sharpe(d) {sh:+.2f} (t={sh*np.sqrt(len(xs)):+.1f})  "
                  f"pos days {100*np.mean(xs>0):.0f}%")


if __name__ == "__main__":
    main()
