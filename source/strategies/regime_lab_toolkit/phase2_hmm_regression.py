"""Phase 2 gate 1 (regression): walkforward_regime2's day-pair loop must reproduce
calling l2_hmm.reg_hmm directly per adjacent day pair -- confirms the walk-forward
glue in regime.py didn't change baseline HMM behavior before we trust the 3D
extension's comparison against it.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_lib"))
from regime import walkforward_regime2

from goog_minute_regime import build
from l2_hmm import reg_hmm

TOL = 1.0   # occupancy %, allow tiny float slack


def main():
    bars = build(1)
    di = bars["day_ix"]

    print("=" * 78)
    print("PHASE 2 GATE 1: walkforward_regime2 vs direct reg_hmm per day pair")
    print("=" * 78)

    out = walkforward_regime2(bars)
    ok = True
    print(f"{'day-pair':<12}{'wf DIR%':>9}{'wf MR%':>8}{'wf FLAT%':>9} | "
          f"{'direct DIR%':>12}{'direct MR%':>11}{'direct FLAT%':>13}")
    for k in range(len(di) - 2):
        lo, hi = di[k + 1], di[k + 2]
        wf_reg = out["hard_reg"][lo:hi]
        wf_dir = 100 * np.mean(wf_reg == 1)
        wf_mr = 100 * np.mean(wf_reg == -1)
        wf_flat = 100 * np.mean(wf_reg == 0)

        mid_fit = bars["mid"][di[k]:di[k + 1]]
        mid_dec = bars["mid"][lo:hi]
        d_reg, _, _ = reg_hmm(mid_fit, mid_dec)
        d_dir = 100 * np.mean(d_reg == 1)
        d_mr = 100 * np.mean(d_reg == -1)
        d_flat = 100 * np.mean(d_reg == 0)

        match = (abs(wf_dir - d_dir) < TOL and abs(wf_mr - d_mr) < TOL
                 and abs(wf_flat - d_flat) < TOL)
        ok &= match
        flag = "" if match else "  <-- MISMATCH"
        print(f"day {k+1:<8}{wf_dir:>9.1f}{wf_mr:>8.1f}{wf_flat:>9.1f} | "
              f"{d_dir:>12.1f}{d_mr:>11.1f}{d_flat:>13.1f}{flag}")

    print("\n" + "=" * 78)
    print(f"PHASE 2 GATE 1: {'PASS' if ok else 'FAIL'}")
    print("=" * 78)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
