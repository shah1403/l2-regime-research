"""Phase 2 gate 2: the REAL experiment. Does adding flow to the HMM's observation
vector (2D -> 3D) actually improve the regime posterior's forward-informativeness,
or not? Either answer is acceptable and must be reported plainly -- see
scripts/l2_deepdiag.py's regime_future_stats (l2_deepdiag.py:48), whose method
this reuses: sample every 25 bars, look at the FUTURE FUT_W-bar window, and ask
whether the label predicted anything about it (drift-ratio, variance-ratio,
autocorrelation, |move|). That analysis is what proved the old VR/autocorrelation
detector's labels were forward-blind on GOOG at minute scale -- this rerun applies
the identical yardstick to both the 2D and 3D HMM posteriors, side by side.

Two comparisons:
  1. Hard-label future-window stats (2D vs 3D), same method as l2_deepdiag.
  2. Continuous correlation of the DIR-up-minus-DIR-down posterior signal
     (P(DIR-up)-P(DIR-down), identical formula for both models since both go
     through regime.dir_up_down_split) against realized forward flow_momentum
     edge -- does the regime posterior predict flow_momentum's own edge?

Both walkforward_regime2/3 are inherently causal walk-forward (fit day k, decode
day k+1) so this experiment uses the FULL decoded range (days 1-19, i.e. every
bar with a real posterior) rather than restricting to a held-out test slice --
every decoded bar is already out-of-sample relative to the day it was fit on,
regardless of calendar position.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_lib"))
from regime import walkforward_regime2, walkforward_regime3
from signals import build_ctx

from goog_minute_regime import build, TRAIN_DAYS
from l2_strategy import variance_ratio, autocorr1

FUT_W = 100      # ~100 minutes forward window, matching l2_deepdiag's convention
SAMPLE_STEP = 25
FLOW_EDGE_HOLD = 5   # matches flow_momentum's default hold, for the edge proxy
REG_NAME = {1: "DIR", -1: "MR", 0: "FLAT"}


def future_stats(mid, hard_reg, lo, hi):
    acc = {}
    r = np.diff(mid, prepend=mid[0])
    for t in range(lo, min(hi, len(mid) - FUT_W), SAMPLE_STEP):
        seg = r[t + 1:t + 1 + FUT_W]
        sd = seg.std()
        drift = abs(mid[t + FUT_W] - mid[t]) / (sd * np.sqrt(FUT_W) + 1e-12)
        a = acc.setdefault(hard_reg[t], dict(dr=[], vr=[], ac=[], mv=[]))
        a["dr"].append(drift)
        a["vr"].append(variance_ratio(seg, 8))
        a["ac"].append(autocorr1(seg))
        a["mv"].append(abs(mid[t + FUT_W] - mid[t]))
    return acc


def separation_score(acc):
    """Simple effect-size proxy: how far DIR's mean drift-ratio sits from the
    pooled MR+FLAT mean, in units of the pooled std. Higher = more informative."""
    if 1 not in acc or len(acc[1]["dr"]) < 5:
        return float("nan")
    dir_dr = np.array(acc[1]["dr"])
    rest = np.concatenate([acc[k]["dr"] for k in (-1, 0) if k in acc and acc[k]["dr"]])
    if len(rest) < 5:
        return float("nan")
    pooled_sd = np.sqrt((dir_dr.var() + rest.var()) / 2) + 1e-9
    return (dir_dr.mean() - rest.mean()) / pooled_sd


def main():
    bars = build(1)
    N = len(bars["mid"])
    wall = bars["day_ix"][TRAIN_DAYS]
    day1_start = bars["day_ix"][1]     # first bar with a real (non-placeholder) posterior

    print("Fitting walk-forward 2D baseline HMM (all day pairs)...", flush=True)
    out2 = walkforward_regime2(bars)
    print("Fitting walk-forward 3D (+flow) HMM (all day pairs)...", flush=True)
    out3 = walkforward_regime3(bars)
    ctx = build_ctx(bars, wall)

    print("\n" + "=" * 78)
    print("1. HARD-LABEL FUTURE-WINDOW STATS (l2_deepdiag.regime_future_stats method)")
    print("=" * 78)
    acc2 = future_stats(bars["mid"], out2["hard_reg"], day1_start, N)
    acc3 = future_stats(bars["mid"], out3["hard_reg"], day1_start, N)

    print(f"{'model':<8}{'label':<6}{'n':>6}{'fut drift-ratio':>17}{'fut VR(8)':>11}"
          f"{'fut AC1':>9}{'fut |move|':>11}")
    for tag, acc in (("2D", acc2), ("3D", acc3)):
        for k in (1, -1, 0):
            if k not in acc or not acc[k]["dr"]:
                continue
            a = acc[k]
            print(f"{tag:<8}{REG_NAME[k]:<6}{len(a['dr']):>6}"
                  f"{np.mean(a['dr']):>17.3f}{np.mean(a['vr']):>11.3f}"
                  f"{np.mean(a['ac']):>9.3f}{np.mean(a['mv']):>11.4f}")

    sep2 = separation_score(acc2)
    sep3 = separation_score(acc3)
    print(f"\n  DIR-vs-rest drift-ratio separation score: 2D={sep2:+.3f}  3D={sep3:+.3f}"
          f"  ({'3D separates further' if sep3 > sep2 else '2D separates further or tied'})")

    print("\n" + "=" * 78)
    print("2. CONTINUOUS CORRELATION: (P(DIR-up)-P(DIR-down)) vs realized "
          f"forward flow_momentum edge (hold={FLOW_EDGE_HOLD}m)")
    print("=" * 78)
    mid = bars["mid"]
    fwd_move_bps = np.full(N, np.nan)
    fwd_move_bps[:N - FLOW_EDGE_HOLD] = ((mid[FLOW_EDGE_HOLD:] - mid[:N - FLOW_EDGE_HOLD])
                                         / mid[:N - FLOW_EDGE_HOLD] * 1e4)
    flow_edge = np.sign(ctx["flow_z"]) * fwd_move_bps

    sample_idx = np.arange(day1_start, N - FLOW_EDGE_HOLD, SAMPLE_STEP)
    cont2 = out2["posterior"][:, 0] - out2["posterior"][:, 1]
    cont3 = out3["posterior"][:, 0] - out3["posterior"][:, 1]

    def corr(a, b, idx):
        a, b = a[idx], b[idx]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 30 or a[m].std() == 0 or b[m].std() == 0:
            return float("nan")
        return float(np.corrcoef(a[m], b[m])[0, 1])

    c2 = corr(cont2, flow_edge, sample_idx)
    c3 = corr(cont3, flow_edge, sample_idx)
    print(f"  corr(2D DIR-up-minus-down posterior, flow_momentum edge) = {c2:+.4f}")
    print(f"  corr(3D DIR-up-minus-down posterior, flow_momentum edge) = {c3:+.4f}")

    print("\n" + "=" * 78)
    print("EXPLICIT FORK -- decision rule: 3D wins if it improves BOTH metrics, or")
    print("improves one without meaningfully worsening the other (>0.02 corr / >0.1 sep)")
    print("=" * 78)
    corr_better = (abs(c3) - abs(c2)) if np.isfinite(c3) and np.isfinite(c2) else float("nan")
    sep_better = (sep3 - sep2) if np.isfinite(sep3) and np.isfinite(sep2) else float("nan")
    three_d_wins = (np.isfinite(corr_better) and np.isfinite(sep_better)
                    and corr_better > -0.02 and sep_better > -0.1
                    and (corr_better > 0.02 or sep_better > 0.1))
    if three_d_wins:
        print("RESULT: 3D (+flow) posterior beats the 2D baseline on forward-"
              "informativeness.\n-> Phase 4 uses the 3D posterior as the assembler's regime input.")
    else:
        print("RESULT: 3D (+flow) posterior does NOT clearly beat the 2D baseline "
              "on forward-informativeness.\n-> Phase 4 falls back to the 2D posterior "
              "for regime-matching, but flow_z stays available as a RAW Stage-1\n"
              "   applicability feature (outside the HMM) -- flow precedes price-shape "
              "per this repo's established finding.")
    print("(This is a genuine experimental outcome, not a formality -- reported as-is.)")


if __name__ == "__main__":
    main()
