"""Deep diagnostics for the MM+DIR system: blame assignment, not PnL.

Answers, with numbers, over per-day L2 files (default: the GOOG month):
  1. Are regimes detected 'correctly'? For each label, FUTURE-window realized
     stats: drift ratio |move|/(vol*sqrt(W)), variance ratio, autocorr. DIR
     labels should precede trending stats, MR reverting stats.
  2. DIR: is direction predicted correctly (sign hit rates), are shapes
     predicted correctly (fwd move conditioned on each kinematic state), does
     the peak warning actually precede the move stalling, and how do executed
     episodes fare (win rate, direction-correct rate, exit reasons)?
  3. MM: markout curve of every fill (PnL marked at +1..+60 snaps, ~1s..55s),
     split by regime label and by the imbalance-skew rule (fills on quotes the
     rule KEEPS vs fills on quotes it would have PULLED). Adverse selection is
     the decay of that curve; the skew split shows if the defense works.
  4. Action x regime: each module's per-action EV in each regime, including
     where it normally would not run.
All gross, no fees. Snapshot cadence ~0.9s on the GOOG files.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_lib"))
from l2_strategy import setup, variance_ratio, autocorr1
from l2_mmdir import (poly_derivs, dir_target, trades_map, WARM,
                      SKEW_THR, RETRADE_GAP, Z1_MIN, Z2_CONF, Z2_PEAK)

FWD_HORIZONS = [10, 30, 60]          # snapshots (~9s / 27s / 55s)
MARKOUTS = [1, 5, 10, 30, 60]
FUT_W = 100
REG_NAME = {1: "DIR", -1: "MR", 0: "FLAT"}


def to_bps(x, med): return x / med * 1e4


def fwd(mid, h):
    out = np.full(len(mid), np.nan)
    out[:len(mid) - h] = mid[h:] - mid[:-h]
    return out


# ---------- 1. regime label validation (future-window stats) ----------
def regime_future_stats(D, acc):
    mid, reg = D["mid"], D["reg"]
    med = np.median(mid)
    r = np.diff(mid, prepend=mid[0])
    for t in range(WARM, D["N"] - FUT_W, 25):        # sample every 25th snap
        seg = r[t + 1:t + 1 + FUT_W]
        sd = seg.std()
        drift = abs(mid[t + FUT_W] - mid[t]) / (sd * np.sqrt(FUT_W) + 1e-12)
        acc.setdefault(reg[t], dict(dr=[], vr=[], ac=[], mv=[]))
        a = acc[reg[t]]
        a["dr"].append(drift)
        a["vr"].append(variance_ratio(seg, 8))
        a["ac"].append(autocorr1(seg))
        a["mv"].append(to_bps(abs(mid[t + FUT_W] - mid[t]), med))


# ---------- 2. DIR kinematics ----------
def dir_diag(D, acc):
    mid, reg, valid = D["mid"], D["reg"], D["valid"]
    med = np.median(mid); N = D["N"]
    z1, z2 = poly_derivs(mid)
    # dynamic thresholds: quantiles of each signal's own |distribution| this day
    q1 = np.quantile(np.abs(z1[WARM:]), 0.70)
    q2c = np.quantile(np.abs(z2[WARM:]), 0.60)
    q2p = np.quantile(np.abs(z2[WARM:]), 0.85)
    acc.setdefault("thr", []).append((q1, q2c, q2p))
    s = np.sign(z1) * (np.abs(z1) > q1)
    state = np.zeros(N)                                # 0 none,2 accel,1 steady,-1 peak-warn
    agree = np.sign(z2) == np.sign(z1)
    peak = (s != 0) & ~agree & (np.abs(z2) > q2p)
    accel = (s != 0) & agree & (np.abs(z2) > q2c) & ~peak
    state[accel] = 2
    state[(s != 0) & ~accel & ~peak] = 1
    state[peak] = -1
    # shape test uses sign(z1) without the magnitude gate: "peak" needs to be
    # testable even after velocity has decayed below its entry threshold
    sh = np.sign(z1)
    shape = np.zeros(N)
    sh_agree = np.sign(z2) == sh
    sh_peak = (sh != 0) & ~sh_agree & (np.abs(z2) > q2p)
    sh_accel = (sh != 0) & sh_agree & (np.abs(z2) > q2c) & ~sh_peak
    shape[sh_accel] = 2
    shape[(sh != 0) & ~sh_accel & ~sh_peak] = 1
    shape[sh_peak] = -1
    F = {h: fwd(mid, h) for h in FWD_HORIZONS}
    for st in (2, 1, -1):
        m = (shape == st) & (reg == 1) & valid
        m[:WARM] = False; m[N - max(FWD_HORIZONS):] = False
        if m.sum() == 0: continue
        a = acc.setdefault(st, {h: [] for h in FWD_HORIZONS} | {"hit": []})
        for h in FWD_HORIZONS:
            a[h].extend(to_bps(sh[m] * F[h][m], med).tolist())
        a["hit"].extend((sh[m] * F[30][m] > 0).tolist())

    # executed episodes with exit reasons (ladder replay, reg-gated, quantile thr)
    tgt = s * np.where(state == 2, 2.0, np.where(state == -1, 0.0, 1.0))
    tgt[s == 0] = 0.0
    pos = 0.0; e_t = 0; last = -RETRADE_GAP; cash = 0.0
    for t in range(WARM, N - 61):
        want = tgt[t] if (reg[t] == 1 and valid[t]) else 0.0
        if want == pos or (t - last) < RETRADE_GAP:
            continue
        px = mid[t + 1]
        if pos == 0:                                   # open episode
            e_t = t; cash = -want * px
        elif want == 0 or np.sign(want) != np.sign(pos):   # close (maybe reverse)
            cash += pos * px
            if reg[t] != 1: reason = "regime"
            elif state[t] == -1: reason = "peak"
            else: reason = "signal"
            post = to_bps(np.sign(pos) * (mid[min(t + 31, N - 1)] - px), med)
            acc.setdefault("ep", []).append(
                dict(pnl=to_bps(cash, med), win=cash > 0,
                     dirok=bool(np.sign(pos) * (px - mid[e_t + 1]) > 0),
                     reason=reason, post=post, reg=int(D["reg"][e_t])))
            cash = -want * px
            if want: e_t = t
        else:                                          # resize within episode
            cash -= (want - pos) * px
        pos = want; last = t


# ---------- 3. MM markouts ----------
def mm_diag(D, tm, acc, mode="touch"):
    mid, bb, ba = D["mid"], D["bb"], D["ba"]
    reg, valid, imb = D["reg"], D["valid"], D["imb"]
    med = np.median(mid); N = D["N"]
    for t in range(WARM, N - 62):
        if not valid[t]: continue
        qb, qa = bb[t], ba[t]
        got_b = got_a = False
        for tp, tsd in tm.get(t, ()):
            if not got_b and (tp < qb or (mode == "touch" and tp == qb and tsd < 0)):
                got_b = True
                kept = imb[t] > -SKEW_THR            # skew rule would keep this bid?
                rec = dict(reg=int(reg[t]), kept=bool(kept),
                           cap=to_bps(mid[t] - qb, med))
                for h in MARKOUTS:
                    rec[h] = to_bps(mid[t + 1 + h] - qb, med)
                acc.append(rec)
            if not got_a and (tp > qa or (mode == "touch" and tp == qa and tsd > 0)):
                got_a = True
                kept = imb[t] < SKEW_THR
                rec = dict(reg=int(reg[t]), kept=bool(kept),
                           cap=to_bps(qa - mid[t], med))
                for h in MARKOUTS:
                    rec[h] = to_bps(qa - mid[t + 1 + h], med)
                acc.append(rec)
            if got_a and got_b: break


def main():
    files = sys.argv[1:] or sorted(glob.glob(os.path.join("data", "stock_data", "GOOG", "L2_top", "*", "*",
                                                          "orderbook_goog_*.npz")))
    reg_acc, dir_acc, mm_acc = {}, {}, []
    for fp in files:
        D = setup(fp)
        tm = trades_map(fp, D["N"])
        regime_future_stats(D, reg_acc)
        dir_diag(D, dir_acc)
        mm_diag(D, tm, mm_acc)
        print(f"  processed {D['symbol']}", flush=True)

    print(f"\n{'='*90}\n1. REGIME LABEL VALIDATION (future {FUT_W}-snap window after each label)\n{'='*90}")
    print(f"{'label':<6}{'n':>7}{'fut drift-ratio':>16}{'fut VR(8)':>11}{'fut AC1':>9}{'fut |move| bps':>15}")
    for k in (1, -1, 0):
        if k not in reg_acc: continue
        a = reg_acc[k]
        print(f"{REG_NAME[k]:<6}{len(a['dr']):>7}{np.mean(a['dr']):>16.3f}"
              f"{np.mean(a['vr']):>11.3f}{np.mean(a['ac']):>9.3f}{np.mean(a['mv']):>15.2f}")

    print(f"\n{'='*90}\n2. DIR KINEMATICS (inside DIR regime; moves aligned to predicted direction, bps)\n{'='*90}")
    names = {2: "accel (z1&z2 agree)", 1: "steady (hold 1x)", -1: "PEAK WARNING (exit)"}
    print(f"{'state':<24}{'n':>8}{'fwd10':>8}{'fwd30':>8}{'fwd60':>8}{'hit30%':>8}")
    for st in (2, 1, -1):
        if st not in dir_acc: continue
        a = dir_acc[st]
        print(f"{names[st]:<24}{len(a[30]):>8}{np.mean(a[10]):>+8.2f}{np.mean(a[30]):>+8.2f}"
              f"{np.mean(a[60]):>+8.2f}{100*np.mean(a['hit']):>8.1f}")
    eps = dir_acc.get("ep", [])
    if eps:
        pnl = np.array([e["pnl"] for e in eps])
        print(f"\n  executed episodes: {len(eps)}  win rate {100*np.mean([e['win'] for e in eps]):.1f}%  "
              f"direction-correct {100*np.mean([e['dirok'] for e in eps]):.1f}%  "
              f"avg pnl {pnl.mean():+.2f} bps")
        for rsn in ("peak", "regime", "signal"):
            sub = [e for e in eps if e["reason"] == rsn]
            if not sub: continue
            print(f"    exit={rsn:<7} n={len(sub):>4}  avg pnl {np.mean([e['pnl'] for e in sub]):+7.2f} bps"
                  f"  post-exit 30-snap move (aligned w/ old pos) {np.mean([e['post'] for e in sub]):+6.2f} bps")

    print(f"\n{'='*90}\n3. MM MARKOUT CURVE (touch fills; bps per fill; capture=half-spread at fill)\n{'='*90}")
    if mm_acc:
        cap = np.mean([r["cap"] for r in mm_acc])
        line = f"  ALL fills (n={len(mm_acc):,})  capture {cap:+.2f} | markout:"
        for h in MARKOUTS:
            line += f"  +{h}snap {np.mean([r[h] for r in mm_acc]):+.2f}"
        print(line)
        for k in (1, -1, 0):
            sub = [r for r in mm_acc if r["reg"] == k]
            if not sub: continue
            print(f"  reg={REG_NAME[k]:<5} (n={len(sub):>6,})  " +
                  "  ".join(f"+{h}:{np.mean([r[h] for r in sub]):+.2f}" for h in MARKOUTS))
        for kept, tag in ((True, "KEPT by skew rule"), (False, "would-be PULLED")):
            sub = [r for r in mm_acc if r["kept"] == kept]
            if not sub: continue
            print(f"  {tag:<18} (n={len(sub):>6,})  " +
                  "  ".join(f"+{h}:{np.mean([r[h] for r in sub]):+.2f}" for h in MARKOUTS))

    print(f"\n{'='*90}\n4. ACTION x REGIME (per-action EV, bps)\n{'='*90}")
    for k in (1, -1, 0):
        sub = [r for r in mm_acc if r["reg"] == k]
        if sub:
            print(f"  MM fill markout@60 in {REG_NAME[k]:<5}: "
                  f"{np.mean([r[60] for r in sub]):+.3f} bps  (n={len(sub):,})")
    if eps:
        for k in (1,):
            sub = [e for e in eps if e["reg"] == k]
            if sub:
                print(f"  DIR episode pnl in {REG_NAME[k]:<5}: "
                      f"{np.mean([e['pnl'] for e in sub]):+.3f} bps  (n={len(sub)})")


if __name__ == "__main__":
    main()
