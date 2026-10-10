"""Two-regime strategy: passive market-making in MR/FLAT + kinematic taker in DIR.

Implements the architecture the regime detector was always meant to drive:

MM module (regime != DIR) -- earn the spread where adverse selection is low:
  * join the touch on both sides each valid snapshot
  * imbalance skew: if imb > skew_thr the book says price is about to rise, so
    stop offering (pull the ask); mirror for negative imb -- the defensive use
    of the directional signal
  * conservative fill model, two bounds reported:
      strict : our quote fills only when a trade prints THROUGH our price
      touch  : trades AT our price on the right side also fill (front-of-queue)
  * inventory capped at +/-INV_CAP units; quoting stops on the loaded side;
    position flattened taker when the regime flips to DIR or at day end

DIR module (regime == DIR) -- the adaptive-polynomial derivative logic:
  * weighted deg-2 fit over the trailing window -> d1 (velocity), d2 (accel),
    both z-scored by rolling realized vol so thresholds adapt per asset
  * position ladder: sign(d1) direction; |z1|>Z1 to be in a trade at all;
    z2 agreeing (accelerating)   -> size 2.0  (aggressive)
    z2 mildly opposing           -> size 1.0  (keep, less confident)
    z2 strongly opposing (>Z2)   -> size 0.0  (peak forming -> exit)
    re-enters if the move resumes; taker fills at t+1; retrade gap limits churn

Fees are parameterized per venue: maker fills earn rebate_bps, taker legs pay
taker_bps (both on notional; gross = both zero).
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l2_strategy import setup, compute_regime, DET, _proj

# ---- DIR (kinematic) params ----
POLY_W = 40
Z1_MIN = 1.0          # velocity z to hold a position
Z2_CONF = 0.5         # accel z agreeing -> aggressive size
Z2_PEAK = 1.0         # accel z opposing -> exit (peak/valley forming)
RETRADE_GAP = 5       # min snapshots between position changes
VOL_WIN = 200
# ---- MM params ----
SKEW_THR = 0.5        # |imb| beyond this -> pull the endangered quote
INV_CAP = 3
WARM = 300


def poly_derivs(mid, W=POLY_W):
    N = len(mid)
    C = swv(mid, W) @ _proj(W, 2, W / 2).T          # rows: c0,c1,c2 at window end
    d1 = np.full(N, np.nan); d2 = np.full(N, np.nan)
    d1[W - 1:] = C[:, 1]; d2[W - 1:] = 2 * C[:, 2]
    r = np.diff(mid, prepend=mid[0])
    var = np.convolve(r * r, np.ones(VOL_WIN) / VOL_WIN, "full")[:N]
    vol = np.sqrt(np.maximum(var, 1e-18))
    z1 = d1 / (vol + 1e-12)
    z2 = d2 / (vol / np.sqrt(POLY_W) + 1e-12)       # accel is a 2nd-order/smaller scale
    return np.nan_to_num(z1), np.nan_to_num(z2)


def dir_target(z1, z2):
    s = np.sign(z1) * (np.abs(z1) > Z1_MIN)
    agree = np.sign(z2) == np.sign(z1)
    size = np.where(agree & (np.abs(z2) > Z2_CONF), 2.0,
                    np.where(~agree & (np.abs(z2) > Z2_PEAK), 0.0, 1.0))
    return s * size


def run_day(D, rebate_bps=0.0, taker_bps=0.0, fill_mode="strict",
            trades_by_snap=None):
    """One combined pass. Returns dict of MM and DIR results (all bps of mid)."""
    N = D["N"]; mid = D["mid"]; bb = D["bb"]; ba = D["ba"]
    imb = D["imb"]; reg = D["reg"]; valid = D["valid"]
    med = np.median(mid); reb = rebate_bps / 1e4 * med; tak = taker_bps / 1e4 * med
    z1, z2 = poly_derivs(mid)
    tgt = dir_target(z1, z2)

    cash = 0.0; inv = 0.0                            # MM book
    mm_fills = 0; mm_pnl_ts = np.zeros(N)
    dpos = 0.0; dcash = 0.0; dir_trades = 0; last_chg = -RETRADE_GAP
    dir_pnl_ts = np.zeros(N)
    mm_snaps = 0

    for t in range(WARM, N - 1):
        te = t + 1
        if not valid[t]:
            mm_pnl_ts[t] = cash + inv * mid[t]; dir_pnl_ts[t] = dcash + dpos * mid[t]
            continue
        if reg[t] != 1:
            # ---------- flatten any DIR position on regime exit ----------
            if dpos != 0 and valid[te]:
                px = bb[te] if dpos > 0 else ba[te]
                dcash += dpos * px - abs(dpos) * tak; dpos = 0; dir_trades += 1
            # ---------------------- market-make ----------------------
            mm_snaps += 1
            qb = bb[t] if (imb[t] > -SKEW_THR and inv < INV_CAP) else None
            qa = ba[t] if (imb[t] < SKEW_THR and inv > -INV_CAP) else None
            # one posted unit per side per snapshot: a quote fills once, then it
            # is gone until re-posted next snapshot (no stale-quote conveyor)
            for (tp, tsd) in trades_by_snap.get(t, ()):
                if qb is not None and (tp < qb or
                        (fill_mode == "touch" and tp == qb and tsd < 0)):
                    cash -= qb; cash += reb; inv += 1; mm_fills += 1; qb = None
                if qa is not None and (tp > qa or
                        (fill_mode == "touch" and tp == qa and tsd > 0)):
                    cash += qa; cash += reb; inv -= 1; mm_fills += 1; qa = None
        else:
            # ---------- flatten MM inventory on entering DIR ----------
            if inv != 0 and valid[te]:
                px = bb[te] if inv > 0 else ba[te]
                cash += inv * px - abs(inv) * tak; inv = 0
            # ---------------------- directional ----------------------
            want = tgt[t]
            if want != dpos and (t - last_chg) >= RETRADE_GAP and valid[te]:
                dq = want - dpos
                px = ba[te] if dq > 0 else bb[te]
                dcash -= dq * px + abs(dq) * tak
                dpos = want; last_chg = t; dir_trades += 1
        mm_pnl_ts[t] = cash + inv * mid[t]
        dir_pnl_ts[t] = dcash + dpos * mid[t]
    # end of day: flatten both books
    if inv != 0:  cash += inv * (bb[-1] if inv > 0 else ba[-1]) - abs(inv) * tak
    if dpos != 0: dcash += dpos * (bb[-1] if dpos > 0 else ba[-1]) - abs(dpos) * tak
    mm_pnl_ts[-1] = cash; dir_pnl_ts[-1] = dcash

    to_bps = lambda x: x / med * 1e4
    return dict(mm_pnl=to_bps(cash), mm_fills=mm_fills, mm_snaps=mm_snaps,
                dir_pnl=to_bps(dcash), dir_trades=dir_trades,
                total=to_bps(cash + dcash),
                mm_pnl_ts=to_bps(mm_pnl_ts), dir_pnl_ts=to_bps(dir_pnl_ts))


def trades_map(path, N):
    raw = np.load(path, allow_pickle=True)
    m = {}
    for p, s, b in zip(raw["tr_price"], raw["tr_side"], raw["tr_bookidx"]):
        m.setdefault(int(b), []).append((float(p), int(s)))
    return m


def main():
    files = sys.argv[1:] or sorted(glob.glob(os.path.join("data", "stock_data", "GOOG", "L2_top", "*", "*",
                                                          "orderbook_googl_*.npz")))
    scen = [("gross", 0.0, 0.0), ("nasdaq", 0.10, 0.15)]
    print(f"{'day':<22}{'sprd':>6}{'DIR%':>6}{'MM%':>6} | "
          f"{'MMfil':>6}{'MM-strict':>10}{'MM-touch':>10} | "
          f"{'DIRtr':>6}{'DIRpnl':>9} | {'tot-str':>9}{'tot-net':>9}")
    agg = {"mm_s": [], "mm_t": [], "dir": [], "tot_s": [], "tot_net": []}
    for fp in files:
        D = setup(fp)
        tm = trades_map(fp, D["N"])
        med = np.median(D["mid"])
        sp = float(np.median(D["spread"][D["valid"]]) / med * 1e4)
        rs = run_day(D, 0, 0, "strict", tm)
        rt = run_day(D, 0, 0, "touch", tm)
        rn = run_day(D, scen[1][1], scen[1][2], "strict", tm)
        dirp = float(100 * np.mean(D["reg"] == 1))
        mmp = float(100 * np.mean(D["reg"] != 1))
        print(f"{D['symbol']:<22}{sp:>6.2f}{dirp:>6.1f}{mmp:>6.1f} | "
              f"{rs['mm_fills']:>6}{rs['mm_pnl']:>+10.2f}{rt['mm_pnl']:>+10.2f} | "
              f"{rs['dir_trades']:>6}{rs['dir_pnl']:>+9.2f} | "
              f"{rs['total']:>+9.2f}{rn['total']:>+9.2f}", flush=True)
        agg["mm_s"].append(rs["mm_pnl"]); agg["mm_t"].append(rt["mm_pnl"])
        agg["dir"].append(rs["dir_pnl"]); agg["tot_s"].append(rs["total"])
        agg["tot_net"].append(rn["total"])
    if agg["tot_s"]:
        n = len(agg["tot_s"])
        def line(tag, xs):
            xs = np.array(xs)
            sh = xs.mean() / xs.std() if xs.std() > 0 else 0
            print(f"  {tag:<18} mean {xs.mean():+8.2f} bps/day  "
                  f"daily Sharpe {sh:+.2f} (t={sh*np.sqrt(n):+.1f})  "
                  f"pos days {100*np.mean(xs>0):.0f}%")
        print(f"\n== {n} days ==")
        line("MM strict (gross)", agg["mm_s"])
        line("MM touch  (gross)", agg["mm_t"])
        line("DIR       (gross)", agg["dir"])
        line("TOTAL strict",      agg["tot_s"])
        line("TOTAL net nasdaq",  agg["tot_net"])


if __name__ == "__main__":
    main()
