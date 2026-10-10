"""Kinematic price prediction on TRADE-COUNT bars (not calendar time, not
volume) -- the most literal "match horizon to trading resolution" clock: a bar
is N trades, period, regardless of how much size or how long they took.

N (trades/bar) is the one asset-specific calibration: a small declared grid is
scored on TRAIN by prediction skill (RMSE ratio vs a no-change forecast -- the
same acceptance bar scripts/goog_prediction_viz.py established), the winner is
frozen, then the full diagnostic (prediction plot, derivatives, residuals,
residual stats) runs on TEST only. W (fit window) and H (prediction horizon)
are fixed bar-counts, not tuned, so N is the only thing calibrated per asset.

Run across GOOG/BTC/ETH from the start (NFTY excluded -- already established
too sparse for any intraday bar clock, not re-tested here).
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_lib"))
from tradebars import bucket_by_trade_count, concat_files
from kinematics import poly_derivs, poly_predict

from l2_strategy import _proj, autocorr1

N_GRID = [50, 100, 200]
W, H, O = 40, 1, 2      # H=1 tick: predict only the very next trade-bar
WARM = W + H + 5

GOOG_TRAIN_DAYS = 5
CRYPTO_TRAIN_FILES = 2
CRYPTO_WEEK = [f"2023-09-{d:02d}" for d in range(1, 8)]


def skill_stats(mid, N, w=W, h=H, o=O, lo=None, hi=None):
    pred = poly_predict(mid, w, h, o)
    n = len(mid)
    lo = lo if lo is not None else WARM
    hi = hi if hi is not None else n - h
    idx = np.arange(lo, hi)
    pred_move = pred[idx] - mid[idx]
    real_move = mid[idx + h] - mid[idx]
    resid = real_move - pred_move
    rmse_pred = np.sqrt(np.mean(resid ** 2))
    rmse_zero = np.sqrt(np.mean(real_move ** 2))
    corr = float(np.corrcoef(pred_move, real_move)[0, 1]) if pred_move.std() > 0 else float("nan")
    hit = float(100 * np.mean(np.sign(pred_move) == np.sign(real_move)))
    return dict(rmse_pred=rmse_pred, rmse_zero=rmse_zero,
               skill_ratio=rmse_pred / rmse_zero, corr=corr, hit=hit,
               pred_move=pred_move, real_move=real_move, resid=resid, idx=idx)


def calibrate_N(train_files):
    best, best_ratio = None, np.inf
    print("  TRAIN grid (trades/bar -> skill ratio, lower=better, <1 beats no-change):")
    for N in N_GRID:
        bars = concat_files(train_files, N)
        if bars is None or len(bars["mid"]) < WARM + 50:
            print(f"    N={N:>4}: too few bars, skipped")
            continue
        s = skill_stats(bars["mid"], N)
        print(f"    N={N:>4}: skill_ratio={s['skill_ratio']:.3f}  corr={s['corr']:+.4f}  "
              f"hit={s['hit']:.1f}%  ({len(bars['mid'])} bars)")
        if s["skill_ratio"] < best_ratio:
            best_ratio, best = s["skill_ratio"], N
    return best


def run_asset(name, train_files, test_files, zoom_bars=400):
    print(f"\n{'='*90}\n{name}\n{'='*90}")
    N = calibrate_N(train_files)
    if N is None:
        print(f"  {name}: no viable N, skipped"); return
    print(f"  FROZEN N = {N} trades/bar")

    bars = concat_files(test_files, N)
    if bars is None or len(bars["mid"]) < WARM + 50:
        print(f"  {name}: too few TEST bars at N={N}, skipped"); return
    mid = bars["mid"]
    s = skill_stats(mid, N)
    print(f"  TEST skill: RMSE(pred)={s['rmse_pred']:.4f}  RMSE(no-change)={s['rmse_zero']:.4f}  "
          f"skill_ratio={s['skill_ratio']:.3f}  corr={s['corr']:+.4f}  sign_hit={s['hit']:.1f}%")
    print(f"  residual: mean={s['resid'].mean():+.4f}  sd={s['resid'].std():.4f}  "
          f"skew={float(((s['resid']-s['resid'].mean())**3).mean()/s['resid'].std()**3):+.3f}")

    z1, z2 = poly_derivs(mid, W)
    pred = poly_predict(mid, W, H, O)

    # -- directional-accuracy diagnostics (per user: the move is not the
    # target; the polynomial is a smoothener -- check whether VELOCITY's sign
    # matches the RESIDUAL's sign, and whether residuals carry serial
    # structure, rather than judging the raw magnitude-forecast alone. --
    resid = s["resid"]
    z1_at = z1[s["idx"]]
    ac1_resid = autocorr1(resid)
    ac_lags = [float(np.corrcoef(resid[:-lag], resid[lag:])[0, 1]) for lag in range(1, 11)]
    sign_match = np.sign(z1_at) == np.sign(resid)
    match_rate = 100 * np.mean(sign_match)
    n_match = len(sign_match)
    se = np.sqrt(0.25 / n_match) * 100
    z_score = (match_rate - 50) / se
    signed_integral = float(resid.sum())
    abs_integral = float(np.abs(resid).sum())
    bias_ratio = signed_integral / abs_integral if abs_integral > 0 else float("nan")
    print(f"  residual AC(1) = {ac1_resid:+.4f}")
    print(f"  directional accuracy: sign(z1)==sign(residual) in {match_rate:.1f}% "
          f"of {n_match} bars (z={z_score:+.2f} vs 50%)")
    print(f"  signed integral / abs integral of residuals = {signed_integral:+.2f} / "
          f"{abs_integral:.2f} = {bias_ratio:+.5f}  (net bias as a fraction of total "
          f"residual activity; 0 = pure symmetric noise, +-1 = fully one-sided)")

    fig = plt.figure(figsize=(16, 17.5))
    gs = fig.add_gridspec(6, 2, height_ratios=[1.6, 1.3, 0.9, 0.9, 1.3, 1.3],
                          hspace=0.5, wspace=0.25)
    axA = fig.add_subplot(gs[0, :])
    axB = fig.add_subplot(gs[1, :])
    axC = fig.add_subplot(gs[2, :], sharex=axB)
    axD1 = fig.add_subplot(gs[3, 0]); axD2 = fig.add_subplot(gs[3, 1])
    axE1 = fig.add_subplot(gs[4, 0]); axE2 = fig.add_subplot(gs[4, 1])
    axF1 = fig.add_subplot(gs[5, 0]); axF2 = fig.add_subplot(gs[5, 1])

    # panel A: zoom -- sampled fit + extrapolation curves
    n = len(mid)
    z0 = min(WARM + 20, n - zoom_bars - H - 5)
    z1b = min(z0 + zoom_bars, n - H - 5)
    x = np.arange(n)
    axA.plot(x[z0:z1b], mid[z0:z1b], lw=0.9, color="#1f77b4", zorder=1)
    samples = np.linspace(z0 + W, z1b - H, 12).astype(int)
    for t in samples:
        win = mid[t - W + 1:t + 1]
        coef = _proj(W, O, W / 2) @ win
        tau_fit = np.arange(-(W - 1), 1).astype(float)
        fit = sum(coef[k] * tau_fit ** k for k in range(O + 1))
        tau_ex = np.arange(0, H + 1).astype(float)
        ex = sum(coef[k] * tau_ex ** k for k in range(O + 1))
        ex = ex - ex[0] + mid[t]
        axA.plot(t + np.arange(-(W - 1), 1), fit, color="#ff7f0e", lw=1.4, zorder=3)
        axA.plot(t + np.arange(0, H + 1), ex, color="#d62728", lw=1.2, ls="--", zorder=3)
        axA.scatter([t + H], [ex[-1]], color="#d62728", s=30, zorder=4)
        if t + H < n:
            axA.scatter([t + H], [mid[t + H]], color="k", marker="x", s=30, zorder=4)
    axA.set_title(f"{name}: N={N} trades/bar, W={W} H={H} O={O} -- fit (orange), "
                 f"extrapolation (red dashed), prediction (red dot), actual (black x)")
    axA.set_ylabel("mid")

    # panel B: predicted vs realized, full test
    show = slice(WARM, n - H)
    tgt = np.arange(WARM, n - H) + H
    axB.plot(x[show], mid[show], lw=0.7, color="#1f77b4", label="realized")
    axB.plot(tgt, pred[WARM:n - H], lw=0.5, color="#d62728", alpha=0.6,
            label="prediction (placed at target bar)")
    axB.legend(fontsize=8); axB.set_ylabel("mid")
    axB.set_title("prediction vs reality (test)")

    # panel C: residual over time
    axC.plot(tgt, s["resid"], lw=0.3, color="#7f7f7f", alpha=0.6)
    k = min(200, max(10, len(s["resid"]) // 20))
    if len(s["resid"]) > k:
        roll = np.convolve(s["resid"], np.ones(k) / k, "valid")
        axC.plot(tgt[k - 1:], roll, lw=1.2, color="#d62728", label=f"rolling mean ({k})")
    axC.axhline(0, color="k", lw=0.6); axC.legend(fontsize=8)
    axC.set_title("residual (realized - predicted move)")
    axC.set_xlabel("trade-bar index")

    # panel D: derivatives
    axD1.plot(x[show], z1[show], lw=0.5, color="#9467bd")
    axD1.axhline(0, color="k", lw=0.5)
    axD1.set_title("z1 (velocity)")
    axD2.plot(x[show], z2[show], lw=0.5, color="#2ca02c")
    axD2.axhline(0, color="k", lw=0.5)
    axD2.set_title("z2 (acceleration)")

    # panel E: skill scatter + residual histogram
    ss = np.random.default_rng(0).choice(len(s["pred_move"]), min(3000, len(s["pred_move"])),
                                         replace=False)
    axE1.scatter(s["pred_move"][ss], s["real_move"][ss], s=3, alpha=0.25, color="#1f77b4")
    lim = np.percentile(np.abs(np.concatenate([s["pred_move"], s["real_move"]])), 99) + 1e-9
    axE1.plot([-lim, lim], [-lim, lim], color="gray", lw=0.8, ls=":")
    axE1.set_xlim(-lim, lim); axE1.set_ylim(-lim, lim)
    axE1.set_xlabel("predicted move"); axE1.set_ylabel("realized move")
    axE1.set_title(f"corr {s['corr']:+.3f}, hit {s['hit']:.1f}%")

    axE2.hist(s["resid"], bins=60, color="#7f7f7f", edgecolor="white", lw=0.2)
    axE2.axvline(0, color="k", lw=0.8)
    axE2.set_title(f"residual dist (skill ratio {s['skill_ratio']:.3f})")

    # panel F1: velocity vs residual -- the directional-accuracy check.
    # NOT predicted-move-vs-realized-move (that conflates direction and
    # magnitude); this isolates whether the SIGN of the smoothed local slope
    # matches the sign of what's left after the polynomial's own extrapolation.
    ssF = np.random.default_rng(1).choice(n_match, min(3000, n_match), replace=False)
    colors = np.where(sign_match[ssF], "#2ca02c", "#d62728")
    axF1.scatter(z1_at[ssF], resid[ssF], s=3, alpha=0.3, c=colors)
    axF1.axhline(0, color="k", lw=0.6); axF1.axvline(0, color="k", lw=0.6)
    axF1.set_xlabel("z1 (velocity)"); axF1.set_ylabel("residual")
    axF1.set_title(f"directional accuracy: sign match {match_rate:.1f}% of "
                   f"{n_match} (z={z_score:+.2f} vs 50%)")

    axF2.bar(range(1, 11), ac_lags, color="#7f7f7f")
    axF2.axhline(0, color="k", lw=0.6)
    axF2.set_xlabel("lag (bars)"); axF2.set_ylabel("autocorr")
    axF2.set_title(f"residual autocorrelation, lags 1-10 (AC(1)={ac1_resid:+.4f})")

    out = os.path.join("reports", "l2_backtest", f"tradebar_kinematic_{name.lower()}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=105, bbox_inches="tight")
    print(f"  plot -> {out}")


def main():
    goog_files = sorted(glob.glob(os.path.join("data", "stock_data", "GOOG", "L2_top", "*", "*", "orderbook_goog_*.npz")))
    run_asset("GOOG", goog_files[:GOOG_TRAIN_DAYS], goog_files[GOOG_TRAIN_DAYS:])

    for sym in ("btcusdt", "ethusdt"):
        week = [os.path.join("data", "crypto_data", sym.upper(), d[:4], d[5:7], f"orderbook_{sym}_{d}.npz") for d in CRYPTO_WEEK]
        run_asset(sym.upper(), week[:CRYPTO_TRAIN_FILES], week[CRYPTO_TRAIN_FILES:])

    print("\nNFTY: not run -- already established too sparse for any intraday bar clock.")


if __name__ == "__main__":
    main()
