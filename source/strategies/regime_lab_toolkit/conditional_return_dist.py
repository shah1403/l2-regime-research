"""Conditional return distributions: P(next tick's return | sign of this tick),
built and kept SEPARATE for positive- and negative-conditioning ticks (not
pooled into one correlation coefficient), with large moves (>1.5 sigma, a
train-frozen reference) excluded from the distribution as "shock" events, not
"normal regime" behavior.

This is closely related to lag-1 autocorrelation (already measured many times
this session, consistently near-zero/weakly-negative at short horizons) but
strictly more general: it can reveal an ASYMMETRY between the up-conditioned
and down-conditioned response that a single pooled correlation number would
average away, and it separates "normal regime" ticks from shock ticks before
estimating anything, which nothing in this program has done before.

"Tick" = per-snapshot mid-price change (the finest, most literal resolution
available uniformly across every asset via l2_strategy.setup()). Day
boundaries are respected (returns are never computed across a file boundary).
sigma_ref (the 1.5 sigma filter threshold) is computed ONCE from TRAIN data,
frozen, then reused unchanged on TEST -- same discipline as everything else.

Interpretation: distribution centered near 0 with no skew -> mean-reverting/
memoryless. Skew in the SAME direction as the conditioning tick's sign ->
genuine continuation/momentum. Kept separate by sign specifically to let a
positive-conditioned skew differ from a negative-conditioned skew.
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
from l2_strategy import setup

SIGMA_MULT = 1.5
GOOG_TRAIN_DAYS = 5
CRYPTO_TRAIN_FILES = 2
CRYPTO_WEEK = [f"2023-09-{d:02d}" for d in range(1, 8)]


def tick_returns(fp: str) -> np.ndarray:
    """Per-day raw tick (snapshot-to-snapshot) mid returns. One file = one day
    -- never differenced across a day boundary."""
    D = setup(fp)
    return np.diff(D["mid"])


def build_conditional(files: list, sigma_ref: float = None):
    """Returns (pos_bucket, neg_bucket, sigma_ref). If sigma_ref is None,
    computes it from this data (call this on TRAIN); otherwise uses the given
    frozen value (call this on TEST with the TRAIN-derived sigma_ref)."""
    all_r = [tick_returns(fp) for fp in files]
    if sigma_ref is None:
        pooled = np.concatenate(all_r)
        sigma_ref = float(pooled.std())

    pos, neg = [], []
    for r in all_r:
        sign = np.sign(r)
        nxt = r[1:]
        cur_sign = sign[:-1]
        keep = np.abs(nxt) <= SIGMA_MULT * sigma_ref     # exclude shocks
        for s, x, k in zip(cur_sign, nxt, keep):
            if not k or s == 0:
                continue
            (pos if s > 0 else neg).append(x)
    return np.array(pos), np.array(neg), sigma_ref


IQR_MULT = 1.5


def iqr_fences(pooled: np.ndarray, k: float = IQR_MULT):
    q1, q3 = np.percentile(pooled, [25, 75])
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


def build_conditional_iqr(files: list, fences: tuple = None):
    """Same as build_conditional, but outliers are excluded via Tukey's rule
    (outside [Q1-1.5*IQR, Q3+1.5*IQR]) instead of a std-dev fence -- robust to
    the very outliers being excluded, unlike std which the outliers inflate.
    `fences` is (lo, hi); None -> compute from this data (TRAIN), else reuse
    the frozen TRAIN-derived fences (TEST)."""
    all_r = [tick_returns(fp) for fp in files]
    if fences is None:
        pooled = np.concatenate(all_r)
        fences = iqr_fences(pooled)
    lo, hi = fences

    pos, neg = [], []
    for r in all_r:
        sign = np.sign(r)
        nxt = r[1:]
        cur_sign = sign[:-1]
        keep = (nxt >= lo) & (nxt <= hi)
        for s, x, k in zip(cur_sign, nxt, keep):
            if not k or s == 0:
                continue
            (pos if s > 0 else neg).append(x)
    return np.array(pos), np.array(neg), fences


def per_day_means(files, keep_fn):
    """Per-day (not per-tick) pos/neg conditional means. Each day-file contributes
    exactly ONE pos-mean and ONE neg-mean, regardless of how many ticks it has --
    this is the fix for treating thousands of autocorrelated intraday ticks as if
    they were independent draws (they are not: consecutive ticks within a day are
    serially correlated, so pooling them all into one sample and computing
    se=sd/sqrt(n) understates the true standard error, exactly the "naive SEs are
    too small" problem documented for overlapping observations in paper/nce_paper.tex).
    Treating each TRADING DAY as the independent unit is the honest correction,
    matching the "daily Sharpe(d)" convention used throughout this project.
    keep_fn(nxt_array) -> bool mask, the same shock filter as the pooled version."""
    day_pos, day_neg = [], []
    for fp in files:
        r = tick_returns(fp)
        sign = np.sign(r)
        nxt = r[1:]
        cur_sign = sign[:-1]
        keep = keep_fn(nxt)
        p = nxt[keep & (cur_sign > 0)]
        n = nxt[keep & (cur_sign < 0)]
        day_pos.append(p.mean() if len(p) else np.nan)
        day_neg.append(n.mean() if len(n) else np.nan)
    return np.array(day_pos), np.array(day_neg)


def report_day_level(label, day_pos, day_neg):
    print(f"  {label} -- DAY-LEVEL (each trading day = one independent obs, honest SE):")
    for name, x in (("pos-conditioned", day_pos), ("neg-conditioned", day_neg)):
        x = x[~np.isnan(x)]
        if len(x) < 2:
            print(f"    {name}: only {len(x)} day(s) -- can't estimate a day-level SE at all")
            continue
        mean, sd = x.mean(), x.std(ddof=1)
        se = sd / np.sqrt(len(x))
        t = mean / se if se > 0 else 0.0
        print(f"    {name:<16} n_days={len(x):>2}  mean={mean:+.5g}  (t={t:+.2f}, se={se:.5g})  "
              f"daily means={np.round(x, 6).tolist()}")


def skewness(x):
    if len(x) < 3 or x.std() == 0:
        return float("nan")
    return float(((x - x.mean()) ** 3).mean() / x.std() ** 3)


def report(label, pos, neg, filter_desc):
    print(f"  {label} ({filter_desc}, excluded-shock filter active):")
    for name, x in (("pos-conditioned", pos), ("neg-conditioned", neg)):
        if len(x) < 30:
            print(f"    {name}: only {len(x)} obs, too few"); continue
        mean, sd, sk = x.mean(), x.std(), skewness(x)
        p_up = 100 * np.mean(x > 0)
        se = sd / np.sqrt(len(x))
        t_mean = mean / se if se > 0 else 0.0
        print(f"    {name:<16} n={len(x):>7}  mean={mean:+.5g} (t={t_mean:+.2f})  "
              f"skew={sk:+.3f}  P(next>0)={p_up:5.1f}%")
    if len(pos) > 30 and len(neg) > 30:
        diff = pos.mean() - neg.mean()
        se_diff = np.sqrt(pos.var() / len(pos) + neg.var() / len(neg))
        t_diff = diff / se_diff if se_diff > 0 else 0.0
        print(f"    pos-mean minus neg-mean = {diff:+.5g}  (t={t_diff:+.2f}) "
              f"-- asymmetry between the two conditioned distributions")


def plot_asset(name, pos_tr, neg_tr, pos_te, neg_te, sigma_ref):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, pos, neg, tag in ((axes[0], pos_tr, neg_tr, "TRAIN"),
                              (axes[1], pos_te, neg_te, "TEST")):
        lim = SIGMA_MULT * sigma_ref
        bins = np.linspace(-lim, lim, 60)
        ax.hist(pos, bins=bins, alpha=0.5, color="#2ca02c", density=True,
               label=f"after +tick (skew {skewness(pos):+.2f})")
        ax.hist(neg, bins=bins, alpha=0.5, color="#d62728", density=True,
               label=f"after -tick (skew {skewness(neg):+.2f})")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"{name} {tag}"); ax.legend(fontsize=8)
        ax.set_xlabel("next tick return")
    fig.tight_layout()
    out = os.path.join("reports", "l2_backtest", f"conditional_dist_{name.lower()}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=105)
    print(f"  plot -> {out}")


def run_asset(name, train_files, test_files):
    print(f"\n{'='*90}\n{name} -- std-dev fence (as originally specified)\n{'='*90}")
    pos_tr, neg_tr, sigma_ref = build_conditional(train_files)
    report("TRAIN (pooled-tick, naive SE)", pos_tr, neg_tr, f"sigma_ref={sigma_ref:.5g}")
    pos_te, neg_te, _ = build_conditional(test_files, sigma_ref=sigma_ref)
    report("TEST (pooled-tick, naive SE, frozen)", pos_te, neg_te, f"sigma_ref={sigma_ref:.5g}")
    plot_asset(name, pos_tr, neg_tr, pos_te, neg_te, sigma_ref)

    std_keep = lambda nxt: np.abs(nxt) <= SIGMA_MULT * sigma_ref
    dp_tr, dn_tr = per_day_means(train_files, std_keep)
    report_day_level("TRAIN", dp_tr, dn_tr)
    dp_te, dn_te = per_day_means(test_files, std_keep)
    report_day_level("TEST (frozen)", dp_te, dn_te)

    print(f"\n{name} -- IQR fence (Tukey's rule, robust alternative)")
    pos_tr_i, neg_tr_i, fences = build_conditional_iqr(train_files)
    report("TRAIN (pooled-tick, naive SE)", pos_tr_i, neg_tr_i, f"fences=[{fences[0]:.5g},{fences[1]:.5g}]")
    pos_te_i, neg_te_i, _ = build_conditional_iqr(test_files, fences=fences)
    report("TEST (pooled-tick, naive SE, frozen)", pos_te_i, neg_te_i, f"fences=[{fences[0]:.5g},{fences[1]:.5g}]")

    lo, hi = fences
    iqr_keep = lambda nxt: (nxt >= lo) & (nxt <= hi)
    dp_tr_i, dn_tr_i = per_day_means(train_files, iqr_keep)
    report_day_level("TRAIN", dp_tr_i, dn_tr_i)
    dp_te_i, dn_te_i = per_day_means(test_files, iqr_keep)
    report_day_level("TEST (frozen)", dp_te_i, dn_te_i)


def main():
    goog_files = sorted(glob.glob(os.path.join("data", "stock_data", "GOOG", "L2_top", "*", "*", "orderbook_goog_*.npz")))
    run_asset("GOOG", goog_files[:GOOG_TRAIN_DAYS], goog_files[GOOG_TRAIN_DAYS:])

    for sym in ("btcusdt", "ethusdt", "solusdt", "xrpusdt"):
        week = [os.path.join("data", "crypto_data", sym.upper(), d[:4], d[5:7], f"orderbook_{sym}_{d}.npz") for d in CRYPTO_WEEK]
        run_asset(sym.upper(), week[:CRYPTO_TRAIN_FILES], week[CRYPTO_TRAIN_FILES:])

    nfty_files = sorted(glob.glob(os.path.join("data", "stock_data", "NFTY_ETF", "*", "*", "orderbook_nfty_*.npz")))
    run_asset("NFTY", nfty_files[:GOOG_TRAIN_DAYS], nfty_files[GOOG_TRAIN_DAYS:])


if __name__ == "__main__":
    main()
