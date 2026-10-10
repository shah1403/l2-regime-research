"""Walk-forward parameter tuning for the regime strategy, per the user's protocol:

  Protocol A : tune everything on day 1 ONLY -> freeze -> test days 2..7
  Protocol B : rolling repeat -- tune on day k -> test day k+1 (6 OOS days)
  Baseline   : default parameters (DET + thr=0.4/hold=15) on the same test days

Tuned knobs (gross PnL t-stat on the tuning day, min 10 trades):
  regime detector : vr_hi {1.05,1.10}  ac_min {0.02,0.06}  cost_mult {0.25,0.5,1.0}
  strategy        : thr {dyn85, dyn70, 0.4}   hold {15, 30}
  -> 12 detector combos x 6 strategy combos = 72 evals per tuning day

Everything is gross (no fees). Data: sampled Binance npz (data/l2), one week.
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_lib"))
from l2_strategy import setup, backtest, compute_regime, DET
from l2_diagnose import dynamic_thr

SYMBOLS = ["btcusdt", "ethusdt", "solusdt", "xrpusdt"]
DAYS = [f"2023-09-{d:02d}" for d in range(1, 8)]

DET_GRID = dict(vr_hi=[1.05, 1.10], ac_min=[0.02, 0.06], cost_mult=[0.25, 0.5, 1.0])
THR_GRID = ["dyn85", "dyn70", "fix04"]
HOLD_GRID = [15, 30]
MIN_TRADES = 10

_setups: dict[str, dict] = {}


def get_day(sym, day):
    key = f"{sym}:{day}"
    if key not in _setups:
        path = os.path.join("data", "crypto_data", sym.upper(), day[:4], day[5:7], f"orderbook_{sym}_{day}.npz")
        D = setup(path)
        D["_thr"] = {"dyn85": dynamic_thr(D["imb"], q=0.85),
                     "dyn70": dynamic_thr(D["imb"], q=0.70),
                     "fix04": 0.4}
        _setups[key] = D
    return _setups[key]


def run_cfg(D, det, thr_key, hold):
    if det is not None:
        D = dict(D)
        D["reg"] = compute_regime(D["mid"], D["spread"], D["tnf"],
                                  **{**DET, **det})
    r = backtest(D, thr=D["_thr"][thr_key], hold=hold)
    return r


def tune(sym, day):
    """Best (det, thr, hold) by gross t-stat on ONE day."""
    D = get_day(sym, day)
    best, best_score, best_r = None, -np.inf, None
    for vr_hi, ac_min, cm in itertools.product(*DET_GRID.values()):
        det = dict(vr_hi=vr_hi, ac_min=ac_min, cost_mult=cm)
        Dd = dict(D)
        Dd["reg"] = compute_regime(D["mid"], D["spread"], D["tnf"], **{**DET, **det})
        for thr_key, hold in itertools.product(THR_GRID, HOLD_GRID):
            r = backtest(Dd, thr=D["_thr"][thr_key], hold=hold)
            score = r["tstat"] if r["trades"] >= MIN_TRADES else -np.inf
            if score > best_score:
                best_score, best = score, dict(det=det, thr=thr_key, hold=hold)
                best_r = r
    return best, best_r


def fmt_run(r):
    return (f"Sh {r['per_trade_sharpe']:+.3f} t {r['tstat']:+5.1f} "
            f"edge {r['edge_bps']:+6.2f}bps ({r['trades']:4d}tr)")


def main():
    for sym in SYMBOLS:
        print(f"\n{'='*100}\n{sym.upper()}  --  tune day = {DAYS[0]}, test = {DAYS[1]}..{DAYS[-1]}\n{'='*100}",
              flush=True)
        try:
            cfg, r_is = tune(sym, DAYS[0])
        except FileNotFoundError as e:
            print(f"  !! missing data: {e}"); continue
        print(f"  tuned cfg: {cfg['det']}  thr={cfg['thr']} hold={cfg['hold']}  "
              f"[in-sample {fmt_run(r_is)}]", flush=True)

        print(f"  {'day':<12} {'default params':>42} | {'tuned on day1 (A)':>42}")
        agg = {"def": [], "A": []}
        for day in DAYS[1:]:
            try:
                D = get_day(sym, day)
            except FileNotFoundError:
                print(f"  {day:<12} missing"); continue
            r_def = backtest(D, thr=0.4, hold=15)
            r_A = run_cfg(D, cfg["det"], cfg["thr"], cfg["hold"])
            agg["def"].append(r_def); agg["A"].append(r_A)
            print(f"  {day:<12} {fmt_run(r_def):>42} | {fmt_run(r_A):>42}", flush=True)

        # Protocol B: rolling tune day k -> test day k+1
        rolls = []
        for k in range(len(DAYS) - 1):
            try:
                cfg_k, _ = tune(sym, DAYS[k])
                D = get_day(sym, DAYS[k + 1])
            except FileNotFoundError:
                continue
            rolls.append(run_cfg(D, cfg_k["det"], cfg_k["thr"], cfg_k["hold"]))
            print(f"  [B] tune {DAYS[k]} -> test {DAYS[k+1]}: {fmt_run(rolls[-1])}",
                  flush=True)

        def summarize(tag, runs):
            if not runs: return
            tr = np.concatenate([r["trade_pnls"] for r in runs if r["trades"] > 1]) \
                if any(r["trades"] > 1 for r in runs) else np.array([0.0])
            sh = tr.mean() / tr.std() if tr.std() > 0 else 0.0
            print(f"  >> {tag:<22} pooled OOS: Sharpe {sh:+.3f}  "
                  f"t {sh*np.sqrt(len(tr)):+6.1f}  ({len(tr)} trades, "
                  f"{100*np.mean(tr>0):.0f}% win)")
        summarize("default", agg["def"])
        summarize("tuned-day1 (A)", agg["A"])
        summarize("rolling retune (B)", rolls)


if __name__ == "__main__":
    main()
