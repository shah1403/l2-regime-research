"""Synthetic NSE tick-to-tick L2 order-book generator.

Real historical NSE depth (tick-by-tick, 5-level snapshot feed) is a paid NSE
TBT / vendor product, so this generates statistically realistic stand-in data in
the exact .npz format the strategy consumes:

    bids, asks : (N, K, 2) = [price, size], K=5 levels (NSE snapshot depth)
    tr_price, tr_qty, tr_side (+1 aggressor buy / -1 sell), tr_bookidx
    symbol

Microstructure choices mirror NSE cash/F&O:
    * price grid = Rs 0.05 tick, spreads of 1-3 ticks for liquid stocks, wider
      and more volatile for a single-name option
    * hidden regime path (trending / mean-reverting / noise) with persistent
      signed order flow driving drift in trends -- so the strategy's regime
      detector and imbalance signal face the structure they were built for
    * top-of-book imbalance correlated with the flow state (the real
      microstructural mechanism the strategy exploits), plus noise
    * ~2-8% corrupted snapshots (crossed books, phantom spreads, empty touch)
      to exercise validate_book()

Swap these files for real vendor .npz exports and nothing downstream changes.
"""
from __future__ import annotations

import os
import numpy as np

K = 5
TICK = 0.05
SESSION_OPEN = 9 * 3600 + 15 * 60          # 09:15 IST
SESSION_CLOSE = 15 * 3600 + 30 * 60        # 15:30 IST


def _timestamps(N, rng, start_day="2026-06-29", mean_dt=2.0):
    """Event-driven snapshot times inside NSE sessions (epoch ms, IST wall clock)."""
    ts = np.empty(N, "int64")
    day = np.datetime64(start_day)
    sec = float(SESSION_OPEN)
    for i in range(N):
        sec += rng.exponential(mean_dt)
        if sec > SESSION_CLOSE:
            day = np.busday_offset(day, 1, roll="forward")
            sec = SESSION_OPEN + (sec - SESSION_CLOSE)
        ts[i] = day.astype("datetime64[ms]").astype("int64") + int(sec * 1000)
    return ts


def _regime_path(N, rng, dwell=(400, 1600), p=(0.32, 0.24, 0.44)):
    reg = np.zeros(N, int); t = 0
    while t < N:
        s = rng.choice([1, -1, 0], p=p)
        ln = int(rng.integers(*dwell))
        reg[t:t + ln] = s; t += ln
    return reg


def simulate(symbol, N=10_000, mid0=1400.0, sigma_bp=1.6, trend_bp=1.1,
             spread_ticks_mean=0.6, base_size=800, size_noise=0.55,
             imb_gain=1.1, invalid_rate=0.03, trade_rate=1.4, seed=0):
    rng = np.random.default_rng(seed)
    reg = _regime_path(N, rng)

    # latent signed-flow state + mid path
    flow = np.zeros(N); mid = np.zeros(N); mid[0] = mid0; anchor = mid0
    for t in range(1, N):
        sig = sigma_bp * 1e-4 * mid[t - 1]
        if reg[t] == 1:                                   # trending: persistent flow
            flow[t] = 0.97 * flow[t - 1] + 0.30 * rng.standard_normal()
            mid[t] = mid[t - 1] + trend_bp * 1e-4 * mid[t - 1] * np.tanh(flow[t]) \
                     + sig * rng.standard_normal()
        elif reg[t] == -1:                                # mean-reverting around anchor
            if reg[t - 1] != -1: anchor = mid[t - 1]
            flow[t] = 0.45 * flow[t - 1] - 0.25 * (mid[t - 1] - anchor) / max(sig, 1e-12) \
                      + 0.35 * rng.standard_normal()
            mid[t] = mid[t - 1] + 0.18 * (anchor - mid[t - 1]) + sig * rng.standard_normal()
        else:                                             # noise
            flow[t] = 0.80 * flow[t - 1] + 0.35 * rng.standard_normal()
            mid[t] = mid[t - 1] + sig * rng.standard_normal()

    # ---- book: touch on the Rs 0.05 grid, spread 1 + Poisson ticks ----
    sp_ticks = 1 + rng.poisson(spread_ticks_mean, N)
    half = sp_ticks * TICK / 2
    bb = np.floor((mid - half) / TICK) * TICK
    ba = np.ceil((mid + half) / TICK) * TICK
    ba = np.maximum(ba, bb + TICK)

    # top-of-book imbalance tied to the flow state (this is the exploitable edge)
    imb_lat = np.tanh(0.9 * flow) * np.where(reg == 1, 1.0, 0.35)
    bids = np.zeros((N, K, 2)); asks = np.zeros((N, K, 2))
    ln_b = size_noise * rng.standard_normal((N, K))
    ln_a = size_noise * rng.standard_normal((N, K))
    depth_gap_b = 1 + (rng.random((N, K)) < 0.12) * rng.integers(1, 3, (N, K))
    depth_gap_a = 1 + (rng.random((N, K)) < 0.12) * rng.integers(1, 3, (N, K))
    for k in range(K):
        off_b = np.where(k == 0, 0, np.cumsum(depth_gap_b[:, :k + 1], axis=1)[:, k] - depth_gap_b[:, 0])
        off_a = np.where(k == 0, 0, np.cumsum(depth_gap_a[:, :k + 1], axis=1)[:, k] - depth_gap_a[:, 0])
        bids[:, k, 0] = bb - off_b * TICK
        asks[:, k, 0] = ba + off_a * TICK
        tilt = imb_gain * imb_lat if k == 0 else 0.35 * imb_lat
        bids[:, k, 1] = np.round(base_size * np.exp(ln_b[:, k] + tilt) * (1 + 0.3 * k))
        asks[:, k, 1] = np.round(base_size * np.exp(ln_a[:, k] - tilt) * (1 + 0.3 * k))
    bids[:, :, 1] = np.maximum(bids[:, :, 1], 1)
    asks[:, :, 1] = np.maximum(asks[:, :, 1], 1)

    # ---- trades: Poisson prints, aggressor side follows flow ----
    tp, tq, tsd, tbi = [], [], [], []
    p_buy = 0.5 + 0.38 * np.tanh(flow)
    n_tr = rng.poisson(trade_rate, N)
    for t in range(N):
        for _ in range(n_tr[t]):
            side = 1 if rng.random() < p_buy[t] else -1
            px = ba[t] if side > 0 else bb[t]
            tp.append(px); tq.append(float(max(1, rng.poisson(base_size * 0.12))))
            tsd.append(side); tbi.append(t)
    if not tp:  # guarantee at least one print
        tp, tq, tsd, tbi = [ba[0]], [1.0], [1], [0]

    # ---- corrupt a slice of snapshots (crossed / phantom / empty touch) ----
    bad = rng.choice(np.arange(50, N), size=int(invalid_rate * N), replace=False)
    for t in bad:
        mode = rng.integers(3)
        if mode == 0:   # crossed book
            bids[t, 0, 0], asks[t, 0, 0] = asks[t, 0, 0] + TICK, bids[t, 0, 0] - TICK
        elif mode == 1: # phantom wide spread
            asks[t, 0, 0] = bids[t, 0, 0] + 400 * TICK * (1 + rng.random())
        else:           # empty touch
            bids[t, 0, 1] = 0.0

    return dict(bids=bids, asks=asks,
                tr_price=np.array(tp), tr_qty=np.array(tq),
                tr_side=np.array(tsd, int), tr_bookidx=np.array(tbi, int),
                ts=_timestamps(N, rng), symbol=symbol)


UNIVERSE = [
    # (symbol, kwargs) -- liquid NSE cash names + index/stock options
    ("RELIANCE",            dict(mid0=1420.0, sigma_bp=1.6, trend_bp=1.1, base_size=900,
                                 invalid_rate=0.025, seed=11)),
    ("HDFCBANK",            dict(mid0=1685.0, sigma_bp=1.4, trend_bp=1.0, base_size=1200,
                                 invalid_rate=0.03,  seed=23)),
    ("ICICIBANK",           dict(mid0=1245.0, sigma_bp=1.5, trend_bp=1.0, base_size=1100,
                                 invalid_rate=0.02,  seed=31)),
    ("INFY",                dict(mid0=1552.0, sigma_bp=1.8, trend_bp=1.2, base_size=700,
                                 invalid_rate=0.03,  seed=43)),
    ("TCS",                 dict(mid0=3462.0, sigma_bp=1.3, trend_bp=0.9, base_size=400,
                                 spread_ticks_mean=1.1, invalid_rate=0.025, seed=53)),
    ("SBIN",                dict(mid0=812.0,  sigma_bp=2.0, trend_bp=1.3, base_size=2000,
                                 invalid_rate=0.03,  seed=61)),
    ("TATAMOTORS",          dict(mid0=988.0,  sigma_bp=2.4, trend_bp=1.5, base_size=1500,
                                 invalid_rate=0.035, seed=71)),
    ("NIFTY25JUL25500CE",   dict(mid0=182.0,  sigma_bp=9.0, trend_bp=6.0, base_size=3000,
                                 spread_ticks_mean=2.2, size_noise=0.8,
                                 invalid_rate=0.08,  trade_rate=0.9, seed=37)),
    ("BANKNIFTY25JUL57500PE", dict(mid0=316.0, sigma_bp=11.0, trend_bp=7.0, base_size=2200,
                                 spread_ticks_mean=3.0, size_noise=0.9,
                                 invalid_rate=0.10,  trade_rate=0.8, seed=83)),
]


def _nse_outdir(sym):
    low = sym.lower()
    if low.startswith("banknifty"):
        return os.path.join("data", "options_data", "BANKNIFTY")
    if low.startswith("nifty"):
        return os.path.join("data", "options_data", "NIFTY")
    return os.path.join("data", "stock_data", "NSE_equities", sym.upper())


def main(N=10_000):
    paths = []
    for sym, kw in UNIVERSE:
        d = simulate(sym, N=N, **kw)
        outdir = _nse_outdir(sym)
        os.makedirs(outdir, exist_ok=True)
        p = os.path.join(outdir, f"orderbook_{sym.lower()}_{N}snaps.npz")
        np.savez_compressed(p, **d)
        paths.append(p)
        print(f"wrote {p}  ({N} snaps, {len(d['tr_price'])} trades)")
    return paths


if __name__ == "__main__":
    main()
