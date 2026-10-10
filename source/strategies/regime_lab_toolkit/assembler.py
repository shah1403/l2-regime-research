"""Two-stage assembler: Stage 1 selects/weights every tool in the toolkit at once
(not a single winner-take-all switch), Stage 2 combines the weighted signal
tools' outputs and applies weighted modifier tools on top.

Tool/modifier direction+size(+mult) arrays are precomputed ONCE, vectorized,
exactly as Phase 3 validated (tools.py's functions never depend on assembler
state). The genuinely STATEFUL part is Stage 1's weighting: each tool's
empirical-confidence term depends on its OWN trailing realized PnL, which only
exists once trades have actually closed -- so run_assembler is a walk-forward
loop that decides, fills, and updates trailing state one trade at a time,
mirroring backtest.run_backtest's fill convention (decide at t, fill at t+1,
one position at a time across the WHOLE assembled system).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest import FEE_BPS

MIN_N_DEFAULT = 8      # matches scripts/goog_minute_regime.py's MIN_TRAIN_TRADES
L_DEFAULT = 150         # matches adaptive_drift's trailing-edge selector window


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# ----------------------------- Stage 1 -----------------------------
def tool_applicability(category: str, regime_posterior_t: np.ndarray,
                       vol_axis_t: float, spread_axis_t: float) -> float:
    """regime_posterior_t: length-4 array [p_dir_up, p_dir_down, p_mr, p_flat].

    category="making" -> favors non-DIR regimes with spread wide relative to
    volatility (untested in the current minimal build -- no making-category
    tool exists yet, Phase 5). This rule IS regime-posterior-gated because
    this repo's evidence supports it (MM adverse selection is measurably worse
    in DIR-labeled periods).

    category="directional" -> 1.0 (NOT gated by P(DIR)). First implementation
    gated directional tools by P(DIR-up)+P(DIR-down), and phase4_assembler_e2e.py
    caught it immediately: flow_momentum lost 75% of its PnL (864->213bps) because
    this repo's own established finding is that flow_momentum's edge concentrates
    in MR/FLAT-labeled periods, not DIR (94% of its winning trades fire outside
    DIR -- see regime_lab's module docstring / the Phase 2 background). Gating a
    "directional" tool by the price-shape DIR posterior reproduces the exact
    starvation this whole architecture exists to avoid. So: applicability for
    directional tools is left to the tool's OWN signal (its `confidence` field)
    and Stage 1's empirical trailing-edge term, not the regime layer -- the
    regime posterior only gates EXECUTION STYLE (making vs not), not direction.
    """
    p_dir = regime_posterior_t[0] + regime_posterior_t[1]
    if category == "directional":
        return 1.0
    elif category == "making":
        return float(np.clip((1.0 - p_dir) * _sigmoid(spread_axis_t - vol_axis_t), 0.0, 1.0))
    raise ValueError(f"unknown category {category!r}")


def trailing_empirical_edge(trailing_trade_pnls: np.ndarray, min_n: int = MIN_N_DEFAULT) -> float:
    """Cold start: fewer than `min_n` trailing trades -> neutral (1.0, statistical
    confidence only). Once there's enough history: sigmoid(t-stat of trailing
    per-trade PnL) -- decays toward 0 for a tool with a track record of losing,
    stays near 1 for a tool that's been winning, ~0.5 for a coin-flip track record.
    """
    n = len(trailing_trade_pnls)
    if n < min_n:
        return 1.0
    x = np.asarray(trailing_trade_pnls, dtype=float)
    sh = x.mean() / x.std() if x.std() > 0 else 0.0
    tstat = sh * np.sqrt(n)
    return float(_sigmoid(tstat))


def select_and_weight(tools: dict, applicability_inputs: dict, trailing_state: dict,
                      min_n: int = MIN_N_DEFAULT, L: int = L_DEFAULT) -> dict:
    """`applicability_inputs[name] = (category, regime_posterior_t, vol_t, spread_t)`.
    `trailing_state[name]` = list of that tool's trailing realized per-trade PnLs
    (causal -- only trades closed strictly before the current bar).
    Returns dict[name -> weight in [0, inf)] (applicability in [0,1] times a
    sigmoid-bounded edge term also in [0,1], so weights are in [0,1] here).
    """
    weights = {}
    for name in tools:
        category, post_t, vol_t, spread_t = applicability_inputs[name]
        appl = tool_applicability(category, post_t, vol_t, spread_t)
        trail = np.array(trailing_state.get(name, [])[-L:])
        edge = trailing_empirical_edge(trail, min_n)
        weights[name] = appl * edge
    return weights


# ----------------------------- Stage 2 -----------------------------
def combine_signals(tool_outputs: dict, weights: dict, mode: str = "soft",
                    min_confirm: int = 1):
    """tool_outputs[name] = (direction, size) for ONE bar (scalars). Returns
    (final_direction, final_size) for that bar.

    mode="soft": weighted blend sum(w*dir*size)/sum(w), sign of the blend is
    final direction, magnitude is the weighted-average size.
    mode="hard": require >=min_confirm active (nonzero weight & direction)
    tools agreeing in sign, else abstain (0,0) -- mirrors
    scripts/l2_strategy.py:158's use_drift_confirm=True pattern.

    With one signal tool (the current minimal toolkit), both modes reduce to a
    pass-through -- exercised here via a synthetic 2-tool stub in
    phase4_assembler_e2e.py so the interface doesn't need a retrofit once a
    second signal tool exists (soft-vs-hard comparison is deferred to Phase 5).
    """
    active = {n: (d, s) for n, (d, s) in tool_outputs.items()
             if weights.get(n, 0.0) > 0 and d != 0 and s > 0}
    if not active:
        return 0, 0.0

    if mode == "soft":
        num = sum(weights[n] * d * s for n, (d, s) in active.items())
        den = sum(weights[n] for n in active)
        if den <= 0:
            return 0, 0.0
        blend = num / den
        return (1 if blend > 0 else -1 if blend < 0 else 0), abs(blend)
    elif mode == "hard":
        pos = [n for n, (d, s) in active.items() if d > 0]
        neg = [n for n, (d, s) in active.items() if d < 0]
        if len(pos) >= min_confirm and len(pos) >= len(neg):
            sizes = [active[n][1] for n in pos]
            return 1, float(np.mean(sizes))
        if len(neg) >= min_confirm and len(neg) > len(pos):
            sizes = [active[n][1] for n in neg]
            return -1, float(np.mean(sizes))
        return 0, 0.0
    raise ValueError(f"unknown mode {mode!r}")


def apply_modifiers(size: float, modifier_outputs: dict, weights: dict) -> float:
    """modifier_outputs[name] = mult (scalar, for this bar). Each modifier's
    effect is interpolated by its own Stage-1 weight: effective_mult =
    1 + w*(mult-1) -- no effect at w=0, full effect at w=1. Effects multiply.
    """
    out = size
    for name, mult in modifier_outputs.items():
        w = weights.get(name, 0.0)
        out *= (1.0 + w * (mult - 1.0))
    return out


# ----------------------------- run_assembler -----------------------------
def run_assembler(bars: dict, tools: dict, modifiers: dict, ctx: dict,
                  regime_out: dict, wall: int, mode: str = "soft",
                  min_confirm: int = 1, fee_bps: float = FEE_BPS,
                  min_n: int = MIN_N_DEFAULT, L: int = L_DEFAULT) -> dict:
    """Precomputes every tool's/modifier's (N,) arrays once (Phase 3 style), then
    walks forward bar by bar computing Stage-1 weights from causal trailing
    state, Stage-2 combining, filling one position at a time, and updating
    trailing state once each trade closes. Test window only: [wall, N).
    """
    mid, bb, ba = bars["mid"], bars["bb"], bars["ba"]
    N = len(mid)
    posterior, vol_axis, spread_axis = regime_out["posterior"], ctx["vol_z"], ctx["spread_z"]

    tool_arr = {name: spec["fn"](ctx, wall=wall) for name, spec in tools.items()}
    mod_arr = {}
    for name, spec in modifiers.items():
        # modifiers act on the (arbitrary, first) signal tool's raw direction --
        # fine for the 1-signal-tool minimal build; multi-signal-tool modifier
        # wiring is a Phase 5 concern.
        any_tool = next(iter(tool_arr.values()))
        mod_arr[name] = spec["fn"](ctx, any_tool["direction"], wall=wall)

    trailing_state = {name: [] for name in list(tools) + list(modifiers)}
    weight_trace = {name: np.full(N, np.nan) for name in list(tools) + list(modifiers)}

    per_trade = []
    equity = np.full(N, np.nan)
    cum = 0.0
    t = wall
    stop = N - 2
    while t < stop:
        appl_inputs = {name: (spec["category"], posterior[t], vol_axis[t], spread_axis[t])
                      for name, spec in tools.items()}
        appl_inputs.update({name: (spec["category"], posterior[t], vol_axis[t], spread_axis[t])
                           for name, spec in modifiers.items()})
        weights = select_and_weight(list(tools) + list(modifiers), appl_inputs,
                                    trailing_state, min_n, L)
        for name in weights:
            weight_trace[name][t] = weights[name]

        tool_outputs_t = {name: (int(arr["direction"][t]), float(arr["raw_size"][t]))
                          for name, arr in tool_arr.items()}
        direction, size = combine_signals(tool_outputs_t, weights, mode, min_confirm)

        if direction != 0 and size > 0:
            mod_outputs_t = {name: float(arr[t]) for name, arr in mod_arr.items()}
            final_size = apply_modifiers(size, mod_outputs_t, weights)
            if final_size <= 0:
                equity[t] = cum
                t += 1
                continue

            equity[t] = cum
            side = 1.0 if direction > 0 else -1.0
            px_in = ba[t + 1] if side > 0 else bb[t + 1]
            hold = tool_arr[next(iter(tools))]["hold"]   # single-signal-tool hold;
            # Phase 5: per-tool holds need attribution once >1 signal tool exists
            e = min(t + 1 + hold, N - 1)
            px_out = bb[e] if side > 0 else ba[e]
            pnl_bps = side * (px_out - px_in) / mid[t] * 1e4 * final_size \
                     - 2 * fee_bps * final_size
            per_trade.append(pnl_bps)
            cum += pnl_bps
            equity[t + 1:e + 1] = cum

            # counterfactual (no modifiers) for modifier-specific attribution
            counterfactual = side * (px_out - px_in) / mid[t] * 1e4 * size \
                             - 2 * fee_bps * size
            for name in tools:
                trailing_state[name].append(pnl_bps)
            for name in modifiers:
                trailing_state[name].append(pnl_bps - counterfactual)
            t = e + 1
        else:
            equity[t] = cum
            t += 1
    equity[t:N] = cum

    per_trade = np.array(per_trade, dtype=float)
    if len(per_trade) == 0:
        result = dict(pnl=0.0, trades=0, per_trade_pnls=per_trade, win_rate=0.0,
                      tstat=0.0, equity=equity)
    else:
        sh = per_trade.mean() / per_trade.std() if per_trade.std() > 0 else 0.0
        result = dict(pnl=float(per_trade.sum()), trades=len(per_trade),
                      per_trade_pnls=per_trade, win_rate=float(100 * np.mean(per_trade > 0)),
                      tstat=float(sh * np.sqrt(len(per_trade))), equity=equity)
    result["weight_trace"] = weight_trace
    return result
