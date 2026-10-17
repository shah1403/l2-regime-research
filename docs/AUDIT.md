# Implementation review and experiment methodology

## Scope and fidelity

Input: `l2_crypto_backtests.zip`, SHA-256 `a4b554719f3617f8186f505895af5d36989c4a67bc2ce8ff9e840c5268474c6e`.
The archive contains 47 Python files, no input datasets and no saved trade ledgers. `source/` is unchanged. A SHA-256 manifest and AST parse checks accompany the copy. Replays use original functions rather than silently repairing them.

## Behavioral fixtures

`verification/behavior_checks.json` records executable fixtures in `audit.py`:

1. Binance trade bucketing assigns a 150 ms trade to the 100 ms snapshot with `searchsorted(..., side='right') - 1`. The slow pipeline consumes the whole bucket at that snapshot, and the converter discards the original trade timestamp. This can reveal later trades too early.
2. `validate_book` uses the full-file median spread. Appending later wide spreads changes an earlier invalid snapshot to valid. Validation is not prefix-invariant.
3. `backtest` replaces an empty closed-trade array with `[0.0]`, then reports one trade. The audit counts actual exits.
4. Terminal P&L includes marked open inventory; closed per-trade P&L alone may not reconcile. A short fixture demonstrates the difference.
5. Execution can fill approximately 1.696 units against only 0.01 displayed units. Depth limits, partial execution and impact are absent.
6. Fixed-cost drift output passed a deterministic prefix comparison. This does not repair the full-file spread inputs or prove every configuration causal.

These fixtures document specific boundary behavior and accounting conventions; each assertion is scoped to the fixture it exercises.

## Additional source-inspection findings

- `regime_lab_toolkit/backtest.py` and `assembler.py` credit a future completed trade's profit across the interval starting immediately after entry. That is not a causal equity curve. The new explorer accumulates closed P&L at exits.
- The bar backtest's early exit both observes and executes on the same bar; executable timing requires a stated ordering or next-bar fill.
- Minute/adaptive episode accounting snapshots equity after entry charges, excluding entry costs from episode returns. Terminal flattening fees need separate reconciliation.
- `tick_momentum_strategy.py` filters observations using the magnitude of the future move. This is a diagnostic conditioned on the outcome, not an executable selection rule.
- `regime.py` initializes neutral outputs to column 2 (MR), although FLAT is column 3.
- The assembler credits constituent tools with the combined trade outcome; this is not independent attribution. Hold/modifier wiring requires separate validation.
- `l2_mmdir.py` counts position changes as directional trades, and uses next-observation fills in an earlier equity row. Fee notional uses a full-sample reference price.
- Hysteresis counts consecutive disagreements, not necessarily consecutive votes for the same candidate regime.
- HMM state diagnostics concatenate separated observations when calculating within-state autocorrelation. Such adjacency is artificial; labels remain heuristics.
- The flow z-score implementation includes the current bar despite a past-only description. Use after bar close is different from using it before close.
- Horizon diagnostics use full-day quantile gates and overlapping observations. They are exploratory, not an independent out-of-sample performance test.
- Float32 price storage, millisecond truncation, timestamp monotonicization, mixed-symbol handling, and positional file pairing can alter ordering or precision.
- Same-bucket maker fills lack proven within-bucket event ordering; queue position, partial fills and impact are not verified.

These inspected issues were not all exhaustively exercised across every parameterization. They limit any blanket certification of the 47-file toolkit.

## Provenance and interpretation

Nine synthetic runs reproduce the supplied fixed seeds and retain injected corrupt books. The signal relationship is deliberately planted. Exact matches for recovered INFY and TATAMOTORS arrays establish their synthetic origin. Per-trade cash-flow reconciliation at zero and 2.5 bps/side fees is in the CSVs.

GOOG fast replay uses one recovered session. It is distinct from the first-passage engine in the other repository. BTC uses one publicly downloadable day from the original fixed experiment grid; the remaining original multi-asset/multi-day grid is not reproduced here. Neither is a live execution test.


