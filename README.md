# L2 Regime Research

## Project visualization

![Project result preview](docs/preview.png)

[Open the interactive explorer](docs/explorer.html) — download and open this self-contained HTML in a browser. GitHub displays the source rather than running it. The explorer reads embedded audited results; full evidence and reproduction commands are below.


Aarush Shah's uploaded backtesting toolkit, preserved byte-for-byte, with independent trade-ledger reconciliation and explicit separation of synthetic and historical evidence.

## Reproduced scope

- 47 Python source files; parsed successfully and hashed in `verification/source_manifest.json`.
- Nine fixed-seed synthetic instruments, 10,000 snapshots each, at zero and 2.5 bps/side fees.
- GOOG, 10 July 2026: fixed one-second fast strategy, next-update and 250 ms entries. The supplied file has 985,948 updates; its vendor provenance is not independently established.
- BTCUSDT, 1 September 2023: original grid's first BTC day, downloaded from Binance's public archive and checked against the provider SHA-256. See the generated provenance and summary files for completion and exact scope.
- Every closed trade in these runs has an independently reconstructed ledger. Reconciliation establishes accounting consistency for these runs, not live profitability or correctness of every strategy.

**Material finding:** the previously recovered TATAMOTORS and INFY NPZ files exactly match this toolkit's synthetic generator, array for array. Their results are synthetic demonstrations with a planted imbalance/flow relationship, not historical NSE performance.

## Reproduce

```bash
python -m pip install -r requirements-verified.txt
python audit.py
# Optional: compare the recovered first-passage datasets
python audit.py --synthetic-reference ../first-passage-research/data/nse
python audit.py --replay ../first-passage-research/data/goog/goog_2026-07-10.npz --name GOOG_2026-07-10
python fetch_btc.py
python audit.py --replay data/binance/BTCUSDT_2023-09-01.npz --name BTCUSDT_2023-09-01
```

Run from any working directory. The original historical scripts retain their original path assumptions and are not presented as a polished production API. The audit entry points above resolve paths explicitly.

## Reading the results

Directional accuracy = fraction of closed trades with `side × (exit mid − entry mid) > 0`; unchanged midprices count as misses. Win rate = fraction of executable-touch trades with positive P&L after the chosen fees. These quantities differ because spreads, size, and fees matter.

Fast-engine P&L is **size-weighted basis points**, not USD, an account return, or an annualized Sharpe. The reported mean/standard-deviation ratio is per trade and unannualized. Dependence among trades makes the original naive t-statistic unsuitable as independent-trial significance evidence. Fee levels are explicit scenarios, not a claim about actual charges.

## Repository map

- `original_upload.zip`: unchanged uploaded archive.
- `source/`: unchanged uploaded Python files, retaining their folder hierarchy.
- `audit.py`: behavioral counterexamples, synthetic reruns, and independent fast/slow ledgers.
- `fetch_btc.py`: deterministic provider download, checksum verification, and conversion.
- `verification/`: source hashes, behavioral checks, provenance, summaries, and one row per closed trade.
- `data/synthetic/`: regenerated inputs with fixed seeds from the source.
- `data/binance/`: provider archive, checksum, and derived NPZ when downloaded.
- `docs/AUDIT.md`: defect inventory and limits of verification.
- `docs/explorer.html`: standalone, offline visualization with a fee slider and links to full ledgers.

No claims of >60% real-market directional accuracy or a ~0.5 internship per-trade Sharpe are certified by these files. No new license is imposed on the supplied source or third-party data.

## Large input files

Original inputs above GitHub’s single-file limit are included as numbered byte parts. Run `python restore_large_files.py` once to reconstruct them and verify their SHA-256. The manifest is `large_files.json`; no data is discarded.

## Fresh historical replay results

All rows use a fixed one-second hold. Net is after a **0.10 bps per side scenario**; units are weighted bps per trade, not USD.

| Dataset | Entry delay | Closed trades | Directional hit | Gross mean | Net mean | Net win rate |
|---|---:|---:|---:|---:|---:|---:|
| BTCUSDT_2023-09-01 | next | 12,143 | 81.96% | 0.5066 | 0.3200 | 61.06% |
| BTCUSDT_2023-09-01 | 250ms | 11,243 | 40.80% | 0.1033 | -0.0842 | 30.10% |
| GOOG_2026-07-10 | next | 75 | 58.67% | -0.3149 | -0.5157 | 13.33% |
| GOOG_2026-07-10 | 250ms | 73 | 46.58% | -0.8109 | -1.0118 | 9.59% |

At the original source’s 4.5 bps/side fee stress, all four rows have negative total and mean P&L. BTC’s next-update result is particularly sensitive to adding 250 ms latency. No grid-wide or live-performance claim follows from these two sessions.
