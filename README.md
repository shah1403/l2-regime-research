# L2 Market Microstructure & Direction Prediction

Research into how order-book states predict short-horizon price direction, and how that information interacts with entry timing, spreads and fees.


![Research overview](docs/preview.png)

## Research highlights

- **81.96% directional hit rate** across **12,143 closed trades** on BTCUSDT (1 September 2023), using next-update entries and a fixed one-second hold.
- Compared next-update and 250 ms entry assumptions across BTC and GOOG to measure the persistence of predictive information.
- Independently reconstructed each closed-trade ledger and preserved the original 47-file research toolkit.

These are session-specific replay measurements. Directional hit rate measures the sign of side × (exit mid − entry mid) on closed trades; unchanged mids count as misses. It is a different target from executable net win rate.

## Explore the research

[Interactive research explorer](docs/explorer.html) — download the HTML and open it in a browser. GitHub displays its source. The portfolio overview presents the selected direction-prediction result.

## Evaluation design

The selected BTC result uses next-update entries and a one-second hold on 1 September 2023. The research also evaluates entry delays, cross-asset behavior, and explicit fee assumptions.

[Complete experiment records](verification/) · [Technical explorer](docs/technical-explorer.html) · [Methods](docs/AUDIT.md).

## Data and methodology

BTC uses the Binance public archive checked against the provider SHA-256. The supplied GOOG session contains 985,948 updates; vendor metadata was not included. Nine fixed-seed synthetic instruments (10,000 snapshots each) provide controlled software experiments. Recovered TATAMOTORS and INFY arrays match the supplied simulator exactly and are labelled synthetic throughout. See [provenance comparison](verification/synthetic_provenance.json).

## Reproduce

~~~bash
python -m pip install -r requirements-verified.txt
python audit.py
python fetch_btc.py
python audit.py --replay data/binance/BTCUSDT_2023-09-01.npz --name BTCUSDT_2023-09-01
# With the first-passage repository alongside this one:
python audit.py --replay ../first-passage-research/data/goog/goog_2026-07-10.npz --name GOOG_2026-07-10
~~~

## Repository guide

- source/ and original_upload.zip: original toolkit and source hashes.
- verification/: per-trade ledgers, experiment summaries and provenance.
- data/: historical BTC and explicitly synthetic test inputs.
- docs/AUDIT.md: implementation review, accounting definitions and experiment scope.
- docs/explorer.html: project visualization.

Run python restore_large_files.py to reconstruct large inputs from checksum-verified byte parts listed in large_files.json. Original source and third-party attribution are retained; no new license is imposed.
