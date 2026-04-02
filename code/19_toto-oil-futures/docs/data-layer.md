# Data Layer Documentation

## Overview

All market data flows through a single SQLite database (`market/commodities.db`). 
The database is the single source of truth — every module (fine-tuning, forecasting, 
evaluation, autoresearch, the web app) reads from and writes to it. No loose CSVs 
or parquet files.

The database is **not checked into git** (it's ~32MB and regenerable). To recreate 
from scratch:

```bash
cd code/19_toto-oil-futures
python -m market.download   # fetch raw OHLCV → raw_ohlcv table
python -m market.prepare    # compute features + CV folds → features, cv_folds tables
```

---

## Module Layout

```
market/
├── db.py                    # Connection, schema, all query/insert helpers
├── download.py              # Orchestrator: config → source → DB
├── prepare.py               # Features, CV folds, Toto dataset builder
├── sources/
│   ├── base.py              # Abstract DataSource interface
│   └── yfinance_source.py   # Yahoo Finance implementation
├── features/
│   ├── base.py              # Abstract FeatureGenerator interface
│   └── registry.py          # Feature name → generator mapping
└── commodities.db           # SQLite database (gitignored)
```

## How Data Flows

```
config.yaml          →  download.py        →  raw_ohlcv table
(tickers, dates,         (calls source)         (immutable raw data)
 source name)                                        │
                                                     ▼
                         prepare.py         →  features table
                         (returns, log_ret,     (computed columns)
                          z-score, etc.)              │
                                                     ▼
                         prepare.py         →  cv_folds table
                         (purged k-fold           (time boundaries)
                          with embargo)               │
                                                     ▼
                         prepare.py         →  build_toto_dataset()
                         (reads features +      (list of series dicts,
                          fold dates)            ready for HF Dataset)
```

---

## SQLite Schema

### `raw_ohlcv` — Raw price data from any source

| Column | Type | Description |
|--------|------|-------------|
| ticker | TEXT | Symbol (e.g., `CL=F`, `GLD`) |
| date | TEXT | ISO 8601 date (`YYYY-MM-DD`) |
| open | REAL | Opening price |
| high | REAL | High price |
| low | REAL | Low price |
| close | REAL | Closing price |
| volume | REAL | Trading volume |
| source | TEXT | Data source name (`yfinance`, `fred`, etc.) |
| fetched_at | TEXT | When we downloaded it (UTC ISO timestamp) |

**Primary key:** `(ticker, date, source)`

This table is append-only and idempotent — re-running download skips existing rows.
The `source` column in the key means the same ticker+date from different sources 
coexist without conflict (useful for data quality cross-checks later).

### `features` — Computed base features

| Column | Type | Description |
|--------|------|-------------|
| ticker | TEXT | Symbol |
| date | TEXT | ISO 8601 date |
| close | REAL | Closing price (copied from raw) |
| returns | REAL | Daily percentage change: `(close - prev) / prev` |
| log_returns | REAL | Log return: `ln(close / prev_close)` |
| normalized | REAL | Rolling z-score: `(close - μ_60d) / σ_60d` |
| volume | REAL | Trading volume (copied from raw) |

**Primary key:** `(ticker, date)`

**Why returns, not prices?** Three reasons:
1. Removes futures contract rollover artifacts (the price jumps at roll dates 
   become noise in returns, not discontinuities)
2. Makes all series comparable in scale (oil at $70 and gold at $3000 both 
   have returns centered near 0)
3. Aligns with directional accuracy as our primary metric

**Why log returns too?** Log returns are additive over time and more symmetric 
than percentage returns. Toto's autoresearch may find one works better than the other.

**Why z-score normalized?** Rolling 60-day z-score removes local trend and scale, 
leaving the model to focus on the shape of recent price action. Another candidate 
target for autoresearch to test.

### `derived_features` — Extensible feature store (EAV pattern)

| Column | Type | Description |
|--------|------|-------------|
| ticker | TEXT | Symbol |
| date | TEXT | ISO 8601 date |
| feature_name | TEXT | Feature identifier (e.g., `volatility_20d`) |
| value | REAL | Computed value |
| computed_at | TEXT | When this was computed |

**Primary key:** `(ticker, date, feature_name)`

This table uses an Entity-Attribute-Value pattern so new features can be added 
without schema changes. See "Extending with New Features" below.

### `cv_folds` — Cross-validation fold boundaries

| Column | Type | Description |
|--------|------|-------------|
| split_id | TEXT | Hash of CV config (n_folds, embargo, purge, n_dates) |
| fold_number | INTEGER | 0, 1, 2, ... |
| split | TEXT | `train` or `test` |
| start_date | TEXT | First date in this split segment |
| end_date | TEXT | Last date in this split segment |
| n_days | INTEGER | Number of trading days |
| embargo_start | TEXT | Start of embargo zone (test folds only) |
| embargo_end | TEXT | End of embargo zone (test folds only) |

**Primary key:** `(split_id, fold_number, split)`

Folds are **time-based** — all tickers share the same temporal boundaries. 
This follows de Prado's rule: never split by asset, always by time, because 
cross-sectional correlation leaks market regime information.

### `experiments` — Fine-tuning run tracking

| Column | Type | Description |
|--------|------|-------------|
| experiment_id | TEXT | UUID or slug (primary key) |
| created_at | TEXT | When the run started |
| config_json | TEXT | Full config snapshot (JSON) |
| cv_fold | INTEGER | Which fold this ran on (NULL = all) |
| status | TEXT | `running`, `completed`, `failed` |
| checkpoint_path | TEXT | Path to saved model weights |
| train_loss | REAL | Final training loss |
| val_loss | REAL | Best validation loss |
| notes | TEXT | Autoresearch hypothesis or human notes |

Every experiment snapshots its full config. You can always reproduce a run or 
understand what produced a result.

### `forecasts` — Prediction outputs

| Column | Type | Description |
|--------|------|-------------|
| experiment_id | TEXT | Which model produced this forecast |
| ticker | TEXT | Symbol |
| forecast_date | TEXT | Date the forecast was made |
| target_date | TEXT | Date being predicted |
| horizon | INTEGER | Days ahead (1, 2, ..., prediction_length) |
| predicted | REAL | Point forecast (median of N samples) |
| predicted_q05 | REAL | 5th percentile (lower confidence bound) |
| predicted_q95 | REAL | 95th percentile (upper confidence bound) |
| actual | REAL | Ground truth (filled in when known) |

**Primary key:** `(experiment_id, ticker, forecast_date, target_date)`

The `actual` column starts NULL for forward predictions and gets backfilled 
as real prices come in. For backtesting, it's filled immediately.

### `evaluations` — Metrics per experiment per ticker per fold

| Column | Type | Description |
|--------|------|-------------|
| experiment_id | TEXT | Which experiment |
| ticker | TEXT | Symbol |
| cv_fold | INTEGER | Which fold's test set was evaluated |
| mae | REAL | Mean Absolute Error |
| rmse | REAL | Root Mean Squared Error |
| directional_acc | REAL | Fraction of days with correct direction |
| n_predictions | INTEGER | Number of predictions scored |

**Primary key:** `(experiment_id, ticker, cv_fold)`

---

## What We Have Right Now

**20 tickers, ~118,700 rows, 2000–2026:**

| Category | Tickers | History | Rows Each |
|----------|---------|---------|-----------|
| Energy futures | CL=F, BZ=F, NG=F, HO=F, RB=F | 2000–2026 | ~6,400 |
| Metals futures | GC=F, SI=F, HG=F | 2000–2026 | ~6,400 |
| Agriculture futures | ZC=F, ZW=F, ZS=F, KC=F, SB=F, CT=F | 2000–2026 | ~6,500 |
| Commodity ETFs | USO, UNG, GLD, SLV, DBA, DBC | 2004–2026 | ~4,800–5,400 |

**CV folds (purged 3-fold with 66-day embargo):**

| Fold | Test Period | Train Days | Test Days |
|------|-------------|------------|-----------|
| 0 | 2000-01-04 to 2008-09-22 | 4,346 | 2,205 |
| 1 | 2008-09-23 to 2017-06-26 | 4,341 | 2,205 |
| 2 | 2017-06-27 to 2026-04-02 | 4,405 | 2,207 |

Note: fold 1's train set excludes the purge zone + embargo around the test 
boundaries on both sides. This is the de Prado purged k-fold method.

**Train vs eval tickers:**
- **Train on all 20** — more data = better generalization for fine-tuning
- **Evaluate on 5**: CL=F (oil), NG=F (natgas), GC=F (gold), HG=F (copper), ZS=F (soybeans)
- The other 15 are "supporting cast" teaching the model commodity dynamics

---

## Pluggable Data Sources

Adding a new data source:

1. Create `market/sources/my_source.py`
2. Subclass `DataSource` from `base.py`
3. Implement `fetch(tickers, start_date, end_date) → DataFrame`
4. Register in `download.py`'s `SOURCES` dict
5. Set `data.source` in `config.yaml`

The DataFrame must have columns: `ticker, date, open, high, low, close, volume`
with `date` as ISO strings and numeric columns as floats.

Current sources:
- **yfinance** — free, no API key, 20+ commodities, daily back to 2000

Future sources (not yet implemented):
- **FRED** — energy + metals at daily, free API key
- **EIA** — best quality for energy, free API key
- **CSV import** — for Kaggle datasets or manual data
- **Kaggle** — mattiuzc dataset has 30+ commodities back to 1970s

---

## Extending with New Features

The feature system is designed for autoresearch to experiment with different 
inputs to Toto (exogenous variables). Architecture:

```python
# market/features/base.py — interface
class FeatureGenerator(ABC):
    name: str                                    # e.g., "volatility_20d"
    def compute(self, df: DataFrame) -> Series:  # single ticker's data → feature values

# market/features/registry.py — lookup table
REGISTRY: dict[str, FeatureGenerator] = {
    "volatility_20d": RollingVolatility(window=20),
    "fourier_dominant_freq": DominantFrequency(),
}
```

To add a new derived feature:

1. Create `market/features/my_feature.py` with a `FeatureGenerator` subclass
2. Register it in `registry.py`
3. Add the name to `config.yaml` under `data.exogenous_features`
4. `prepare.py` computes it and stores in `derived_features` table
5. The feature gets passed to Toto as an exogenous variable via `ev_fields`

**Feature ideas for autoresearch to explore:**

| Feature | What it captures | Implementation |
|---------|-----------------|----------------|
| `volatility_Nd` | Rolling std of returns (N=5,20,60) | `returns.rolling(N).std()` |
| `fourier_dominant_freq` | Dominant cycle length | FFT on rolling window |
| `fourier_spectral_energy` | Energy in frequency bands | Band-pass filter |
| `momentum_rsi` | Relative Strength Index | Standard RSI formula |
| `momentum_macd` | MACD signal line crossover | EMA(12) - EMA(26) |
| `cross_asset_corr` | Rolling correlation with another ticker | `returns.rolling(20).corr(other)` |
| `volume_ratio` | Volume relative to 20-day average | `volume / volume.rolling(20).mean()` |

None of these are built yet — just the interface is in place. The EAV table 
means no schema changes needed when adding features.

---

## Common Queries

```sql
-- Recent CL=F returns
SELECT date, close, returns FROM features 
WHERE ticker='CL=F' ORDER BY date DESC LIMIT 10;

-- All tickers and their date ranges
SELECT ticker, MIN(date), MAX(date), COUNT(*) 
FROM features GROUP BY ticker;

-- Get train data for fold 0
SELECT f.* FROM features f
JOIN cv_folds c ON f.date BETWEEN c.start_date AND c.end_date
WHERE c.split_id='pkf_3f_aa67d9d6003c' AND c.fold_number=0 AND c.split='train'
AND f.ticker='CL=F'
ORDER BY f.date;

-- Compare experiments on CL=F
SELECT e.experiment_id, e.notes, ev.directional_acc, ev.mae
FROM evaluations ev JOIN experiments e ON ev.experiment_id = e.experiment_id
WHERE ev.ticker='CL=F' ORDER BY ev.directional_acc DESC;
```

---

## CLI Commands

```bash
# Download all tickers from config
python -m market.download

# Download specific tickers only
python -m market.download --tickers CL=F GC=F

# Override start date
python -m market.download --start-date 2015-01-01

# Compute features + generate CV folds
python -m market.prepare

# Print summary of what's in the DB
python -m market.prepare --info
```
