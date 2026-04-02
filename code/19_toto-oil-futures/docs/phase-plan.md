# Phase Plan — Commodities Forecasting MVP

Date: 2026-04-02

---

## Architecture: Extensibility Points

The system is designed so autoresearch (or a human) can swap any of these without touching other modules:

```
┌─────────────────────────────────────────────────────────────────┐
│                        config.yaml                              │
│  (every tunable knob lives here — autoresearch mutates this)    │
└─────────┬───────────────────────────────────────────────────────┘
          │
          ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│ Data Sources │     │   Prepare    │     │   SQLite DB      │
│ (pluggable)  │────▶│  (features)  │────▶│  (single source  │
│              │     │              │     │   of truth)       │
│ • yfinance   │     │ • target col │     │                  │
│ • FRED       │     │ • returns    │     │ tables:          │
│ • CSV import │     │ • log_returns│     │ • raw_ohlcv      │
│ • Kaggle     │     │ • normalize  │     │ • features       │
│ • EIA        │     │ • exog vars  │     │ • splits         │
└──────────────┘     └──────────────┘     │ • experiments    │
                                          │ • forecasts      │
                                          │ • evaluations    │
                                          └────────┬─────────┘
                                                   │
                          ┌────────────────────────┼────────────────┐
                          ▼                        ▼                ▼
                   ┌─────────────┐        ┌──────────────┐  ┌────────────┐
                   │  Fine-tune  │        │   Forecast   │  │  Evaluate  │
                   │  (Toto)     │        │  (inference)  │  │  (metrics) │
                   └─────────────┘        └──────────────┘  └────────────┘
                          │                        │                │
                          └────────────────────────┴────────────────┘
                                          │
                                   ┌──────▼──────┐
                                   │ Autoresearch │  (Phase 2)
                                   │ reads results│
                                   │ mutates config
                                   │ re-runs      │
                                   └─────────────┘
```

### What autoresearch can tune (by changing config.yaml):
- **Target variable**: close, returns, log_returns, normalized_close
- **Prediction horizon**: 1, 5, 10, 20 days
- **Context factor**: 4, 8, 16 (context_length = patch_size × factor)
- **Ticker universe**: which tickers to train on, which to evaluate on
- **Data source**: yfinance, FRED, CSV, Kaggle — or combine multiple
- **Exogenous variables**: volume, other tickers, VIX, macro indicators
- **Hyperparameters**: lr, warmup_steps, stable_steps, decay_steps, batch_size
- **Feature engineering**: raw vs differenced vs log-differenced vs z-scored

### Cross-Validation Strategy (de Prado)

Based on Marcos López de Prado's *Advances in Financial Machine Learning* (Chapter 7):

**Why not standard k-fold:** Financial time series are autocorrelated and labels overlap in time.
Standard k-fold shuffles data, interleaving train/test in time → catastrophic leakage.

**Our approach: Purged K-Fold CV with Embargo**

```
Timeline:  2000 ──────────────────────────────────────── 2026

3-fold purged CV (split by TIME, all tickers in same fold):

Fold 1:  [=====TEST=====][embargo][===========TRAIN===========]
Fold 2:  [====TRAIN====][embargo][===TEST===][embargo][=TRAIN=]
Fold 3:  [===========TRAIN===========][embargo][=====TEST=====]

embargo = ~1-2% of data ≈ 50 trading days ≈ 2.5 months
purge  = remove any training sample whose 5-day label window
         overlaps the test fold boundary
```

**Key rules (de Prado):**
1. **Split by TIME, not by asset.** All tickers in a given time period → same fold.
   Cross-sectional correlation means splitting by asset leaks market regime.
2. **Purge** training samples whose label window overlaps test boundaries.
   With 5-day prediction horizon, purge 5 days around each boundary.
3. **Embargo** ~50 trading days after each test boundary to handle feature
   autocorrelation (1-2% of ~5,000 total days).
4. **3 folds for MVP** (not 5 — we don't have enough data for 5 meaningful folds).
5. **CPCV (N=6, p=2 → 15 combinations)** is the Phase 2 / autoresearch upgrade.

**Ticker split strategy:**
- Train folds: ALL tickers (futures + ETFs) — maximum data for fine-tuning
- Evaluation tickers: Energy futures only (CL=F, NG=F, HO=F, RB=F, BZ=F)
  — these are what we actually want to forecast
- The other tickers (metals, agriculture, ETFs) are "supporting cast" that
  help the model learn general commodity dynamics

**Walk-forward as final check only:**
De Prado warns against using walk-forward for model selection (single path,
non-stationary training size, recency bias). Use it only as the last sanity
check after selecting the model via purged CV.

### Derived Features — Extensibility Design

The `features` table stores base features. Derived/computed features live in
a separate `derived_features` table with a registry pattern so autoresearch
can request any combination as Toto exogenous variables.

**MVP features (Phase 1):**
- close, returns, log_returns, normalized_close, volume

**Future features (autoresearch can request via config):**
- Volatility: rolling std of returns (5d, 20d, 60d windows)
- Fourier: dominant frequency, spectral energy in bands
- Momentum: RSI, MACD, rate of change
- Cross-asset: correlation with other tickers, spread ratios
- Macro: VIX (if added as data source), yield curve slope

**How it works:**
```python
# market/features/base.py
class FeatureGenerator:
    """Abstract: compute(ticker, dates) → DataFrame with date + feature columns"""

# market/features/volatility.py
class VolatilityFeatures(FeatureGenerator):
    windows = [5, 20, 60]
    def compute(self, ticker, dates): ...

# market/features/fourier.py
class FourierFeatures(FeatureGenerator):
    def compute(self, ticker, dates): ...

# market/features/registry.py
REGISTRY = {
    "volatility_5d": VolatilityFeatures(window=5),
    "volatility_20d": VolatilityFeatures(window=20),
    "fourier_dominant_freq": FourierFeatures(),
    ...
}
```

Config says which features to use as exogenous variables:
```yaml
data:
  target: "returns"
  exogenous_features:
    - volatility_20d
    - fourier_dominant_freq
```

Prepare step reads config → calls registry → computes features → stores in DB →
passes to Toto as `ev_fields`. **We don't build any of this for MVP** — we just
design the interface so it slots in cleanly later.

### SQLite Schema (single file: `market/commodities.db`)

```sql
-- Raw data from any source
CREATE TABLE raw_ohlcv (
    ticker      TEXT NOT NULL,
    date        TEXT NOT NULL,    -- ISO 8601
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    volume      REAL,
    source      TEXT NOT NULL,    -- 'yfinance', 'fred', 'csv', etc.
    fetched_at  TEXT NOT NULL,    -- when we downloaded it
    PRIMARY KEY (ticker, date, source)
);

-- Base computed features (one row per ticker per date)
CREATE TABLE features (
    ticker          TEXT NOT NULL,
    date            TEXT NOT NULL,
    close           REAL,
    returns         REAL,         -- daily pct change
    log_returns     REAL,         -- log(close/prev_close)
    normalized      REAL,         -- z-scored within rolling window
    volume          REAL,
    PRIMARY KEY (ticker, date)
);

-- Extensible derived features (EAV pattern for flexibility)
-- Autoresearch can add new feature types without schema changes
CREATE TABLE derived_features (
    ticker          TEXT NOT NULL,
    date            TEXT NOT NULL,
    feature_name    TEXT NOT NULL, -- e.g. 'volatility_20d', 'fourier_dominant_freq'
    value           REAL,
    computed_at     TEXT NOT NULL,
    PRIMARY KEY (ticker, date, feature_name)
);

-- CV fold definitions (purged k-fold, per split config)
-- All tickers share the same time boundaries within a split_id
CREATE TABLE cv_folds (
    split_id        TEXT NOT NULL, -- hash of CV config (n_folds, embargo, horizon)
    fold_number     INTEGER NOT NULL,
    split           TEXT NOT NULL, -- 'train', 'test'
    start_date      TEXT NOT NULL,
    end_date        TEXT NOT NULL,
    n_days          INTEGER,
    PRIMARY KEY (split_id, fold_number, split)
);
-- Note: no ticker column — folds are time-based, all tickers share boundaries
-- Purge/embargo boundaries stored as metadata in split config

-- Experiment tracking (autoresearch writes here)
CREATE TABLE experiments (
    experiment_id   TEXT PRIMARY KEY,  -- uuid or slug
    created_at      TEXT NOT NULL,
    config_json     TEXT NOT NULL,     -- full config snapshot (includes CV fold used)
    cv_fold         INTEGER,          -- which fold was used for this run (NULL = all)
    status          TEXT NOT NULL,     -- 'running', 'completed', 'failed'
    checkpoint_path TEXT,
    train_loss      REAL,
    val_loss        REAL,
    notes           TEXT               -- autoresearch hypothesis
);

-- Forecast outputs
CREATE TABLE forecasts (
    experiment_id   TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    forecast_date   TEXT NOT NULL,     -- date the forecast was made
    target_date     TEXT NOT NULL,     -- date being predicted
    horizon         INTEGER NOT NULL,  -- days ahead
    predicted       REAL,
    predicted_q05   REAL,              -- 5th percentile
    predicted_q95   REAL,              -- 95th percentile
    actual          REAL,              -- filled in when known
    PRIMARY KEY (experiment_id, ticker, forecast_date, target_date)
);

-- Evaluation metrics (one row per experiment per ticker per fold)
CREATE TABLE evaluations (
    experiment_id       TEXT NOT NULL,
    ticker              TEXT NOT NULL,
    cv_fold             INTEGER NOT NULL,
    mae                 REAL,
    rmse                REAL,
    directional_acc     REAL,
    n_predictions       INTEGER,
    PRIMARY KEY (experiment_id, ticker, cv_fold)
);
```

### Project Structure (updated)

```
19_toto-oil-futures/
├── config.yaml                # All settings — autoresearch mutates this
├── requirements.txt
├── README.md
│
├── market/
│   ├── db.py                  # SQLite connection + schema init + query helpers
│   ├── sources/
│   │   ├── base.py            # Abstract DataSource interface
│   │   ├── yfinance_source.py # yfinance implementation
│   │   └── (fred_source.py)   # future: FRED, EIA, Kaggle, etc.
│   ├── features/
│   │   ├── base.py            # Abstract FeatureGenerator interface
│   │   ├── registry.py        # Feature name → generator mapping
│   │   └── (volatility.py)    # future: rolling vol, ATR, etc.
│   │   └── (fourier.py)       # future: spectral features
│   │   └── (momentum.py)      # future: RSI, MACD, etc.
│   ├── download.py            # Orchestrator: reads config, calls sources, writes to DB
│   └── prepare.py             # Computes features, generates CV folds, builds Toto dataset
│
├── model/
│   ├── finetune.py            # Fine-tune Toto, log experiment to DB
│   └── forecast.py            # Run inference, write forecasts to DB
│
├── eval/
│   └── evaluate.py            # Read forecasts from DB, compute metrics, write to DB
│
├── run_remote.py              # RunPod orchestration: create pod, upload, train, download, terminate
│
├── notebooks/
│   └── mvp_demo.ipynb         # End-to-end demo (screen-record for video)
│
├── autoresearch/              # Phase 5 scaffold
│   ├── program.md             # Agent instructions
│   ├── train.py               # Single-file pipeline the agent mutates
│   └── (results come from DB)
│
└── local-docs/                # Planning docs (you're reading one)
```

---

## Phase 1: Data Foundation ✅ COMPLETE
**Goal:** Download commodities data, store in SQLite, compute features, generate CV folds.
**Milestone:** Can query `SELECT * FROM features WHERE ticker='CL=F'` and get clean daily returns. CV folds exist in `cv_folds` table.

| Step | Deliverable | What |
|------|-------------|------|
| 1.1 | `market/db.py` | SQLite connection manager, schema creation, query helpers. All DB access goes through this module. |
| 1.2 | `market/sources/base.py` | Abstract `DataSource` class: `fetch(tickers, start, end) → DataFrame`. Simple contract. |
| 1.3 | `market/sources/yfinance_source.py` | First implementation. Downloads OHLCV, returns standardized DataFrame. |
| 1.4 | `market/download.py` | Reads `config.yaml` tickers + date range, calls source, writes to `raw_ohlcv` table. Idempotent (skip dates already fetched). |
| 1.5 | `market/prepare.py` | Reads from `raw_ohlcv`, computes base features (returns, log_returns, normalized), writes to `features` table. Generates purged k-fold CV boundaries with embargo, writes to `cv_folds` table. Builds Toto `custom_dataset` dict from DB data for a given fold. |
| 1.6 | `market/features/` (scaffold only) | Create `base.py` (abstract FeatureGenerator), `registry.py` (feature registry dict). No actual derived features yet — just the interface so autoresearch can plug them in later. |
| 1.7 | `config.yaml` + `requirements.txt` | Initial config with all defaults including CV params. Pin dependency versions. |

**Verify:** ✅ Run download → prepare → query DB → see clean data for all tickers. `cv_folds` table has 3 folds with correct purge/embargo boundaries. 118,669 rows, 20 tickers, 2000–2026.

---

## Phase 2: Zero-Shot Baseline + RunPod Setup — CODE COMPLETE, NEEDS RUN
**Goal:** Run Toto without fine-tuning to establish the floor. Get RunPod working.
**Milestone:** Directional accuracy number for zero-shot Toto on CL=F test set, stored in `evaluations` table.

**Compute: RunPod** — L4 GPU pod (~$0.40/hr), PyTorch template, ~$0.50 per session.
See [runpod-reference.md](runpod-reference.md) for full details.

| Step | Deliverable | Status | What |
|------|-------------|--------|------|
| 2.1 | `model/forecast.py` | ✅ Done | Load base Toto model, read test data from DB, run `TotoForecaster`, write predictions to `forecasts` table. |
| 2.2 | `eval/evaluate.py` | ✅ Done | Read forecasts + actuals from DB, compute MAE/RMSE/directional accuracy, write to `evaluations` table. Tested with mock data. |
| 2.3 | `run_remote.py` | ✅ Done | RunPod orchestration: create pod, upload code + DB via SSH, run commands, download results, terminate pod. API key loads from `keys.env`. |
| 2.4 | `model/finetune.py` | ✅ Done | Fine-tune Toto using their pipeline, log experiment to DB. (Built ahead of schedule for Phase 3.) |
| 2.5 | RunPod account setup | ✅ Done | API key in `keys.env`, SSH key added, `runpod` SDK installed, connection tested. |
| 2.6 | Run zero-shot baseline | ⬜ TODO | `python run_remote.py forecast --fold 2` — first real GPU run. |

**Verify:** `evaluations` table has zero-shot directional accuracy for CL=F. RunPod workflow is repeatable in one command.

---

## Phase 3: Fine-Tuning
**Goal:** Fine-tune Toto on commodities data, prove it beats zero-shot.
**Milestone:** Fine-tuned model with higher directional accuracy than zero-shot, experiment logged in `experiments` table.

| Step | Deliverable | What |
|------|-------------|------|
| 3.1 | `model/finetune.py` | Read prepared data from `prepare.py` (which reads from DB). Run Toto's fine-tuning pipeline. Log experiment to `experiments` table (config snapshot, losses, checkpoint path). |
| 3.2 | First fine-tune run | `run_remote.py` → RunPod: train on all tickers, evaluate on val set. Compare to zero-shot. ~30 min on L4. |
| 3.3 | Test set evaluation | Run best checkpoint on held-out test set. Final honest number. |

**Verify:** `experiments` table has at least one completed run. `evaluations` table shows fine-tuned vs zero-shot comparison.

---

## Phase 4: Demo Notebook
**Goal:** End-to-end pipeline in one visual notebook. This is the video content.
**Milestone:** Notebook runs locally with pre-computed results, produces charts.

| Step | Deliverable | What |
|------|-------------|------|
| 4.1 | `notebooks/mvp_demo.ipynb` | Cell-by-cell: show data → show CV folds → zero-shot results → fine-tune results → compare → plot. Reads from DB (results already computed on RunPod). |
| 4.2 | Visualizations | Actual vs predicted charts, confidence bands, directional accuracy bar chart (zero-shot vs fine-tuned), training loss curve. |
| 4.3 | Screen-record ready | Notebook tells the story. No GPU needed to run it — just reads results from DB. |

**Verify:** Notebook runs locally, produces compelling visuals.

---

## Phase 5: Autoresearch Scaffold
**Goal:** Structure so an AI agent can iterate experiments autonomously on RunPod.
**Milestone:** `autoresearch/train.py` runs one experiment end-to-end on RunPod and logs results to DB.

| Step | Deliverable | What |
|------|-------------|------|
| 5.1 | `autoresearch/train.py` | Single script: load config → prepare data → fine-tune → forecast → evaluate → print single metric. Runs on the RunPod pod. |
| 5.2 | `autoresearch/program.md` | Instructions for the agent: what to explore, constraints, how to read results, what "better" means. |
| 5.3 | Config-driven experiments | `train.py` accepts a `--config` override (JSON string or file) so the agent can change any parameter without editing code. |
| 5.4 | Agent loop via `run_remote.py` | Agent generates config → calls `run_remote.py` → reads metrics from returned DB → decides next experiment. Pod stays alive between runs to avoid cold starts. |

**Verify:** Agent can run N experiments on a single RunPod pod session, each with different configs, and all results land in the DB.

---

## Phase 6: Forward Testing Web App (stretch)
**Goal:** Daily predictions behind a simple UI for live forward testing.
**Milestone:** Streamlit app showing tomorrow's forecast for CL=F with confidence bands.

| Step | Deliverable | What |
|------|-------------|------|
| 6.1 | `app.py` | Streamlit app. Reads latest data from DB, loads best checkpoint, runs forecast (CPU inference is fine for one prediction), displays prediction + history + accuracy log. |
| 6.2 | Daily update script | Cron/manual script that downloads today's data, appends to DB, runs forecast, stores result. |

**Verify:** Open browser, see today's prediction for oil, see historical accuracy.

---

## Compute Strategy

**RunPod** for all GPU work. See [runpod-reference.md](runpod-reference.md).

| Phase | What runs on RunPod | Estimated cost |
|-------|--------------------|----|
| Phase 2 | Zero-shot inference (all eval tickers) | ~$0.25 (30 min L4) |
| Phase 3 | Fine-tuning + evaluation (1 run) | ~$0.25 (30 min L4) |
| Phase 5 | Autoresearch (N experiments) | ~$2-5 (2-5 hrs L4) |

**Local** for data download, preparation, evaluation analysis, notebook, web app.
Toto inference on CPU is slow but possible for single predictions (Phase 6).

**`run_remote.py` is the bridge:**
```
Local: prepare DB → run_remote.py → RunPod: train/forecast → run_remote.py → Local: evaluate/visualize
```

---

## Critical Path

```
Phase 1 (Data) ✅  ──▶  Phase 2 (Zero-shot + RunPod) [code ✅, run ⬜]  ──▶  Phase 3 (Fine-tune)
                                                                    │
                              ┌─────────────────────────────────────┤
                              ▼                                     ▼
                     Phase 4 (Notebook)                   Phase 5 (Autoresearch)
                                                                    │
                                                                    ▼
                                                          Phase 6 (Web app)
```

Phases 1-4 are the MVP. Phase 5 enables overnight autoresearch. Phase 6 is forward testing.

---

## Key Design Principles

1. **SQLite is the single source of truth.** All data flows through the DB. No loose CSVs or parquet files. Every module reads from and writes to the DB. The DB travels between local and RunPod — `run_remote.py` uploads it before training and downloads it after.

2. **Config-driven everything.** Every parameter lives in `config.yaml`. Changing what you predict, which tickers, how long the context is — it's all config. Code never hardcodes these values. This is what makes autoresearch possible.

3. **Same code runs locally and on RunPod.** The `market/`, `model/`, and `eval/` modules don't know about RunPod. They read from DB, do their work, write to DB. `run_remote.py` is the only file that knows about RunPod — it's just a shuttle for the DB and code.

4. **Data sources are pluggable.** `DataSource` is an abstract class. yfinance is the first implementation. Adding FRED, CSV import, or Kaggle is just a new file in `market/sources/` that implements the same interface. `download.py` reads `config.yaml` to know which source to use.

5. **Experiments are reproducible.** Every fine-tuning run snapshots its full config to `experiments.config_json`. You can always re-read what produced a given result. Autoresearch builds on this — it reads past experiments to decide what to try next.

6. **Evaluation is honest.** Purged k-fold CV with embargo (de Prado). Folds are time-based — all tickers share the same temporal boundaries. Purge and embargo prevent label and feature leakage. Walk-forward is a final sanity check only, not used for model selection.

7. **Features are extensible.** Base features (returns, log_returns, volume) are computed in Phase 1. Derived features (volatility, Fourier, momentum, cross-asset) follow the `FeatureGenerator` interface and register in a dict. Config says which to include as Toto exogenous variables. Adding a new feature = one new file, one registry entry, one config line.
