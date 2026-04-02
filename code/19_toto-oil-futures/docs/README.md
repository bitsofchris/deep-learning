# Toto Oil Futures — Commodities Forecasting with Toto

Fine-tune the [Toto](https://huggingface.co/Datadog/Toto-Open-Base-1.0) time series foundation model on commodities futures, evaluate with de Prado's purged k-fold CV, and see if it can beat zero-shot.

---

## Quick Start

```bash
cd code/19_toto-oil-futures
source /Users/chris/repos/deep-learning/.venv/bin/activate

# 1. Download data (already done — 20 tickers, 2000–2026)
python -m market.download

# 2. Compute features + CV folds (already done — 118k rows, 3 folds)
python -m market.prepare

# 3. Run experiments on RunPod (see Execution Plan below)
python run_remote.py forecast --fold 2
python run_remote.py finetune --fold 2
```

---

## Current Status

| Phase | Status | Notes |
|-------|--------|-------|
| Data (download + prepare) | **DONE** | 20 tickers, 118k rows, 3 purged CV folds |
| Zero-shot baseline | **Code done, needs run** | `run_remote.py forecast` |
| Fine-tuning | **Code done, needs run** | `run_remote.py finetune` |
| Demo notebook | Not started | `notebooks/` empty |
| Autoresearch | Not started | Phase 5 |

**DB state:** `market/commodities.db` has raw_ohlcv (118,689), features (118,669), cv_folds (3 folds). Experiments/forecasts/evaluations are empty — no GPU runs yet.

---

## Execution Plan (4 Experiments, All Parallel)

See [experiment-design.md](../local-docs/experiment-design.md) for full details.

**Question:** Can fine-tuning Toto on commodities beat zero-shot? How does vol-adjusted returns compare to raw returns?

```bash
cd code/19_toto-oil-futures

# Terminal 1: Zero-shot on returns (~5 min)
python run_remote.py forecast --fold 2 --target returns --experiment-id zs_returns

# Terminal 2: Zero-shot on vol-adjusted returns (~5 min)
python run_remote.py forecast --fold 2 --target vol_adj_returns --experiment-id zs_voladj

# Terminal 3: Fine-tune on returns (~30 min, TensorBoard auto-opens)
python run_remote.py finetune --fold 2 --target returns --experiment-id ft_returns

# Terminal 4: Fine-tune on vol-adjusted returns (~30 min)
python run_remote.py finetune --fold 2 --target vol_adj_returns --experiment-id ft_voladj
```

Each creates its own pod, runs independently, downloads to `results/<experiment_id>.db`.

```bash
# After all 4 finish — merge into local DB and compare
python -m eval.merge_results results/*.db
python -m eval.evaluate --compare zs_returns zs_voladj ft_returns_eval ft_voladj_eval
```

**Fold 2:** Train 2000–2017 (17 years), test 2017–2026 (8.7 years), 66-day embargo.
**Eval tickers:** CL=F (oil), NG=F, GC=F, HG=F, ZS=F.
**Success bar:** Beat naive always-up baseline of 52.2% directional accuracy.
**Total cost:** ~$0.50 (4 pods, ~30 min wall time).

---

## Command Reference

### Primary Commands (one command does everything)

```bash
# Zero-shot inference: create pod → forecast → evaluate → download → terminate
python run_remote.py forecast --fold 2 --target returns --experiment-id zs_returns

# Fine-tune + forecast: create pod → train → forecast → evaluate → download → terminate
python run_remote.py finetune --fold 2 --target returns --experiment-id ft_returns
```

Key flags:
- `--target` — which column to predict (`returns`, `vol_adj_returns`, `log_returns`)
- `--experiment-id` — name the experiment (used for DB labels + download filenames)
- `--fold` — CV fold number (0, 1, 2)
- `--keep-pod` — don't terminate after (for debugging)
- `--no-browser` — skip auto-opening TensorBoard

Both commands:
- Stream live logs to your terminal
- `finetune` auto-opens TensorBoard in browser (RunPod proxy)
- Training runs in tmux — survives SSH drops and Ctrl+C
- Downloads to `results/<experiment_id>.db` (not overwriting local DB)
- Auto-terminates pod (unless `--keep-pod`)

### Monitoring (while experiment runs)

```bash
python run_remote.py monitor --pod-id <id>       # tail live logs
python run_remote.py tensorboard --pod-id <id>    # TensorBoard in browser
python run_remote.py ssh --pod-id <id> -i         # interactive shell on pod
```

### Results (after download)

```bash
# Merge parallel pod results into local DB (idempotent, safe to re-run)
python -m eval.merge_results results/*.db

# Then query
python -m eval.evaluate --list                                    # list all experiments
python -m eval.evaluate --experiment-id <id>                      # metrics for one
python -m eval.evaluate --compare <id1> <id2>                     # side-by-side
python -m eval.evaluate --experiment-id <id> --tickers CL=F NG=F  # specific tickers
```

### Escape Hatches

```bash
python run_remote.py create                            # just create a pod
python run_remote.py run --pod-id <id> "command"       # run arbitrary command
python run_remote.py download --pod-id <id>            # pull results without terminating
python run_remote.py terminate --pod-id <id>           # stop billing
```

### Data Pipeline (local, already done)

```bash
python -m market.download     # fetch OHLCV → raw_ohlcv table
python -m market.prepare      # compute features + CV folds → features, cv_folds tables
```

### Parallel Experiments (merge results from separate pods)

```bash
python -m eval.merge_results results/*.db    # merge remote DBs into local DB
```

---

## What You See During a Run

1. **Terminal** — live streamed logs (model loading, training progress, per-ticker forecasts, metrics table)
2. **TensorBoard** (finetune only) — opens `https://{pod_id}-6006.proxy.runpod.net` with training/validation loss curves
3. **After completion** — formatted metrics table printed, results downloaded to local DB

If you Ctrl+C: training keeps running in tmux on the pod. Reconnect with `monitor`.

---

## What Gets Downloaded

| File | What | Where |
|------|------|-------|
| `market/commodities.db` | Updated DB with experiments, forecasts, evaluations | Queried by `eval.evaluate` |
| `results/run.log` | Full training output | Debug reference |
| `results/lightning_logs/` | TensorBoard event files | Plot training curves locally |

---

## Key Design Decisions

- **SQLite is the single source of truth** — all data flows through the DB, which shuttles between local and RunPod
- **Config-driven** — `config.yaml` controls everything (target, tickers, horizons, hyperparams)
- **Purged k-fold CV** (de Prado) — split by time not asset, 5-day purge, 66-day embargo
- **tmux on RunPod** — training survives SSH drops, no lost work
- **Experiment isolation** — each run gets a UUID, full config snapshot in DB

---

## Project Structure

```
19_toto-oil-futures/
├── config.yaml              # All settings (target, tickers, model hyperparams)
├── run_remote.py            # RunPod orchestration (the main entry point)
├── requirements.txt
│
├── market/
│   ├── db.py                # SQLite schema + connection + query helpers
│   ├── download.py          # Fetch OHLCV from yfinance → raw_ohlcv
│   ├── prepare.py           # Compute features, generate CV folds
│   ├── commodities.db       # The database (33MB, gitignored)
│   ├── sources/             # Pluggable data sources (yfinance, future: FRED)
│   └── features/            # Feature registry scaffold (future: volatility, fourier)
│
├── model/
│   ├── finetune.py          # Fine-tune Toto, log experiment to DB
│   └── forecast.py          # Run inference, write predictions to DB
│
├── eval/
│   ├── evaluate.py          # Compute metrics, compare experiments
│   └── merge_results.py     # Merge parallel pod results into local DB
│
├── results/                 # Downloaded run logs + lightning logs
├── checkpoints/             # Model checkpoints (gitignored)
├── notebooks/               # Demo notebook (Phase 4)
│
├── docs/
│   ├── README.md            # ← YOU ARE HERE
│   ├── architecture.md      # System design + SQLite schema (TODO: consolidate)
│   ├── runpod-reference.md  # RunPod setup, pricing, gotchas
│   └── toto-technical-notes.md  # Toto API, tensors, fine-tune config
│
└── local-docs/
    └── experiment-design.md # Detailed experiment plan, metrics, success criteria
```

---

## Docs Guide

| Doc | When to read |
|-----|-------------|
| **This README** | First. Setup, commands, status. |
| [experiment-design.md](../local-docs/experiment-design.md) | Before running experiments. Targets, metrics, success criteria. |
| [runpod-reference.md](runpod-reference.md) | If RunPod setup breaks. Pricing, SSH, gotchas. |
| [toto-technical-notes.md](toto-technical-notes.md) | If modifying model code. Tensor formats, API calls. |
| [phase-plan.md](phase-plan.md) | For the big picture roadmap and architecture diagram. |
