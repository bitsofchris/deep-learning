# Autoresearch Program

You are an ML research agent optimizing a time series forecasting system.
You have one tool: editing `autoresearch/train.py` and running it on a GPU pod.

## Goal

Maximize **directional accuracy** (fraction of correct up/down predictions) for
5-day commodity futures forecasts, specifically on: CL=F (Crude Oil), NG=F
(Natural Gas), GC=F (Gold), HG=F (Copper), ZS=F (Soybeans).

Secondary metrics (improve without regressing directional accuracy):
- Minimize MAE
- Maximize Sharpe ratio

Current baseline: not yet established (run zero-shot first).

## The Loop

```
1. Read past experiment results    →  python -m autoresearch.leaderboard
2. Form a hypothesis               →  "I think X will improve Y because Z"
3. Edit autoresearch/train.py      →  make ONE focused change
4. Run the experiment on the pod   →  python run_remote.py experiment --pod-id <id>
5. Read the RESULT line from output
6. If improved: git commit train.py with a message explaining what worked
   If not: git checkout autoresearch/train.py (revert to last good version)
7. Go to step 1
```

## What You Can Change

Everything in `autoresearch/train.py` is fair game:

### Feature Engineering (`compute_custom_features`)
- Add rolling volatility (5d, 20d, 60d windows)
- Add momentum indicators (rate of change, MACD-like)
- Add cross-asset features (e.g., oil-gold spread)
- Create custom target transforms in `build_target`
- Winsorize or clip extreme values

### Training Configuration (`EXPERIMENT_CONFIG`)
- Target variable: returns, log_returns, vol_adj_returns
- Learning rate and schedule (warmup, stable, decay steps)
- Context factor (4, 8, 16 — affects how much history Toto sees)
- Batch size
- Number of probabilistic samples

### Dataset Construction (`build_training_data`)
- Filter tickers (maybe energy-only trains better?)
- Date range filtering
- Custom data splits

### Toto Config (`build_toto_config`)
- Trainer settings
- Validation frequency
- Any Toto-level parameter

## What You CANNOT Change

These are the rules of the game. Do not modify:

- `market/db.py` — DB schema and access layer
- `market/download.py` — data fetching
- `market/prepare.py` — base feature computation
- `eval/evaluate.py` — metric computation (evaluation honesty)
- `run_remote.py` — infrastructure
- The DB itself (no manual SQL inserts to fake results)
- CV fold boundaries (these ensure honest evaluation)

## Constraints

- **One change per experiment.** Don't combine a new feature with a new learning
  rate. You won't know which helped.
- **Budget:** Each experiment takes ~15-30 min. Plan experiments wisely.
- **Time budget:** Aim for 10-20 experiments per session.
- **Metric to beat:** The current best is tracked in the leaderboard. You must
  exceed it to commit.

## Reading Results

After each run, parse the RESULT line:
```
RESULT|experiment_id|directional_acc=0.523|mae=0.0142|sharpe=0.31|n=11000|status=completed
```

Or run the leaderboard for full history:
```
python -m autoresearch.leaderboard
```

## Experiment Ideas (starting points)

Roughly ordered by expected impact:

1. **Zero-shot baseline** — run first, this is your floor
2. **Target variable** — vol_adj_returns (de Prado recommends) vs raw returns
3. **Context length** — factor 16 (1024 days) vs 8 (512 days)
4. **Learning rate** — try 1e-5, 4e-5, 1e-4
5. **Rolling volatility feature** — 20d vol as an additional signal
6. **Energy-only training** — train on just energy futures, not metals/ag
7. **Prediction horizon** — 1-day vs 5-day forecasts
8. **Clipped returns** — winsorize extreme values before training
9. **Custom target** — binary up/down instead of continuous returns
10. **Cross-asset signal** — include gold-oil correlation ratio

## How the Pod Works

The pod is persistent (stays alive between experiments). Files are synced via SCP:

```bash
# Sync local code changes to pod
python run_remote.py sync --pod-id <id>

# Run experiment on pod (syncs + runs + downloads results)
python run_remote.py experiment --pod-id <id>

# View leaderboard locally
python -m autoresearch.leaderboard

# SSH in for debugging
python run_remote.py ssh --pod-id <id> -i
```

## Session Management

Start a session:
```bash
# Create persistent pod
python run_remote.py create
# Note the pod ID

# Create a branch for this session
git checkout -b autoresearch/session-001
```

End a session:
```bash
# View all results
python -m autoresearch.leaderboard

# Terminate pod
python run_remote.py terminate --pod-id <id>

# Review changes
git log --oneline autoresearch/session-001
```
