# Commodities Forecasting MVP — Claude Code Spec

## What We're Building
A minimal, modular system that:
1. Downloads commodities futures data
2. Fine-tunes the Toto time series foundation model (151M params) on that data
3. Generates price forecasts and evaluates accuracy
4. Is structured so we can later plug in Karpathy's autoresearch loop to iterate autonomously

## Compute
- **MVP**: RunPod on-demand pods (L4 GPU, 24GB VRAM — plenty for 151M params)
- **Autoresearch loop (later)**: Modal (per-second billing, $30/mo free credits, `@app.function(gpu="T4")`)
- All code should work locally (CPU) and on RunPod (GPU) with no changes

## Project Structure
```
19_toto-oil-futures/
├── README.md                 # Setup instructions, how to run
├── requirements.txt          # Pin versions
├── config.yaml               # All hyperparams and settings in one place
│
├── market/
│   ├── download.py           # Downloads commodities data via yfinance
│   └── prepare.py            # Cleans, normalizes, creates train/val/test splits
│                              # Outputs HuggingFace Dataset objects ready for Toto
│
├── model/
│   ├── finetune.py           # Fine-tunes Toto on prepared data
│   │                          # Loads from config.yaml, saves checkpoints
│   └── forecast.py           # Loads a checkpoint, runs inference, outputs predictions
│
├── eval/
│   └── evaluate.py           # Computes metrics: MAE, RMSE, directional accuracy
│                              # Compares forecast vs actuals on test set
│                              # Prints a single summary score (this becomes the autoresearch metric)
│
├── notebooks/
│   └── mvp_demo.ipynb        # End-to-end demo notebook (reads pre-computed results from DB)
│                              # Just calls the modules above in sequence
│                              # This is what we screen-record for the video
│
├── local-docs/               # Planning and technical docs (not in video)
│
└── autoresearch/             # Phase 2 — not for MVP, but structure it now
    ├── program.md            # Autoresearch skill file (directions for the agent)
    ├── train.py              # The single mutable file the agent modifies
    └── results.tsv           # Experiment log
```

## Module Details

### market/download.py
- Use `yfinance` to pull daily OHLCV for commodities futures
- Tickers to include (at minimum):
  - Energy: CL=F (crude oil), NG=F (natural gas), HO=F (heating oil), RB=F (gasoline)
  - Metals: GC=F (gold), SI=F (silver), HG=F (copper), PL=F (platinum)
  - Agriculture: ZC=F (corn), ZS=F (soybeans), ZW=F (wheat), KC=F (coffee), CT=F (cotton), SB=F (sugar)
- Pull maximum available history
- Save raw data as parquet files, one per ticker
- Print a summary: ticker, date range, row count, any gaps

### market/prepare.py
- Load raw parquet files
- Handle missing values (forward fill, then drop leading NaNs)
- Create features: close price (primary target), returns, log returns
- Split chronologically: train (70%) / val (15%) / test (15%)
- Convert to HuggingFace `Dataset` format compatible with Toto's fine-tuning pipeline
  - Toto expects a custom_dataset dict with:
    - `dataset_name`: string identifier
    - `dataset`: HF Dataset object
    - `target_fields`: list of column names for targets
    - `target_transform_fns`: list of callables (one per target field)
    - `ev_fields`: (optional) exogenous variable column names
    - `ev_transform_fns`: (optional) callables for exogenous vars
  - Each row = one time series with a `timestamp` column + target columns
  - Internally Toto converts to GluonTS format → CausalMaskedTimeseries tensors
  - See Toto repo's `benchmark_finetuning.py` and `toto/data/util/helpers.py` for the exact interface
- Save prepared datasets to disk

### model/finetune.py
- Load Toto from `Datadog/Toto-Open-Base-1.0` via HuggingFace
- Load prepared training data
- Fine-tune using Toto's built-in fine-tuning pipeline:
  ```python
  from toto.scripts import finetune_toto as finetune
  lightning_module, patch_size = finetune.init_lightning(config)
  datamodule = finetune.get_datamodule(config, patch_size, custom_dataset, setup=True)
  _, best_ckpt_path, best_val_loss = finetune.train(lightning_module, datamodule, config)
  trained_model = finetune.load_finetuned_toto(config["pretrained_model"], best_ckpt_path, device)
  ```
- Config from `config.yaml`: learning rate, epochs, context_factor, prediction length, batch size
- Note: context_length = patch_size (64) * context_factor
- Save fine-tuned checkpoint
- Log training loss curve

### model/forecast.py
- Load fine-tuned checkpoint (or base model for zero-shot baseline)
- Run inference on validation or test data using TotoForecaster
- Toto outputs probabilistic forecasts (median of N samples = point forecast)
- Output predictions as a dataframe: date, ticker, actual, predicted, prediction_intervals
- Save to CSV

### eval/evaluate.py
- Load predictions CSV
- Compute per-ticker and aggregate metrics:
  - MAE, RMSE (price-level accuracy)
  - Directional accuracy (did we get the direction right?)
  - This directional accuracy becomes the single metric for autoresearch
- Print a clean summary table
- Also compare fine-tuned vs zero-shot baseline

### notebooks/mvp_demo.ipynb
- Cell 1: Install dependencies (`pip install toto-ts yfinance`)
- Cell 2: Download data (calls market/download.py)
- Cell 3: Prepare data (calls market/prepare.py)
- Cell 4: Run zero-shot baseline forecast
- Cell 5: Fine-tune Toto
- Cell 6: Run fine-tuned forecast
- Cell 7: Evaluate and compare
- Cell 8: Plot actual vs predicted for a few tickers
- This is the demo — keep it visual and simple

## config.yaml defaults
```yaml
data:
  source: "yfinance"
  start_date: "2000-01-01"
  target: "returns"           # what we predict: close, returns, log_returns
  # All tickers used for fine-tuning (more data = better generalization)
  train_tickers:
    # Energy
    - CL=F   # Crude Oil WTI
    - BZ=F   # Brent Crude
    - NG=F   # Natural Gas
    - HO=F   # Heating Oil
    - RB=F   # RBOB Gasoline
    # Metals
    - GC=F   # Gold
    - SI=F   # Silver
    - HG=F   # Copper
    # Agriculture
    - ZC=F   # Corn
    - ZW=F   # Wheat
    - ZS=F   # Soybeans
    - KC=F   # Coffee
    - SB=F   # Sugar
    - CT=F   # Cotton
    # ETFs (cleaner data, no roll artifacts)
    - USO    # US Oil Fund
    - UNG    # US Natural Gas Fund
    - GLD    # Gold ETF
    - SLV    # Silver ETF
    - DBA    # Agriculture ETF
    - DBC    # Commodity Index ETF
  # Tickers we actually evaluate/forecast on
  eval_tickers:
    - CL=F   # Crude Oil — primary target
    - NG=F   # Natural Gas
    - GC=F   # Gold
    - HG=F   # Copper
    - ZS=F   # Soybeans
  exogenous_features: []      # future: ["volatility_20d", "fourier_dominant_freq"]

cv:
  method: "purged_kfold"      # purged_kfold | cpcv | walk_forward
  n_folds: 3                  # 3 for MVP (limited data), 5+ with more data
  embargo_pct: 0.01           # 1% of data ≈ 50 trading days
  purge_days: 5               # must match prediction_length (label overlap)
  # CPCV params (Phase 2)
  # cpcv_n_groups: 6
  # cpcv_p_test: 2

model:
  base_model: "Datadog/Toto-Open-Base-1.0"
  context_factor: 8           # context_length = patch_size(64) * context_factor = 512
  prediction_length: 5        # Forecast horizon (trading days)
  batch_size: 16
  learning_rate: 4.0e-5       # Toto default
  warmup_steps: 1000
  stable_steps: 200
  decay_steps: 200
  num_samples: 256            # Toto uses median of N samples for point forecast

eval:
  primary_metric: "directional_accuracy"  # This is the autoresearch target
```

## Key Implementation Notes

1. **Toto's interface**: Toto is a probabilistic model — it outputs a distribution, not a point forecast. Take the median of 256 samples as the point forecast (this is what the Toto authors recommend).

2. **Chronological splits only**: Never shuffle time series data. Train on past, validate on future, test on further future.

3. **Keep it univariate per ticker for MVP**: Fine-tune one model on one commodity at a time initially. Multivariate (multiple tickers as covariates) is a Phase 2 feature.

4. **Zero-shot baseline is important**: Always compare fine-tuned performance against Toto's zero-shot capability. If zero-shot is already good, fine-tuning might overfit.

5. **Toto data pipeline**: HF Dataset → GluonTS format (via transform_fev_dataset) → CausalMaskedTimeseries tensors (via instance_to_causal) → batched via collate_causal.

## Autoresearch Integration (Phase 2 — just scaffold now)

The `autoresearch/` directory follows Karpathy's pattern:
- `train.py` is the single file the agent modifies
- It wraps `market/prepare.py` → `model/finetune.py` → `eval/evaluate.py` into one script
- It prints a single metric (directional accuracy) at the end
- `program.md` tells the agent what to explore:
  - Learning rate, context length, prediction length
  - Which tickers to include/exclude
  - Feature engineering (returns vs log returns vs normalized price)
  - Ensemble strategies (train on individual tickers, combine forecasts)
- `results.tsv` logs: experiment_id, description, metric, status (kept/discarded)
- The agent runs on Modal with a 5-minute time budget per experiment

## Immediate Build Order
1. Set up the project structure (create all directories and empty files)
2. Implement `market/download.py` and test it — make sure we get clean data
3. Implement `market/prepare.py` — get the HF Dataset format right for Toto
4. Implement `model/finetune.py` — get a single fine-tune run working
5. Implement `model/forecast.py` and `eval/evaluate.py`
6. Wire it all together in `notebooks/mvp_demo.ipynb`
7. Test end-to-end on one ticker (CL=F crude oil) before expanding
