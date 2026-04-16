"""Autoresearch train.py — the agent's canvas.

This is the ONE FILE the autoresearch agent modifies between experiments.
Everything is fair game: feature engineering, dataset construction,
hyperparameters, context strategy, target transforms, etc.

Protected modules (agent MUST NOT modify):
  - market/db.py          — DB schema and access
  - market/download.py    — data fetching
  - eval/evaluate.py      — metric computation (evaluation honesty)
  - run_remote.py         — infrastructure

Usage (on RunPod pod):
    python -m autoresearch.train                    # run with defaults
    python -m autoresearch.train --mode finetune    # fine-tune then forecast
    python -m autoresearch.train --mode zeroshot    # zero-shot only
    python -m autoresearch.train --experiment-id my_exp_001

The script prints a machine-readable result line at the end:
    RESULT|experiment_id|directional_acc|mae|sharpe|status
"""

import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# CONFIGURATION — agent can change any of these
# ============================================================================

EXPERIMENT_CONFIG = {
    # What to predict
    "target": "returns",  # returns, log_returns, vol_adj_returns
    # Model / training
    "base_model": "Datadog/Toto-Open-Base-1.0",
    "mode": "finetune",  # zeroshot, finetune
    "context_factor": 8,  # context_length = 64 * context_factor
    "prediction_length": 5,  # trading days ahead
    "batch_size": 16,
    "learning_rate": 4.0e-5,
    "warmup_steps": 1000,
    "stable_steps": 200,
    "decay_steps": 200,
    "num_samples": 256,
    # Data
    "train_tickers": [
        "CL=F",
        "BZ=F",
        "NG=F",
        "HO=F",
        "RB=F",  # Energy
        "GC=F",
        "SI=F",
        "HG=F",  # Metals
        "ZC=F",
        "ZW=F",
        "ZS=F",
        "KC=F",
        "SB=F",
        "CT=F",  # Agriculture
        "USO",
        "UNG",
        "GLD",
        "SLV",
        "DBA",
        "DBC",  # ETFs
    ],
    "eval_tickers": ["CL=F", "NG=F", "GC=F", "HG=F", "ZS=F"],
    "fold": 2,  # CV fold to use
    # Evaluation
    "primary_metric": "directional_acc",
}


# ============================================================================
# FEATURE ENGINEERING — agent can add/modify feature computations here
# ============================================================================


def compute_custom_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute any custom features the agent wants.

    Input: DataFrame with columns [ticker, date, close, returns, log_returns,
           vol_adj_returns, normalized, volume]
    Output: Same DataFrame, potentially with new columns added.

    The agent can add any feature computations here. Examples:
    - Rolling volatility windows
    - Momentum indicators
    - Cross-asset features
    - Fourier-derived features
    - Custom target transforms
    """
    # === Agent: add feature engineering below this line ===

    return df


def build_target(df: pd.DataFrame, target_col: str) -> np.ndarray:
    """Build the prediction target from feature data.

    The agent can modify this to create custom targets. For example:
    - Clip extreme returns
    - Apply winsorization
    - Create binary up/down targets
    - Combine multiple signals
    """
    values = df[target_col].values.astype(np.float64)
    # Replace NaN with 0 (Toto expects no NaNs)
    values[np.isnan(values)] = 0.0
    return values


# ============================================================================
# DATASET CONSTRUCTION — agent can change how data is prepared for Toto
# ============================================================================


def build_training_data(conn, config: dict) -> list[dict]:
    """Build training dataset from DB.

    Returns list of series dicts, one per ticker.
    Agent can modify: which tickers, date filtering, feature selection.
    """
    from market.db import get_cv_folds, get_feature_data, query_df

    # Get fold boundaries
    splits = query_df(conn, "SELECT DISTINCT split_id FROM cv_folds")
    split_id = splits["split_id"].iloc[0]
    folds = get_cv_folds(conn, split_id)
    train_folds = [
        f for f in folds if f["split"] == "train" and f["fold_number"] == config["fold"]
    ]

    if not train_folds:
        raise ValueError(f"No train fold for fold={config['fold']}")

    train_fold = train_folds[0]
    train_start = train_fold["start_date"]
    train_end = train_fold["end_date"]

    logger.info("Training period: %s to %s", train_start, train_end)

    series_list = []
    for ticker in config["train_tickers"]:
        df = get_feature_data(conn, [ticker], train_start, train_end)
        if df.empty:
            continue

        df = df.sort_values("date")
        df = compute_custom_features(df)
        target_values = build_target(df, config["target"])

        series_list.append(
            {
                "ticker": ticker,
                "timestamp": df["date"].tolist(),
                config["target"]: target_values.tolist(),
            }
        )

    logger.info("Built training data: %d series", len(series_list))
    return series_list


def build_hf_dataset(series_list: list[dict], target_col: str):
    """Convert series list to HuggingFace Dataset for Toto."""
    from datasets import Dataset

    rows = []
    for s in series_list:
        rows.append(
            {
                "timestamp": s["timestamp"],
                target_col: s[target_col],
            }
        )

    hf_dataset = Dataset.from_list(rows)

    def drop_nan_fn(arr):
        arr = np.array(arr, dtype=np.float64)
        arr[np.isnan(arr)] = 0.0
        return arr

    return {
        "dataset_name": "commodities_autoresearch",
        "dataset": hf_dataset,
        "target_fields": [target_col],
        "target_transform_fns": [drop_nan_fn],
    }


# ============================================================================
# TRAINING — agent can modify the Toto training config
# ============================================================================


def build_toto_config(config: dict) -> dict:
    """Build Toto's fine-tuning config dict.

    Agent can modify: learning rate schedule, batch size, validation frequency,
    checkpoint strategy, etc.
    """
    total_steps = (
        config["warmup_steps"] + config["stable_steps"] + config["decay_steps"]
    )

    return {
        "seed": 42,
        "pretrained_model": config["base_model"],
        "model": {
            "val_prediction_len": config["prediction_length"],
            "lr": config["learning_rate"],
            "min_lr": config["learning_rate"] / 4,
            "warmup_steps": config["warmup_steps"],
            "stable_steps": config["stable_steps"],
            "decay_steps": config["decay_steps"],
        },
        "data": {
            "context_factor": config["context_factor"],
            "train_batch_size": config["batch_size"],
            "val_batch_size": 1,
            "num_workers": 0,
            "num_train_samples": 100,
            "add_exogenous_features": False,
            "prediction_horizon": config["prediction_length"],
            "max_rows": 5000,
        },
        "trainer": {
            "max_steps": total_steps,
            "log_every_n_steps": 10,
            "num_sanity_val_steps": 1,
            "enable_progress_bar": True,
            "refresh_rate": 1,
            "val_check_interval": 100,
        },
        "checkpoint": {
            "dirpath": str(PROJECT_ROOT / "checkpoints"),
            "filename": "{epoch}-{step}-{val_loss:.4f}",
            "monitor": "val_loss",
            "mode": "min",
            "save_top_k": 1,
        },
        "logging": {
            "save_dir": str(PROJECT_ROOT / "lightning_logs"),
            "name": "autoresearch",
        },
    }


def run_finetune(conn, config: dict, experiment_id: str) -> str | None:
    """Fine-tune Toto. Returns checkpoint path or None."""
    from toto.scripts import finetune_toto as finetune

    series_list = build_training_data(conn, config)
    if not series_list:
        raise ValueError("No training data")

    custom_dataset = build_hf_dataset(series_list, config["target"])
    toto_config = build_toto_config(config)

    logger.info("Starting fine-tuning (%d series)...", len(series_list))
    lightning_module, patch_size = finetune.init_lightning(toto_config)
    datamodule = finetune.get_datamodule(
        toto_config, patch_size, custom_dataset, setup=True
    )
    _, best_ckpt_path, best_val_loss = finetune.train(
        lightning_module, datamodule, toto_config
    )

    logger.info(
        "Fine-tuning done: val_loss=%.4f, checkpoint=%s", best_val_loss, best_ckpt_path
    )

    # Update experiment with training results
    conn.execute(
        "UPDATE experiments SET val_loss = ?, checkpoint_path = ? WHERE experiment_id = ?",
        (best_val_loss, best_ckpt_path, experiment_id),
    )

    return best_ckpt_path


# ============================================================================
# FORECASTING — agent can modify context strategy, sampling, etc.
# ============================================================================


def run_forecast(conn, config: dict, experiment_id: str, checkpoint_path: str | None):
    """Run forecasts on eval tickers for the test fold."""
    import torch
    from toto.inference.forecaster import TotoForecaster

    from market.db import get_cv_folds, get_feature_data, query_df
    from model.forecast import build_masked_timeseries, load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Forecasting on device: %s", device)

    model = load_model(checkpoint_path, config["base_model"], device)
    forecaster = TotoForecaster(model.model if hasattr(model, "model") else model)

    patch_size = 64
    context_length = patch_size * config["context_factor"]
    prediction_length = config["prediction_length"]
    num_samples = config["num_samples"]

    # Get test fold
    splits = query_df(conn, "SELECT DISTINCT split_id FROM cv_folds")
    split_id = splits["split_id"].iloc[0]
    folds = get_cv_folds(conn, split_id)
    test_folds = [
        f for f in folds if f["split"] == "test" and f["fold_number"] == config["fold"]
    ]

    if not test_folds:
        raise ValueError(f"No test fold for fold={config['fold']}")

    test_fold = test_folds[0]
    test_start = test_fold["start_date"]
    test_end = test_fold["end_date"]
    logger.info("Test period: %s to %s", test_start, test_end)

    total_predictions = 0

    for ticker in config["eval_tickers"]:
        df = get_feature_data(conn, [ticker], end_date=test_end)
        if df.empty:
            logger.warning("No data for %s, skipping", ticker)
            continue

        df = df.sort_values("date")
        df = compute_custom_features(df)  # Apply same features as training
        target_col = config["target"]
        all_values = build_target(df, target_col)
        all_dates = df["date"].tolist()

        # Find test period indices
        test_mask = (df["date"] >= test_start) & (df["date"] <= test_end)
        test_indices = df.index[test_mask].tolist()

        if len(test_indices) < prediction_length:
            logger.warning("Too few test points for %s, skipping", ticker)
            continue

        forecast_origins = list(
            range(
                test_indices[0],
                test_indices[-1] - prediction_length + 1,
                prediction_length,
            )
        )

        logger.info("  %s: %d forecast origins", ticker, len(forecast_origins))

        for origin_idx in forecast_origins:
            ctx_end = origin_idx
            ctx_start = max(0, ctx_end - context_length)
            ctx_values = all_values[ctx_start:ctx_end]
            ctx_dates = all_dates[ctx_start:ctx_end]

            if len(ctx_values) < patch_size:
                continue

            masked_ts = build_masked_timeseries(ctx_values, ctx_dates, device)
            forecast = forecaster.forecast(
                masked_ts,
                prediction_length=prediction_length,
                num_samples=num_samples,
                samples_per_batch=num_samples,
                use_kv_cache=True,
            )

            sample_device = (
                forecast.samples.device if hasattr(forecast, "samples") else device
            )
            median = forecast.median.cpu().numpy().flatten()
            q05 = (
                forecast.quantile(q=torch.tensor([0.05], device=sample_device))
                .cpu()
                .numpy()
                .flatten()
            )
            q95 = (
                forecast.quantile(q=torch.tensor([0.95], device=sample_device))
                .cpu()
                .numpy()
                .flatten()
            )

            forecast_date = all_dates[origin_idx - 1]

            for h in range(prediction_length):
                target_idx = origin_idx + h
                if target_idx >= len(all_dates):
                    break

                conn.execute(
                    """INSERT OR REPLACE INTO forecasts
                       (experiment_id, ticker, forecast_date, target_date,
                        horizon, predicted, predicted_q05, predicted_q95, actual)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        experiment_id,
                        ticker,
                        forecast_date,
                        all_dates[target_idx],
                        h + 1,
                        float(median[h]),
                        float(q05[h]),
                        float(q95[h]),
                        float(all_values[target_idx]),
                    ),
                )
                total_predictions += 1

    logger.info("Wrote %d predictions", total_predictions)
    return total_predictions


# ============================================================================
# EVALUATION — uses the protected eval module (agent cannot modify metrics)
# ============================================================================


def run_evaluation(conn, config: dict, experiment_id: str) -> dict:
    """Evaluate forecasts. Returns metrics dict."""
    from eval.evaluate import compute_metrics
    from market.db import query_df

    forecasts = query_df(
        conn,
        "SELECT * FROM forecasts WHERE experiment_id = ?",
        (experiment_id,),
    )

    if forecasts.empty:
        return {
            "directional_acc": None,
            "mae": None,
            "sharpe": None,
            "n_predictions": 0,
        }

    cv_fold = config["fold"]

    # Per-ticker metrics
    for ticker in sorted(forecasts["ticker"].unique()):
        ticker_preds = forecasts[forecasts["ticker"] == ticker]
        metrics = compute_metrics(ticker_preds)
        conn.execute(
            """INSERT OR REPLACE INTO evaluations
               (experiment_id, ticker, cv_fold, mae, rmse, directional_acc, sharpe, n_predictions)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                experiment_id,
                ticker,
                cv_fold,
                metrics["mae"],
                metrics["rmse"],
                metrics["directional_acc"],
                metrics["sharpe"],
                metrics["n_predictions"],
            ),
        )

    # Aggregate metrics (the number that matters)
    all_metrics = compute_metrics(forecasts)
    conn.execute(
        """INSERT OR REPLACE INTO evaluations
           (experiment_id, ticker, cv_fold, mae, rmse, directional_acc, sharpe, n_predictions)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            experiment_id,
            "ALL",
            cv_fold,
            all_metrics["mae"],
            all_metrics["rmse"],
            all_metrics["directional_acc"],
            all_metrics["sharpe"],
            all_metrics["n_predictions"],
        ),
    )

    return all_metrics


# ============================================================================
# MAIN PIPELINE — the full experiment loop
# ============================================================================


def run_experiment(
    config: dict | None = None,
    experiment_id: str | None = None,
    mode: str | None = None,
) -> dict:
    """Run one complete experiment: train → forecast → evaluate → report.

    Returns dict with experiment_id, metrics, and status.
    """
    if config is None:
        config = EXPERIMENT_CONFIG.copy()

    if mode is not None:
        config["mode"] = mode

    if experiment_id is None:
        experiment_id = f"ar_{config['mode']}_{uuid.uuid4().hex[:8]}"

    from market.db import get_db

    logger.info("=" * 60)
    logger.info("  AUTORESEARCH EXPERIMENT: %s", experiment_id)
    logger.info(
        "  Mode: %s | Target: %s | LR: %s",
        config["mode"],
        config["target"],
        config["learning_rate"],
    )
    logger.info("=" * 60)

    with get_db() as conn:
        # Log experiment start
        conn.execute(
            """INSERT OR REPLACE INTO experiments
               (experiment_id, created_at, config_json, cv_fold, status, notes)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                experiment_id,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(config),
                config["fold"],
                "running",
                f"autoresearch: mode={config['mode']}, target={config['target']}",
            ),
        )

        checkpoint_path = None
        try:
            # Step 1: Fine-tune (if not zero-shot)
            if config["mode"] == "finetune":
                checkpoint_path = run_finetune(conn, config, experiment_id)

            # Step 2: Forecast
            n_predictions = run_forecast(conn, config, experiment_id, checkpoint_path)

            # Step 3: Evaluate
            metrics = run_evaluation(conn, config, experiment_id)

            # Step 4: Mark complete
            conn.execute(
                "UPDATE experiments SET status = 'completed' WHERE experiment_id = ?",
                (experiment_id,),
            )

            # Print machine-readable result line (agent parses this)
            da = metrics.get("directional_acc")
            mae = metrics.get("mae")
            sharpe = metrics.get("sharpe")
            print(
                f"\nRESULT|{experiment_id}"
                f"|directional_acc={da}"
                f"|mae={mae}"
                f"|sharpe={sharpe}"
                f"|n={metrics.get('n_predictions', 0)}"
                f"|status=completed"
            )

            # Print human-readable summary
            print(f"\n{'=' * 60}")
            print(f"  EXPERIMENT COMPLETE: {experiment_id}")
            print(
                f"  Directional Accuracy: {da:.1%}"
                if da
                else "  Directional Accuracy: N/A"
            )
            print(f"  MAE:                  {mae:.6f}" if mae else "  MAE: N/A")
            print(
                f"  Sharpe:               {sharpe:.2f}" if sharpe else "  Sharpe: N/A"
            )
            print(f"  Predictions:          {n_predictions}")
            print(f"{'=' * 60}\n")

            return {
                "experiment_id": experiment_id,
                "metrics": metrics,
                "status": "completed",
                "checkpoint_path": checkpoint_path,
            }

        except Exception as e:
            logger.error("Experiment failed: %s", e)
            conn.execute(
                "UPDATE experiments SET status = 'failed', notes = ? WHERE experiment_id = ?",
                (f"FAILED: {e}", experiment_id),
            )
            print(f"\nRESULT|{experiment_id}|status=failed|error={e}")
            raise


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Autoresearch experiment runner")
    parser.add_argument(
        "--mode", choices=["zeroshot", "finetune"], help="Override mode"
    )
    parser.add_argument("--experiment-id", type=str, help="Custom experiment ID")
    parser.add_argument("--target", type=str, help="Override target column")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate")
    parser.add_argument("--context-factor", type=int, help="Override context factor")
    parser.add_argument("--fold", type=int, help="Override CV fold")
    args = parser.parse_args()

    config = EXPERIMENT_CONFIG.copy()
    if args.target:
        config["target"] = args.target
    if args.learning_rate:
        config["learning_rate"] = args.learning_rate
    if args.context_factor:
        config["context_factor"] = args.context_factor
    if args.fold is not None:
        config["fold"] = args.fold

    run_experiment(config=config, experiment_id=args.experiment_id, mode=args.mode)


if __name__ == "__main__":
    main()
