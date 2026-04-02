"""Forecast module — run Toto inference and write predictions to DB.

Loads a Toto model (base for zero-shot, or fine-tuned checkpoint),
builds MaskedTimeseries inputs from DB feature data, runs the forecaster,
and writes predictions to the forecasts table.

Usage:
    python -m model.forecast                          # zero-shot, all eval tickers, latest fold
    python -m model.forecast --checkpoint path/to/ckpt  # fine-tuned model
    python -m model.forecast --tickers CL=F GC=F      # specific tickers
    python -m model.forecast --fold 2                  # specific CV fold
"""

import argparse
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Add parent to path so market package is importable
sys.path.insert(0, str(PROJECT_ROOT))


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_masked_timeseries(
    series_values: np.ndarray, timestamps: list[str], device: torch.device
):
    """Build a MaskedTimeseries input for Toto from a single univariate series.

    Args:
        series_values: 1D array of float values (the target column)
        timestamps: list of ISO date strings, same length as series_values
        device: torch device
    """
    from toto.data.util.dataset import MaskedTimeseries

    n = len(series_values)
    series_tensor = torch.tensor(series_values, dtype=torch.float32).unsqueeze(
        0
    )  # (1, n)

    padding_mask = torch.ones(1, n, dtype=torch.bool)
    id_mask = torch.zeros(1, n, dtype=torch.long)

    # Convert dates to unix timestamps
    ts_seconds = []
    for d in timestamps:
        dt = pd.Timestamp(d)
        ts_seconds.append(dt.timestamp())
    timestamp_tensor = torch.tensor(ts_seconds, dtype=torch.float64).unsqueeze(
        0
    )  # (1, n)

    # Daily data = 86400 seconds interval
    interval = torch.tensor([86400.0], dtype=torch.float64)  # (1,)

    return MaskedTimeseries(
        series=series_tensor.to(device),
        padding_mask=padding_mask.to(device),
        id_mask=id_mask.to(device),
        timestamp_seconds=timestamp_tensor.to(device),
        time_interval_seconds=interval.to(device),
        num_exogenous_variables=0,
    )


def load_model(checkpoint_path: str | None, base_model: str, device: torch.device):
    """Load Toto model — base (zero-shot) or fine-tuned from checkpoint."""
    from toto.model.toto import Toto

    if checkpoint_path:
        logger.info("Loading fine-tuned model from %s", checkpoint_path)
        from toto.scripts.finetune_toto import load_finetuned_toto

        model = load_finetuned_toto(base_model, checkpoint_path, device)
    else:
        logger.info("Loading base model %s (zero-shot)", base_model)
        model = Toto.from_pretrained(base_model).to(device)

    model.eval()
    return model


def forecast_ticker(
    forecaster,
    series_values: np.ndarray,
    timestamps: list[str],
    prediction_length: int,
    num_samples: int,
    context_length: int,
    device: torch.device,
) -> dict:
    """Run forecast for a single ticker.

    Uses a rolling window: takes the last context_length points, forecasts
    prediction_length steps ahead.

    Returns dict with forecast arrays.
    """
    # Use last context_length points as input
    if len(series_values) < context_length:
        logger.warning(
            "Series too short (%d < %d context_length), using all available",
            len(series_values),
            context_length,
        )
        input_values = series_values
        input_timestamps = timestamps
    else:
        input_values = series_values[-context_length:]
        input_timestamps = timestamps[-context_length:]

    masked_ts = build_masked_timeseries(input_values, input_timestamps, device)

    forecast = forecaster.forecast(
        masked_ts,
        prediction_length=prediction_length,
        num_samples=num_samples,
        use_kv_cache=True,
    )

    result = {
        "median": forecast.median.cpu().numpy().flatten(),
        "q05": forecast.quantile(q=torch.tensor([0.05])).cpu().numpy().flatten(),
        "q95": forecast.quantile(q=torch.tensor([0.95])).cpu().numpy().flatten(),
    }
    return result


def run_forecasts(
    config: dict | None = None,
    checkpoint_path: str | None = None,
    tickers: list[str] | None = None,
    fold: int | None = None,
    experiment_id: str | None = None,
    notes: str | None = None,
    target: str | None = None,
):
    """Run forecasts for specified tickers and fold, write results to DB."""
    if config is None:
        config = load_config()

    from market.db import get_db, get_feature_data, get_cv_folds, query_df

    model_cfg = config["model"]
    data_cfg = config["data"]
    target_col = target or data_cfg["target"]
    logger.info("Target column: %s", target_col)
    prediction_length = model_cfg["prediction_length"]
    num_samples = model_cfg["num_samples"]
    context_factor = model_cfg["context_factor"]
    patch_size = 64  # Toto's fixed patch size
    context_length = patch_size * context_factor

    if tickers is None:
        tickers = data_cfg["eval_tickers"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Load model
    model = load_model(checkpoint_path, model_cfg["base_model"], device)

    from toto.inference.forecaster import TotoForecaster

    forecaster = TotoForecaster(model.model if hasattr(model, "model") else model)

    # Create experiment record
    if experiment_id is None:
        experiment_id = f"{'zeroshot' if not checkpoint_path else 'finetuned'}_{uuid.uuid4().hex[:8]}"

    with get_db() as conn:
        # Find the split_id
        splits = query_df(conn, "SELECT DISTINCT split_id FROM cv_folds")
        if splits.empty:
            raise ValueError("No CV folds found. Run prepare first.")
        split_id = splits["split_id"].iloc[0]

        # Get folds
        folds = get_cv_folds(conn, split_id)
        test_folds = [f for f in folds if f["split"] == "test"]

        if fold is not None:
            test_folds = [f for f in test_folds if f["fold_number"] == fold]

        if not test_folds:
            raise ValueError(f"No test fold found for fold={fold}")

        # Log experiment
        conn.execute(
            """INSERT OR REPLACE INTO experiments
               (experiment_id, created_at, config_json, cv_fold, status, checkpoint_path, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                experiment_id,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(config),
                fold,
                "running",
                checkpoint_path,
                notes
                or ("zero-shot baseline" if not checkpoint_path else "fine-tuned"),
            ),
        )

        total_predictions = 0

        for test_fold in test_folds:
            fold_num = test_fold["fold_number"]
            test_start = test_fold["start_date"]
            test_end = test_fold["end_date"]
            logger.info(
                "=== Fold %d: test [%s, %s] ===", fold_num, test_start, test_end
            )

            for ticker in tickers:
                # Get ALL data up to end of test period (need context before test start)
                df = get_feature_data(conn, [ticker], end_date=test_end)
                if df.empty:
                    logger.warning("No data for %s, skipping", ticker)
                    continue

                df = df.sort_values("date")
                all_values = df[target_col].values
                all_dates = df["date"].tolist()
                all_close = df["close"].values

                # Find test period indices
                test_mask = (df["date"] >= test_start) & (df["date"] <= test_end)
                test_indices = df.index[test_mask].tolist()

                if len(test_indices) < prediction_length:
                    logger.warning(
                        "Too few test points for %s fold %d, skipping", ticker, fold_num
                    )
                    continue

                # Rolling forecast through test period
                # Step through test set, making a forecast every prediction_length days
                forecast_origins = list(
                    range(
                        test_indices[0],
                        test_indices[-1] - prediction_length + 1,
                        prediction_length,
                    )
                )

                logger.info(
                    "  %s: %d forecast origins in test period",
                    ticker,
                    len(forecast_origins),
                )

                for origin_idx in forecast_origins:
                    # Context is everything up to (but not including) the origin
                    ctx_end = origin_idx
                    ctx_start = max(0, ctx_end - context_length)

                    ctx_values = all_values[ctx_start:ctx_end].astype(np.float64)
                    ctx_dates = all_dates[ctx_start:ctx_end]

                    if len(ctx_values) < patch_size:
                        continue

                    # Run forecast
                    result = forecast_ticker(
                        forecaster,
                        ctx_values,
                        ctx_dates,
                        prediction_length,
                        num_samples,
                        context_length,
                        device,
                    )

                    forecast_date = all_dates[origin_idx - 1]  # last known date

                    # Write predictions
                    for h in range(prediction_length):
                        target_idx = origin_idx + h
                        if target_idx >= len(all_dates):
                            break

                        actual_val = float(all_values[target_idx])
                        target_date = all_dates[target_idx]

                        conn.execute(
                            """INSERT OR REPLACE INTO forecasts
                               (experiment_id, ticker, forecast_date, target_date,
                                horizon, predicted, predicted_q05, predicted_q95, actual)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                experiment_id,
                                ticker,
                                forecast_date,
                                target_date,
                                h + 1,
                                float(result["median"][h]),
                                float(result["q05"][h]),
                                float(result["q95"][h]),
                                actual_val,
                            ),
                        )
                        total_predictions += 1

                logger.info("  %s: wrote predictions", ticker)

        # Update experiment status
        conn.execute(
            "UPDATE experiments SET status = 'completed' WHERE experiment_id = ?",
            (experiment_id,),
        )

    logger.info(
        "=== Forecast complete: %d predictions, experiment=%s ===",
        total_predictions,
        experiment_id,
    )
    return experiment_id


def main():
    parser = argparse.ArgumentParser(description="Run Toto forecasts")
    parser.add_argument("--checkpoint", type=str, help="Path to fine-tuned checkpoint")
    parser.add_argument("--tickers", nargs="+", help="Override eval tickers")
    parser.add_argument("--fold", type=int, help="Specific CV fold to evaluate")
    parser.add_argument("--experiment-id", type=str, help="Custom experiment ID")
    parser.add_argument("--notes", type=str, help="Experiment notes")
    parser.add_argument(
        "--target",
        type=str,
        help="Target column override (returns, vol_adj_returns, log_returns)",
    )
    args = parser.parse_args()

    run_forecasts(
        checkpoint_path=args.checkpoint,
        tickers=args.tickers,
        fold=args.fold,
        experiment_id=args.experiment_id,
        notes=args.notes,
        target=args.target,
    )


if __name__ == "__main__":
    main()
