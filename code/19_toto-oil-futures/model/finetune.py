"""Fine-tuning module — fine-tune Toto on commodities data from DB.

Reads prepared data from SQLite, builds Toto's custom_dataset dict,
runs the fine-tuning pipeline, logs experiment to DB.

Usage:
    python -m model.finetune                  # train on all tickers, all folds
    python -m model.finetune --fold 0         # train on fold 0 only
    python -m model.finetune --config '{"model": {"learning_rate": 5e-5}}'
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
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

sys.path.insert(0, str(PROJECT_ROOT))


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def merge_config(base: dict, overrides: dict) -> dict:
    """Deep merge overrides into base config."""
    result = base.copy()
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result


def build_hf_dataset(series_list: list[dict], target_col: str):
    """Convert list of series dicts to HuggingFace Dataset for Toto.

    Each series dict has: ticker, timestamp (list), target_col (list), volume (list), close (list)
    Returns Toto's custom_dataset dict format.
    """
    from datasets import Dataset

    # Each row in HF Dataset = one time series
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
        """Replace NaN with 0."""
        arr = np.array(arr, dtype=np.float64)
        arr[np.isnan(arr)] = 0.0
        return arr

    custom_dataset = {
        "dataset_name": "commodities",
        "dataset": hf_dataset,
        "target_fields": [target_col],
        "target_transform_fns": [drop_nan_fn],
    }

    return custom_dataset


def build_toto_finetune_config(config: dict) -> dict:
    """Convert our config.yaml into Toto's expected finetune config format."""
    model_cfg = config["model"]

    return {
        "seed": 42,
        "pretrained_model": model_cfg["base_model"],
        "model": {
            "val_prediction_len": model_cfg["prediction_length"],
            "lr": model_cfg["learning_rate"],
            "min_lr": model_cfg["learning_rate"] / 4,
            "warmup_steps": model_cfg["warmup_steps"],
            "stable_steps": model_cfg["stable_steps"],
            "decay_steps": model_cfg["decay_steps"],
        },
        "data": {
            "context_factor": model_cfg["context_factor"],
            "train_batch_size": model_cfg["batch_size"],
            "val_batch_size": 1,
            "num_workers": 0,
            "num_train_samples": 100,
            "add_exogenous_features": False,
            "prediction_horizon": None,
            "max_rows": 5000,
        },
        "trainer": {
            "max_steps": model_cfg["warmup_steps"]
            + model_cfg["stable_steps"]
            + model_cfg["decay_steps"],
            "log_every_n_steps": 10,
            "num_sanity_val_steps": 1,
            "enable_progress_bar": True,
            "refresh_rate": 1,
            "val_check_interval": 100,
        },
        "checkpoint": {
            "dirpath": str(CHECKPOINT_DIR),
            "filename": "{epoch}-{step}-{val_loss:.4f}",
            "monitor": "val_loss",
            "mode": "min",
            "save_top_k": 1,
        },
        "logging": {
            "save_dir": str(PROJECT_ROOT / "lightning_logs"),
            "name": "toto_finetuning",
        },
    }


def run_finetune(
    config: dict | None = None,
    fold: int | None = None,
    config_overrides: str | None = None,
    experiment_id: str | None = None,
):
    """Run fine-tuning pipeline."""
    if config is None:
        config = load_config()

    if config_overrides:
        overrides = json.loads(config_overrides)
        config = merge_config(config, overrides)

    from market.db import get_db, get_cv_folds, query_df
    from market.prepare import build_toto_dataset

    data_cfg = config["data"]
    target_col = data_cfg["target"]
    train_tickers = data_cfg["train_tickers"]

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if experiment_id is None:
        experiment_id = f"finetune_{uuid.uuid4().hex[:8]}"

    with get_db() as conn:
        # Get CV fold info
        splits = query_df(conn, "SELECT DISTINCT split_id FROM cv_folds")
        if splits.empty:
            raise ValueError("No CV folds. Run prepare first.")
        split_id = splits["split_id"].iloc[0]

        folds = get_cv_folds(conn, split_id)
        train_folds = [f for f in folds if f["split"] == "train"]

        if fold is not None:
            train_folds = [f for f in train_folds if f["fold_number"] == fold]

        if not train_folds:
            raise ValueError(f"No train fold for fold={fold}")

        # Use first matching fold
        train_fold = train_folds[0]
        fold_num = train_fold["fold_number"]
        train_start = train_fold["start_date"]
        train_end = train_fold["end_date"]

        logger.info(
            "=== Fine-tuning on fold %d: train [%s, %s] ===",
            fold_num,
            train_start,
            train_end,
        )

        # Build dataset from DB
        logger.info("Building training dataset...")
        series_list = build_toto_dataset(
            conn, train_tickers, train_start, train_end, target_col
        )

        if not series_list:
            raise ValueError("No training data")

        logger.info("Built dataset: %d series", len(series_list))

        # Log experiment
        conn.execute(
            """INSERT OR REPLACE INTO experiments
               (experiment_id, created_at, config_json, cv_fold, status, notes)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                experiment_id,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(config),
                fold_num,
                "running",
                f"Fine-tune fold {fold_num}, target={target_col}, lr={config['model']['learning_rate']}",
            ),
        )

    # Build HF dataset and Toto config
    custom_dataset = build_hf_dataset(series_list, target_col)
    toto_config = build_toto_finetune_config(config)

    # Run fine-tuning
    logger.info("Starting Toto fine-tuning...")
    from toto.scripts import finetune_toto as finetune

    lightning_module, patch_size = finetune.init_lightning(toto_config)
    datamodule = finetune.get_datamodule(
        toto_config, patch_size, custom_dataset, setup=True
    )
    _, best_ckpt_path, best_val_loss = finetune.train(
        lightning_module, datamodule, toto_config
    )

    logger.info("Best checkpoint: %s (val_loss=%.4f)", best_ckpt_path, best_val_loss)

    # Update experiment record
    with get_db() as conn:
        conn.execute(
            """UPDATE experiments
               SET status = 'completed', checkpoint_path = ?, val_loss = ?
               WHERE experiment_id = ?""",
            (best_ckpt_path, best_val_loss, experiment_id),
        )

    logger.info("=== Fine-tuning complete: %s ===", experiment_id)
    return experiment_id, best_ckpt_path


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Toto on commodities data")
    parser.add_argument("--fold", type=int, help="CV fold to train on")
    parser.add_argument("--config", type=str, help="JSON config overrides")
    parser.add_argument("--experiment-id", type=str, help="Custom experiment ID")
    args = parser.parse_args()

    run_finetune(
        fold=args.fold,
        config_overrides=args.config,
        experiment_id=args.experiment_id,
    )


if __name__ == "__main__":
    main()
