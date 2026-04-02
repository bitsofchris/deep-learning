"""Evaluation module — compute metrics from forecasts in DB.

Reads predictions from the forecasts table, computes MAE, RMSE,
and directional accuracy, writes results to evaluations table.

Usage:
    python -m eval.evaluate --experiment-id zeroshot_abc123
    python -m eval.evaluate --experiment-id zeroshot_abc123 --tickers CL=F
    python -m eval.evaluate --compare exp1 exp2       # side-by-side comparison
"""

import argparse
import logging
import sys
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


def compute_metrics(predictions: pd.DataFrame) -> dict:
    """Compute evaluation metrics from a predictions DataFrame.

    Expects columns: predicted, actual, horizon
    Returns dict with mae, rmse, directional_acc, sharpe, n_predictions.
    """
    df = predictions.dropna(subset=["predicted", "actual"])
    if df.empty:
        return {
            "mae": None,
            "rmse": None,
            "directional_acc": None,
            "sharpe": None,
            "n_predictions": 0,
        }

    errors = df["predicted"] - df["actual"]
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))

    # Directional accuracy: did the forecast get the sign right?
    # For returns: predicted > 0 means "up", actual > 0 means "up"
    pred_dir = (df["predicted"] > 0).astype(int)
    actual_dir = (df["actual"] > 0).astype(int)
    directional_acc = float((pred_dir == actual_dir).mean())

    # Simulated Sharpe: if we go long when predicted > 0, flat otherwise
    # The "return" we capture is actual * sign(predicted)
    strategy_returns = df["actual"] * np.where(df["predicted"] > 0, 1, -1)
    if strategy_returns.std() > 0:
        sharpe = float(np.sqrt(252) * strategy_returns.mean() / strategy_returns.std())
    else:
        sharpe = 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "directional_acc": directional_acc,
        "sharpe": sharpe,
        "n_predictions": len(df),
    }


def evaluate_experiment(
    experiment_id: str,
    tickers: list[str] | None = None,
):
    """Evaluate an experiment and write metrics to DB."""
    from market.db import get_db, query_df

    with get_db() as conn:
        # Get experiment info
        exp = query_df(
            conn,
            "SELECT * FROM experiments WHERE experiment_id = ?",
            (experiment_id,),
        )
        if exp.empty:
            logger.error("Experiment '%s' not found", experiment_id)
            return

        logger.info("=== Evaluating experiment: %s ===", experiment_id)
        logger.info("  Notes: %s", exp.iloc[0].get("notes", ""))

        # Get forecasts
        forecasts = query_df(
            conn,
            "SELECT * FROM forecasts WHERE experiment_id = ?",
            (experiment_id,),
        )
        if forecasts.empty:
            logger.error("No forecasts found for experiment '%s'", experiment_id)
            return

        if tickers:
            forecasts = forecasts[forecasts["ticker"].isin(tickers)]

        # Get CV fold for this experiment
        cv_fold = exp.iloc[0].get("cv_fold")
        if cv_fold is None:
            cv_fold = 0  # default

        # Compute metrics per ticker
        results = []
        for ticker in sorted(forecasts["ticker"].unique()):
            ticker_preds = forecasts[forecasts["ticker"] == ticker]
            metrics = compute_metrics(ticker_preds)

            results.append({"ticker": ticker, **metrics})

            # Write to evaluations table
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

        # Also compute aggregate across all tickers
        all_metrics = compute_metrics(forecasts)
        results.append({"ticker": "ALL", **all_metrics})
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

        # Trial-aware reporting
        n_trials = conn.execute(
            "SELECT COUNT(*) FROM experiments WHERE status='completed'"
        ).fetchone()[0]

        # Print results
        results_df = pd.DataFrame(results)
        print("\n" + "=" * 70)
        print(f"  Experiment: {experiment_id}")
        print(f"  Notes: {exp.iloc[0].get('notes', '')}")
        if n_trials > 1:
            print(
                f"  ⚠ Trial {n_trials} of {n_trials} — multiple comparisons increase false positive risk"
            )
        print("=" * 70)
        print(
            f"\n  {'Ticker':8s} {'MAE':>8s} {'RMSE':>8s} {'Dir Acc':>8s} {'Sharpe':>8s} {'N':>6s}"
        )
        print("  " + "-" * 52)
        for _, row in results_df.iterrows():
            mae_str = f"{row['mae']:.4f}" if row["mae"] is not None else "N/A"
            rmse_str = f"{row['rmse']:.4f}" if row["rmse"] is not None else "N/A"
            da_str = (
                f"{row['directional_acc']:.1%}"
                if row["directional_acc"] is not None
                else "N/A"
            )
            sh_str = f"{row['sharpe']:.2f}" if row.get("sharpe") is not None else "N/A"
            print(
                f"  {row['ticker']:8s} {mae_str:>8s} {rmse_str:>8s} {da_str:>8s} {sh_str:>8s} {row['n_predictions']:>6d}"
            )
        print()

        return results_df


def compare_experiments(experiment_ids: list[str]):
    """Side-by-side comparison of multiple experiments."""
    from market.db import get_db, query_df

    with get_db() as conn:
        print("\n" + "=" * 80)
        print("  EXPERIMENT COMPARISON")
        print("=" * 80)

        # Get experiment info
        for eid in experiment_ids:
            exp = query_df(
                conn, "SELECT * FROM experiments WHERE experiment_id = ?", (eid,)
            )
            if exp.empty:
                print(f"\n  {eid}: NOT FOUND")
                continue
            print(f"\n  {eid}: {exp.iloc[0].get('notes', 'no notes')}")

        # Get evaluations for all experiments
        placeholders = ",".join("?" * len(experiment_ids))
        evals = query_df(
            conn,
            f"""SELECT experiment_id, ticker, directional_acc, mae, n_predictions
                FROM evaluations
                WHERE experiment_id IN ({placeholders})
                ORDER BY ticker, experiment_id""",
            tuple(experiment_ids),
        )

        if evals.empty:
            print("\n  No evaluations found. Run evaluate first.")
            return

        # Pivot to compare
        for ticker in sorted(evals["ticker"].unique()):
            t_evals = evals[evals["ticker"] == ticker]
            print(f"\n  {ticker}:")
            print(f"    {'Experiment':30s} {'Dir Acc':>8s} {'MAE':>8s} {'N':>6s}")
            print("    " + "-" * 54)
            for _, row in t_evals.iterrows():
                da_str = (
                    f"{row['directional_acc']:.1%}"
                    if row["directional_acc"] is not None
                    else "N/A"
                )
                mae_str = f"{row['mae']:.4f}" if row["mae"] is not None else "N/A"
                print(
                    f"    {row['experiment_id']:30s} {da_str:>8s} {mae_str:>8s} {int(row['n_predictions']):>6d}"
                )

        print()


def list_experiments():
    """List all experiments in the DB."""
    from market.db import get_db, query_df

    with get_db() as conn:
        exps = query_df(
            conn,
            """SELECT e.experiment_id, e.status, e.notes, e.checkpoint_path,
                      COUNT(f.predicted) as n_forecasts
               FROM experiments e
               LEFT JOIN forecasts f ON e.experiment_id = f.experiment_id
               GROUP BY e.experiment_id
               ORDER BY e.created_at DESC""",
        )

        if exps.empty:
            print("\nNo experiments found.")
            return

        print("\n  Experiments:")
        print(f"  {'ID':30s} {'Status':10s} {'Forecasts':>10s} {'Notes'}")
        print("  " + "-" * 80)
        for _, row in exps.iterrows():
            ckpt = " (fine-tuned)" if row.get("checkpoint_path") else ""
            print(
                f"  {row['experiment_id']:30s} {row['status']:10s} "
                f"{int(row['n_forecasts']):>10d} {row.get('notes', '')}{ckpt}"
            )
        print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate forecasts")
    parser.add_argument("--experiment-id", type=str, help="Experiment to evaluate")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers to evaluate")
    parser.add_argument("--compare", nargs="+", help="Compare multiple experiments")
    parser.add_argument("--list", action="store_true", help="List all experiments")
    args = parser.parse_args()

    if args.list:
        list_experiments()
    elif args.compare:
        compare_experiments(args.compare)
    elif args.experiment_id:
        evaluate_experiment(args.experiment_id, tickers=args.tickers)
    else:
        # Default: evaluate most recent experiment
        from market.db import get_db, query_df

        with get_db() as conn:
            latest = query_df(
                conn,
                "SELECT experiment_id FROM experiments ORDER BY created_at DESC LIMIT 1",
            )
            if latest.empty:
                print("No experiments found. Run forecast first.")
            else:
                evaluate_experiment(
                    latest.iloc[0]["experiment_id"], tickers=args.tickers
                )


if __name__ == "__main__":
    main()
