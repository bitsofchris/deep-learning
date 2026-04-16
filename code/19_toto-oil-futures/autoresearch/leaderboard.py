"""Leaderboard — view experiment history and track autoresearch progress.

Usage:
    python -m autoresearch.leaderboard              # ranked by directional accuracy
    python -m autoresearch.leaderboard --by mae     # ranked by MAE (ascending)
    python -m autoresearch.leaderboard --by sharpe  # ranked by Sharpe (descending)
    python -m autoresearch.leaderboard --detail ar_finetune_abc123  # show one experiment
    python -m autoresearch.leaderboard --diff ar_exp1 ar_exp2       # diff two configs
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def show_leaderboard(sort_by: str = "directional_acc"):
    """Show ranked experiment results."""
    from market.db import get_db, query_df

    with get_db() as conn:
        # Get aggregate metrics (ticker='ALL')
        evals = query_df(
            conn,
            """
            SELECT
                e.experiment_id,
                e.created_at,
                e.status,
                e.notes,
                e.val_loss,
                ev.directional_acc,
                ev.mae,
                ev.rmse,
                ev.sharpe,
                ev.n_predictions
            FROM experiments e
            LEFT JOIN evaluations ev
                ON e.experiment_id = ev.experiment_id AND ev.ticker = 'ALL'
            ORDER BY e.created_at DESC
        """,
        )

        if evals.empty:
            print("\n  No experiments yet. Run your first experiment:")
            print("  python -m autoresearch.train --mode zeroshot\n")
            return

        # Sort by metric
        ascending = sort_by == "mae"  # lower is better for MAE
        ranked = evals.dropna(subset=[sort_by]).sort_values(
            sort_by, ascending=ascending
        )

        # Also show incomplete/failed experiments
        incomplete = evals[evals[sort_by].isna()]

        # Header
        n_completed = len(ranked)
        n_total = len(evals)
        best_da = ranked["directional_acc"].max() if not ranked.empty else None

        print(f"\n{'=' * 80}")
        print(f"  AUTORESEARCH LEADERBOARD")
        print(f"  {n_completed} completed / {n_total} total experiments")
        if best_da is not None:
            print(f"  Best directional accuracy: {best_da:.1%}")
        print(f"  Sorted by: {sort_by} ({'ascending' if ascending else 'descending'})")
        print(f"{'=' * 80}")

        # Ranked table
        if not ranked.empty:
            print(
                f"\n  {'Rank':>4s}  {'Experiment':30s} {'Dir Acc':>8s} {'MAE':>10s} "
                f"{'Sharpe':>7s} {'N':>7s} {'Status':>10s}"
            )
            print("  " + "-" * 82)

            for i, (_, row) in enumerate(ranked.iterrows()):
                rank = i + 1
                da = (
                    f"{row['directional_acc']:.1%}"
                    if pd.notna(row["directional_acc"])
                    else "N/A"
                )
                mae = f"{row['mae']:.6f}" if pd.notna(row["mae"]) else "N/A"
                sharpe = f"{row['sharpe']:.2f}" if pd.notna(row["sharpe"]) else "N/A"
                n = (
                    f"{int(row['n_predictions']):,}"
                    if pd.notna(row["n_predictions"])
                    else "N/A"
                )
                marker = " <-- best" if rank == 1 else ""

                print(
                    f"  {rank:>4d}  {row['experiment_id']:30s} {da:>8s} {mae:>10s} "
                    f"{sharpe:>7s} {n:>7s} {row['status']:>10s}{marker}"
                )

        # Show incomplete experiments
        if not incomplete.empty:
            print(f"\n  Incomplete/failed:")
            for _, row in incomplete.iterrows():
                print(
                    f"    {row['experiment_id']:30s} {row['status']:>10s}  {row.get('notes', '')}"
                )

        print()


def show_detail(experiment_id: str):
    """Show detailed results for one experiment."""
    from market.db import get_db, query_df

    with get_db() as conn:
        # Experiment info
        exp = query_df(
            conn, "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
        )
        if exp.empty:
            print(f"\n  Experiment '{experiment_id}' not found.\n")
            return

        row = exp.iloc[0]
        config = json.loads(row["config_json"]) if row.get("config_json") else {}

        print(f"\n{'=' * 70}")
        print(f"  Experiment: {experiment_id}")
        print(f"  Created:    {row.get('created_at', 'N/A')}")
        print(f"  Status:     {row.get('status', 'N/A')}")
        print(f"  Notes:      {row.get('notes', 'N/A')}")
        print(f"  Val Loss:   {row.get('val_loss', 'N/A')}")
        print(f"  Checkpoint: {row.get('checkpoint_path', 'N/A')}")
        print(f"{'=' * 70}")

        # Config summary
        print(f"\n  Config:")
        for key in [
            "target",
            "mode",
            "learning_rate",
            "context_factor",
            "prediction_length",
            "batch_size",
            "fold",
        ]:
            if key in config:
                print(f"    {key}: {config[key]}")

        # Per-ticker metrics
        evals = query_df(
            conn,
            """
            SELECT ticker, directional_acc, mae, rmse, sharpe, n_predictions
            FROM evaluations WHERE experiment_id = ?
            ORDER BY ticker
        """,
            (experiment_id,),
        )

        if not evals.empty:
            print(
                f"\n  {'Ticker':8s} {'Dir Acc':>8s} {'MAE':>10s} {'RMSE':>10s} "
                f"{'Sharpe':>7s} {'N':>7s}"
            )
            print("  " + "-" * 55)
            for _, r in evals.iterrows():
                da = (
                    f"{r['directional_acc']:.1%}"
                    if pd.notna(r["directional_acc"])
                    else "N/A"
                )
                mae = f"{r['mae']:.6f}" if pd.notna(r["mae"]) else "N/A"
                rmse = f"{r['rmse']:.6f}" if pd.notna(r["rmse"]) else "N/A"
                sharpe = f"{r['sharpe']:.2f}" if pd.notna(r["sharpe"]) else "N/A"
                n = (
                    f"{int(r['n_predictions']):,}"
                    if pd.notna(r["n_predictions"])
                    else "N/A"
                )
                print(
                    f"  {r['ticker']:8s} {da:>8s} {mae:>10s} {rmse:>10s} {sharpe:>7s} {n:>7s}"
                )

        print()


def show_diff(exp_id_1: str, exp_id_2: str):
    """Show config differences between two experiments."""
    from market.db import get_db, query_df

    with get_db() as conn:
        exp1 = query_df(
            conn,
            "SELECT config_json FROM experiments WHERE experiment_id = ?",
            (exp_id_1,),
        )
        exp2 = query_df(
            conn,
            "SELECT config_json FROM experiments WHERE experiment_id = ?",
            (exp_id_2,),
        )

        if exp1.empty or exp2.empty:
            print(f"\n  One or both experiments not found.\n")
            return

        config1 = json.loads(exp1.iloc[0]["config_json"])
        config2 = json.loads(exp2.iloc[0]["config_json"])

        print(f"\n{'=' * 70}")
        print(f"  CONFIG DIFF: {exp_id_1} vs {exp_id_2}")
        print(f"{'=' * 70}")

        # Flatten configs for comparison
        def flatten(d, prefix=""):
            items = {}
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    items.update(flatten(v, key))
                else:
                    items[key] = v
            return items

        flat1 = flatten(config1)
        flat2 = flatten(config2)
        all_keys = sorted(set(flat1.keys()) | set(flat2.keys()))

        diffs = []
        for key in all_keys:
            v1 = flat1.get(key)
            v2 = flat2.get(key)
            if v1 != v2:
                diffs.append((key, v1, v2))

        if diffs:
            print(f"\n  {'Parameter':35s} {exp_id_1[:20]:>20s} {exp_id_2[:20]:>20s}")
            print("  " + "-" * 77)
            for key, v1, v2 in diffs:
                print(f"  {key:35s} {str(v1):>20s} {str(v2):>20s}")
        else:
            print("\n  Configs are identical.")

        # Also show metric comparison
        evals = query_df(
            conn,
            """
            SELECT experiment_id, directional_acc, mae, sharpe
            FROM evaluations
            WHERE experiment_id IN (?, ?) AND ticker = 'ALL'
        """,
            (exp_id_1, exp_id_2),
        )

        if not evals.empty:
            print(f"\n  {'Metric':20s} {exp_id_1[:20]:>20s} {exp_id_2[:20]:>20s}")
            print("  " + "-" * 62)
            for metric in ["directional_acc", "mae", "sharpe"]:
                v1 = evals.loc[evals["experiment_id"] == exp_id_1, metric]
                v2 = evals.loc[evals["experiment_id"] == exp_id_2, metric]
                s1 = (
                    f"{v1.iloc[0]:.4f}"
                    if not v1.empty and pd.notna(v1.iloc[0])
                    else "N/A"
                )
                s2 = (
                    f"{v2.iloc[0]:.4f}"
                    if not v2.empty and pd.notna(v2.iloc[0])
                    else "N/A"
                )
                print(f"  {metric:20s} {s1:>20s} {s2:>20s}")

        print()


def get_best_metric(metric: str = "directional_acc") -> float | None:
    """Get the current best value for a metric. Used by the agent loop."""
    from market.db import get_db, query_df

    with get_db() as conn:
        result = query_df(
            conn,
            f"""
            SELECT MAX({metric}) as best
            FROM evaluations
            WHERE ticker = 'ALL'
        """,
        )
        if result.empty or pd.isna(result.iloc[0]["best"]):
            return None
        return float(result.iloc[0]["best"])


def main():
    parser = argparse.ArgumentParser(description="Autoresearch experiment leaderboard")
    parser.add_argument(
        "--by",
        default="directional_acc",
        choices=["directional_acc", "mae", "sharpe"],
        help="Sort metric",
    )
    parser.add_argument("--detail", type=str, help="Show detail for one experiment")
    parser.add_argument(
        "--diff", nargs=2, metavar=("EXP1", "EXP2"), help="Diff two experiment configs"
    )
    args = parser.parse_args()

    if args.detail:
        show_detail(args.detail)
    elif args.diff:
        show_diff(args.diff[0], args.diff[1])
    else:
        show_leaderboard(sort_by=args.by)


if __name__ == "__main__":
    main()
