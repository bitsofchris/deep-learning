import time
import yaml
import argparse
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime

from data_utils import get_data_loaders
from model import LeNet5


def _get_all_experiments(experiment_file):
    with open(experiment_file, "r") as f:
        config = yaml.safe_load(f)

    # Extract the 'experiments' list
    experiment_groups = config["experiments"]
    all_experiments = []

    for group in experiment_groups:
        method = group["pruning_method"]
        sizes = group.get("sizes", [])

        if method == "random":
            # For each size, define a single experiment (random)
            for s in sizes:
                # e.g., build a dict
                exp = {
                    "pruning_method": "random",
                    "dataset_size": s,
                    # possibly set runs=3, num_epochs=5, etc.
                }
                all_experiments.append(exp)
        elif method == "cluster":
            # We'll have ks, selection_strategies, sizes, use_pca
            use_pca = group["use_pca"]
            ks = group["ks"]
            selection_strategies = group["selection_strategies"]

            for s in sizes:
                for k in ks:
                    for strategy in selection_strategies:
                        exp = {
                            "pruning_method": "cluster",
                            "dataset_size": s,
                            "k": k,
                            "selection_strategy": strategy,
                            "apply_pca": use_pca,
                        }
                        all_experiments.append(exp)
    print(f"Total expanded experiments = {len(all_experiments)}")
    return all_experiments


def train_and_evaluate(
    pruning_method="none",
    dataset_size=600,
    selection_strategy=None,
    apply_pca=False,
    k=1,
    n_closest=1,
    n_furthest=1,
    batch_size=64,
    learning_rate=1e-3,
    num_epochs=5,
    runs=5,
):
    """
    Runs multiple training loops for a given experiment config.

    Returns:
        A list of dicts, one per run, each containing:
          - "run_id": int
          - "accuracy": float
          - "time": float
          - Other key parameters from the experiment
    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    run_data = []  # to store accuracy/time for each run

    for run_id in range(runs):
        print(
            f"=== Run {run_id+1}/{runs} for method={pruning_method}, "
            f"size={dataset_size}, strategy={selection_strategy} ==="
        )
        start_time = time.time()

        # 1) Build pruning_kwargs
        if pruning_method == "random":
            fraction = dataset_size / 60000.0  # e.g., if full dataset ~60k
            pruning_kwargs = {"percentage": fraction}
        elif pruning_method == "cluster":
            pruning_kwargs = {
                "target_size": dataset_size,
                "selection_strategy": selection_strategy,
                "apply_pca": apply_pca,
                "k": k,
                "n_closest": n_closest,
                "n_furthest": n_furthest,
            }
        else:
            pruning_kwargs = {}

        # 2) Get DataLoaders
        train_loader, test_loader = get_data_loaders(
            batch_size=batch_size,
            pruning_method=pruning_method,
            pruning_kwargs=pruning_kwargs,
        )

        # 3) Define model, loss, optimizer
        model = LeNet5(num_classes=10).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # 4) Train
        model.train()
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 5) Evaluate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        run_time = time.time() - start_time
        acc = 100.0 * correct / total
        print(f"--> run={run_id+1}, accuracy={acc:.2f}%, time={run_time:.2f}s")

        # Collect run-level data
        run_data.append(
            {
                "run_id": run_id + 1,
                "accuracy": acc,
                "time": run_time,
                # Optional: also store experiment-level context in each run
                "pruning_method": pruning_method,
                "dataset_size": dataset_size,
                "selection_strategy": selection_strategy,
                "apply_pca": apply_pca,
                "k": k,
                "n_closest": n_closest,
                "n_furthest": n_furthest,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
            }
        )

    return run_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments for CNN LeNet-5")
    parser.add_argument(
        "--experiments_file",
        type=str,
        default="experiments_small.yaml",
        help="Path to the experiments configuration file",
    )
    args = parser.parse_args()

    all_results = []
    all_experiments = _get_all_experiments(args.experiments_file)

    for exp_idx, config in enumerate(all_experiments):
        print(
            f"------------------ Experiment {exp_idx+1}/{len(all_experiments)} ------------------"
        )
        runs_output = train_and_evaluate(**config)

        # runs_output is a list of dicts, one per run
        for run_dict in runs_output:
            run_dict["experiment_id"] = exp_idx + 1
            all_results.append(run_dict)

    # Convert results to DataFrame and save to CSV
    df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"results/experiment_results_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)

    # Aggregate results
    group_cols = [
        "pruning_method",
        "dataset_size",
        "selection_strategy",
        "apply_pca",
        "k",
        "n_closest",
        "n_furthest",
    ]
    agg_results = (
        df.groupby(group_cols)
        .agg(
            median_accuracy=("accuracy", "median"),
            median_time=("time", "median"),
            count_runs=("run_id", "count"),
        )
        .reset_index()
    )
    print("\n=== Aggregated Median Results ===")
    print(agg_results)
