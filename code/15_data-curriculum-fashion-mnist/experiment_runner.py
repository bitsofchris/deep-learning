import os
import time
import yaml
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from typing import Dict, List, Any
import logging

from model import LeNet5, get_device
from data_ordering import get_data_loaders
from results_manager import ResultsManager


class CurriculumExperimentRunner:
    """Runs curriculum learning experiments based on YAML configuration."""

    def __init__(
        self, experiment_name: str = "curriculum_learning", results_dir: str = "results"
    ):
        """Initialize experiment runner.

        Args:
            experiment_name: Name for this set of experiments
            results_dir: Directory to save results
        """
        self.experiment_name = experiment_name
        self.results_dir = results_dir
        self.device = get_device()

        # Initialize results manager
        self.results_manager = ResultsManager(
            experiment_name=experiment_name, base_dir=results_dir
        )

        self.logger = logging.getLogger(__name__)

    def load_experiment_config(self, config_file: str) -> List[Dict]:
        """Load and expand experiment configuration from YAML.

        Args:
            config_file: Path to YAML configuration file (relative to script location)

        Returns:
            List of expanded experiment configurations
        """
        # Make path relative to the script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_file)

        # If file doesn't exist in script dir, try current working directory
        if not os.path.exists(config_path):
            config_path = config_file

        self.logger.info(f"Loading experiment config from: {config_path}")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        experiment_groups = config["experiments"]
        all_experiments = []

        for group in experiment_groups:
            ordering_method = group["ordering_method"]

            if ordering_method == "random":
                # Random baseline experiments
                runs = group.get("runs", 3)
                num_epochs_list = group.get("num_epochs", [10])
                batch_sizes = group.get("batch_sizes", [64])
                learning_rates = group.get("learning_rates", [0.001])
                random_seeds = group.get("random_seeds", [42])  # Default single seed

                for epochs in num_epochs_list:
                    for batch_size in batch_sizes:
                        for lr in learning_rates:
                            for seed in random_seeds:
                                exp = {
                                    "ordering_method": "random",
                                    "runs": runs,
                                    "num_epochs": epochs,
                                    "batch_size": batch_size,
                                    "learning_rate": lr,
                                    "curriculum_params": None,
                                    "random_seed": seed,
                                }
                                all_experiments.append(exp)

            elif ordering_method == "curriculum":
                # Curriculum learning experiments
                runs = group.get("runs", 3)
                num_epochs_list = group.get("num_epochs", [10])
                batch_sizes = group.get("batch_sizes", [64])
                learning_rates = group.get("learning_rates", [0.001])
                strategies = group.get("strategies", ["medoid_first"])
                n_clusters_list = group.get("n_clusters", [50])
                use_pca_list = group.get("use_pca", [True])
                pca_components_list = group.get("pca_components", [50])

                for epochs in num_epochs_list:
                    for batch_size in batch_sizes:
                        for lr in learning_rates:
                            for strategy in strategies:
                                for n_clusters in n_clusters_list:
                                    for use_pca in use_pca_list:
                                        for pca_components in pca_components_list:
                                            exp = {
                                                "ordering_method": "curriculum",
                                                "runs": runs,
                                                "num_epochs": epochs,
                                                "batch_size": batch_size,
                                                "learning_rate": lr,
                                                "curriculum_params": {
                                                    "strategy": strategy,
                                                    "n_clusters": n_clusters,
                                                    "use_pca": use_pca,
                                                    "pca_components": pca_components,
                                                },
                                            }
                                            all_experiments.append(exp)

        self.logger.info(f"Loaded {len(all_experiments)} experiment configurations")
        return all_experiments

    def _precompute_curriculum_ordering(self, config: Dict) -> Dict:
        """Pre-compute curriculum ordering to reuse across multiple runs.

        Args:
            config: Experiment configuration dictionary

        Returns:
            Dictionary with curriculum data (ordered_indices, cluster_info, etc.)
        """
        from data_ordering import DataOrdering

        # Create data orderer
        data_orderer = DataOrdering(
            batch_size=config["batch_size"], num_workers=2, random_state=42
        )

        # Load raw datasets
        train_dataset, _ = data_orderer.get_fashion_mnist_datasets("data")

        # Extract all training images and labels for clustering
        self.logger.info("Extracting training data for curriculum ordering...")
        all_images = []
        all_labels = []

        # Use a temporary loader to extract all data efficiently
        temp_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1000, shuffle=False
        )
        for batch_images, batch_labels in temp_loader:
            all_images.append(batch_images)
            all_labels.append(batch_labels)

        all_images = torch.cat(all_images, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        self.logger.info(f"Extracted {len(all_images)} training samples")

        # Get curriculum ordering
        curriculum_params = config.get("curriculum_params", {})
        ordered_indices, cluster_info = data_orderer.get_curriculum_ordering(
            images=all_images, labels=all_labels, **curriculum_params
        )

        return {
            "ordered_indices": ordered_indices,
            "cluster_info": cluster_info,
            "train_dataset": train_dataset,
            "curriculum_params": curriculum_params,
        }

    def run_single_experiment(
        self,
        config: Dict,
        experiment_id: int,
        run_id: int,
        curriculum_data: Dict = None,
    ) -> Dict:
        """Run a single experiment configuration.

        Args:
            config: Experiment configuration dictionary
            experiment_id: Unique experiment identifier
            run_id: Run number for this experiment

        Returns:
            Dictionary with experiment results
        """
        self.logger.info(f"Starting experiment {experiment_id}, run {run_id}")
        self.logger.info(f"Config: {config}")

        # Set random seeds for reproducibility
        random_seed = config.get("random_seed", 42)
        torch.manual_seed(random_seed + run_id)  # Different seed per run
        np.random.seed(random_seed + run_id)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed + run_id)

        start_time = time.time()

        # Create data loaders
        if config["ordering_method"] == "random":
            # Random baseline - normal data loading
            train_loader, test_loader, cluster_info = get_data_loaders(
                ordering_strategy="random",
                curriculum_params=None,
                batch_size=config["batch_size"],
                num_workers=2,
                data_dir="data",
            )

        elif config["ordering_method"] == "curriculum" and curriculum_data is not None:
            # Use pre-computed curriculum ordering
            from data_ordering import OrderedDataset

            # Create ordered dataset using pre-computed indices
            ordered_dataset = OrderedDataset(
                curriculum_data["train_dataset"], curriculum_data["ordered_indices"]
            )

            # Create train loader with curriculum ordering
            train_loader = torch.utils.data.DataLoader(
                ordered_dataset,
                batch_size=config["batch_size"],
                shuffle=False,  # Maintain curriculum order
                num_workers=2,
            )

            # Create test loader normally
            _, test_dataset = curriculum_data["train_dataset"], None
            from torchvision import datasets, transforms

            transform = transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.2860,), (0.3530,)),
                ]
            )

            test_dataset = datasets.FashionMNIST(
                root="data", train=False, download=True, transform=transform
            )

            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=config["batch_size"],
                shuffle=False,
                num_workers=2,
            )

            cluster_info = curriculum_data["cluster_info"]

        else:
            # Fallback to normal data loading
            train_loader, test_loader, cluster_info = get_data_loaders(
                ordering_strategy=config["ordering_method"],
                curriculum_params=config.get("curriculum_params"),
                batch_size=config["batch_size"],
                num_workers=2,
                data_dir="data",
            )

        self.logger.info(
            f"Loaded data: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test"
        )

        # Initialize model, loss, optimizer
        model = LeNet5(num_classes=10).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

        # Training loop
        best_accuracy = 0.0
        convergence_epoch = None

        for epoch in range(1, config["num_epochs"] + 1):
            # Training
            train_metrics = self._train_epoch(model, train_loader, optimizer, criterion)

            # Validation
            val_metrics = self._validate(model, test_loader, criterion)

            # Update learning rate
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            # Track best accuracy
            if val_metrics["accuracy"] > best_accuracy:
                best_accuracy = val_metrics["accuracy"]

            # Check for convergence (simplified)
            if convergence_epoch is None and epoch > 5:
                if val_metrics["accuracy"] >= best_accuracy - 0.5:
                    convergence_epoch = epoch

            # Log training progress
            if epoch % 5 == 0 or epoch == config["num_epochs"]:
                self.logger.info(
                    f"Epoch {epoch}/{config['num_epochs']}: "
                    f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                    f"Val Acc: {val_metrics['accuracy']:.2f}%, "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}"
                )

            # Log per-epoch metrics
            self.results_manager.log_training_epoch(
                experiment_id, run_id, epoch, train_metrics, val_metrics, current_lr
            )

        total_time = time.time() - start_time

        # Final evaluation
        final_val_metrics = self._validate(model, test_loader, criterion)

        # Prepare results
        results = {
            "final_accuracy": final_val_metrics["accuracy"],
            "best_accuracy": best_accuracy,
            "final_loss": final_val_metrics["loss"],
            "training_time": total_time,
            "convergence_epoch": convergence_epoch,
        }

        # Save final model
        experiment_config = {
            "experiment_id": experiment_id,
            "run_id": run_id,
            "ordering_method": config["ordering_method"],
            "curriculum_params": config.get("curriculum_params"),
            "num_epochs": config["num_epochs"],
            "batch_size": config["batch_size"],
            "learning_rate": config["learning_rate"],
        }

        checkpoint_path = self.results_manager.save_final_model(
            model, optimizer, results, experiment_id, run_id, experiment_config
        )
        results["model_checkpoint"] = checkpoint_path

        # Save cluster info for curriculum experiments
        if cluster_info and config["ordering_method"] == "curriculum":
            # Save visualization of cluster examples
            if curriculum_data is not None:
                # Get specific images for medoids and edges from the full dataset
                base_dataset = curriculum_data["train_dataset"]
                medoid_indices = cluster_info.get("medoid_indices", [])
                edge_indices = cluster_info.get("edge_indices", [])

                # Get first 10 medoids and edges for visualization
                viz_indices = list(medoid_indices[:10]) + list(edge_indices[:10])
                viz_images = []
                viz_labels = []

                for idx in viz_indices:
                    if idx < len(base_dataset):
                        img, label = base_dataset[idx]
                        viz_images.append(img.unsqueeze(0))
                        viz_labels.append(label)

                if viz_images:
                    viz_images = torch.cat(viz_images, dim=0)
                    viz_labels = torch.tensor(viz_labels)

                    # Update cluster_info with visualization-specific indices
                    viz_cluster_info = cluster_info.copy()
                    viz_cluster_info["medoid_indices"] = list(
                        range(min(10, len(medoid_indices)))
                    )
                    viz_cluster_info["edge_indices"] = list(
                        range(
                            min(10, len(medoid_indices)),
                            min(20, len(medoid_indices) + len(edge_indices)),
                        )
                    )

                    strategy_name = curriculum_data["curriculum_params"]["strategy"]
                    self.results_manager.save_cluster_visualizations(
                        viz_images, viz_labels, viz_cluster_info, strategy_name
                    )

        self.logger.info(
            f"Experiment {experiment_id}, run {run_id} completed: "
            f"Final accuracy: {final_val_metrics['accuracy']:.2f}%"
        )

        return results

    def _train_epoch(self, model, train_loader, optimizer, criterion):
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for data, target in train_loader:
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        return {"loss": avg_loss, "accuracy": accuracy}

    def _validate(self, model, test_loader, criterion):
        """Validate the model."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        avg_loss = total_loss / len(test_loader)
        accuracy = 100.0 * correct / total

        return {"loss": avg_loss, "accuracy": accuracy}

    def run_experiments(self, config_file: str) -> pd.DataFrame:
        """Run all experiments from configuration file.

        Args:
            config_file: Path to YAML configuration file

        Returns:
            DataFrame with all experiment results
        """
        self.logger.info(f"Starting experiment suite from {config_file}")

        # Load experiments
        all_experiments = self.load_experiment_config(config_file)

        all_results = []

        # Cache for curriculum orderings (to avoid re-clustering)
        curriculum_cache = {}

        # Run each experiment configuration
        for exp_idx, config in enumerate(all_experiments):
            experiment_id = exp_idx + 1

            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"EXPERIMENT {experiment_id}/{len(all_experiments)}")
            self.logger.info(f"{'='*80}")

            # Pre-compute curriculum ordering if needed (once per experiment config)
            curriculum_data = None
            if config["ordering_method"] == "curriculum":
                # Create a cache key based on curriculum parameters
                curriculum_params = config.get("curriculum_params", {})
                cache_key = (
                    curriculum_params.get("strategy", "medoid_first"),
                    curriculum_params.get("n_clusters", 50),
                    curriculum_params.get("use_pca", True),
                    curriculum_params.get("pca_components", 50),
                )

                if cache_key not in curriculum_cache:
                    self.logger.info(
                        "Computing curriculum ordering (will be reused across runs)..."
                    )
                    curriculum_data = self._precompute_curriculum_ordering(config)
                    curriculum_cache[cache_key] = curriculum_data
                    self.logger.info(f"Cached curriculum ordering for key: {cache_key}")
                else:
                    curriculum_data = curriculum_cache[cache_key]
                    self.logger.info(
                        f"Reusing cached curriculum ordering for key: {cache_key}"
                    )

            # Run multiple runs for statistical significance
            for run in range(1, config["runs"] + 1):
                self.logger.info(f"\n--- Run {run}/{config['runs']} ---")

                # Run single experiment with pre-computed curriculum
                results = self.run_single_experiment(
                    config, experiment_id, run, curriculum_data
                )

                # Prepare experiment record
                experiment_record = {
                    "experiment_id": experiment_id,
                    "run_id": run,
                    "ordering_method": config["ordering_method"],
                    "num_epochs": config["num_epochs"],
                    "batch_size": config["batch_size"],
                    "learning_rate": config["learning_rate"],
                    "random_seed": config.get("random_seed", 42),
                    **results,
                }

                # Add curriculum-specific parameters
                if config.get("curriculum_params"):
                    curriculum_params = config["curriculum_params"]
                    experiment_record.update(
                        {
                            "curriculum_strategy": curriculum_params.get("strategy"),
                            "n_clusters": curriculum_params.get("n_clusters"),
                            "use_pca": curriculum_params.get("use_pca"),
                            "pca_components": curriculum_params.get("pca_components"),
                        }
                    )
                else:
                    experiment_record.update(
                        {
                            "curriculum_strategy": None,
                            "n_clusters": None,
                            "use_pca": None,
                            "pca_components": None,
                        }
                    )

                # Log results
                self.results_manager.log_experiment_result(experiment_record, results)
                all_results.append(experiment_record)

        # Create results DataFrame
        results_df = pd.DataFrame(all_results)

        # Generate summary analysis
        self._generate_experiment_summary(results_df)

        # Generate plots and final report
        self.results_manager.plot_training_curves()
        report_path = self.results_manager.generate_experiment_report()

        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"EXPERIMENT SUITE COMPLETED")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Results saved to: {self.results_manager.results_dir}")
        self.logger.info(f"Report: {report_path}")

        return results_df

    def _generate_experiment_summary(self, results_df: pd.DataFrame):
        """Generate and log experiment summary."""
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"EXPERIMENT SUMMARY")
        self.logger.info(f"{'='*50}")

        # Group by experimental conditions (including random seed for sensitivity analysis)
        if "random_seed" in results_df.columns:
            # Group by both method and random seed for detailed analysis
            summary = (
                results_df.groupby(
                    ["ordering_method", "curriculum_strategy", "random_seed"]
                )
                .agg(
                    {
                        "final_accuracy": ["mean", "std", "count", "min", "max"],
                        "training_time": ["mean", "std"],
                        "convergence_epoch": ["mean", "std"],
                    }
                )
                .round(3)
            )

            # Also create overall summary across all seeds
            overall_summary = (
                results_df.groupby(["ordering_method", "curriculum_strategy"])
                .agg(
                    {
                        "final_accuracy": ["mean", "std", "count", "min", "max"],
                        "training_time": ["mean", "std"],
                        "convergence_epoch": ["mean", "std"],
                    }
                )
                .round(3)
            )
        else:
            summary = (
                results_df.groupby(["ordering_method", "curriculum_strategy"])
                .agg(
                    {
                        "final_accuracy": ["mean", "std", "count"],
                        "training_time": ["mean", "std"],
                        "convergence_epoch": ["mean", "std"],
                    }
                )
                .round(3)
            )
            overall_summary = summary

        self.logger.info("Per-Seed Summary Statistics:")
        self.logger.info(f"\n{summary}")

        if "random_seed" in results_df.columns:
            self.logger.info("\nOverall Summary (across all seeds):")
            self.logger.info(f"\n{overall_summary}")

            # Random seed sensitivity analysis
            if results_df["ordering_method"].eq("random").any():
                random_results = results_df[results_df["ordering_method"] == "random"]
                seed_sensitivity = (
                    random_results.groupby("random_seed")["final_accuracy"]
                    .agg(["mean", "std", "min", "max"])
                    .round(3)
                )

                self.logger.info("\n" + "=" * 50)
                self.logger.info("RANDOM SEED SENSITIVITY ANALYSIS")
                self.logger.info("=" * 50)
                self.logger.info(f"\n{seed_sensitivity}")

                # Calculate overall variance across seeds
                seed_means = random_results.groupby("random_seed")[
                    "final_accuracy"
                ].mean()
                seed_variance = seed_means.var()
                seed_range = seed_means.max() - seed_means.min()

                self.logger.info(f"\nRandom Seed Variance: {seed_variance:.6f}")
                self.logger.info(f"Random Seed Range: {seed_range:.3f}%")
                self.logger.info(
                    f"Coefficient of Variation: {seed_means.std()/seed_means.mean()*100:.2f}%"
                )

        # Compare curriculum vs random
        if (
            "random" in results_df["ordering_method"].values
            and "curriculum" in results_df["ordering_method"].values
        ):
            random_acc = results_df[results_df["ordering_method"] == "random"][
                "final_accuracy"
            ].mean()
            curriculum_acc = results_df[results_df["ordering_method"] == "curriculum"][
                "final_accuracy"
            ].mean()

            improvement = curriculum_acc - random_acc
            self.logger.info(f"\nCurriculum Learning Effect:")
            self.logger.info(f"Random baseline: {random_acc:.2f}%")
            self.logger.info(f"Curriculum: {curriculum_acc:.2f}%")
            self.logger.info(f"Improvement: {improvement:+.2f}%")


def main():
    """Main entry point for running experiments."""
    parser = argparse.ArgumentParser(description="Run curriculum learning experiments")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments.yaml",
        help="Path to experiment configuration file",
    )
    parser.add_argument(
        "--name", type=str, default="curriculum_learning", help="Experiment suite name"
    )
    args = parser.parse_args()

    # Initialize and run experiments
    runner = CurriculumExperimentRunner(experiment_name=args.name)
    results_df = runner.run_experiments(args.config)

    return results_df


if __name__ == "__main__":
    results = main()
