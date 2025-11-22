import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import logging

from model import LeNet5, get_device
from results_manager import ResultsManager


class BaselineTrainer:
    """Baseline trainer for Fashion-MNIST with random data ordering."""

    def __init__(self, batch_size=64, learning_rate=0.001, num_epochs=20):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = get_device()

        self.logger = logging.getLogger(__name__)

        # Initialize data transforms - resize to 32x32 for LeNet5 compatibility
        self.transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),  # Resize to match LeNet5 expected input
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,)),  # Fashion-MNIST statistics
            ]
        )

    def get_data_loaders(self):
        """Create Fashion-MNIST data loaders with random ordering."""
        # Download and create datasets
        train_dataset = datasets.FashionMNIST(
            root="data", train=True, download=True, transform=self.transform
        )

        test_dataset = datasets.FashionMNIST(
            root="data", train=False, download=True, transform=self.transform
        )

        # Create data loaders with shuffling for random ordering
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Random ordering
            num_workers=2,
        )

        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2
        )

        return train_loader, test_loader

    def train_epoch(self, model, train_loader, optimizer, criterion):
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
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

            if batch_idx % 200 == 0:
                self.logger.info(
                    f"Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.6f}"
                )

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        return {"loss": avg_loss, "accuracy": accuracy}

    def validate(self, model, test_loader, criterion):
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

    def train_and_evaluate(self, experiment_id=1, run_id=1, results_manager=None):
        """Complete training and evaluation cycle.

        Args:
            experiment_id: Unique experiment identifier
            run_id: Run number for this experiment
            results_manager: ResultsManager instance for logging

        Returns:
            Dict with training results
        """
        self.logger.info(
            f"Starting training - Experiment {experiment_id}, Run {run_id}"
        )
        self.logger.info(f"Device: {self.device}")

        start_time = time.time()

        # Get data loaders
        train_loader, test_loader = self.get_data_loaders()
        self.logger.info(
            f"Loaded Fashion-MNIST: {len(train_loader.dataset)} train, "
            f"{len(test_loader.dataset)} test"
        )

        # Initialize model, loss, optimizer
        model = LeNet5(num_classes=10).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

        # Training loop
        best_accuracy = 0.0
        convergence_epoch = None

        for epoch in range(1, self.num_epochs + 1):
            self.logger.info(f"Epoch {epoch}/{self.num_epochs}")

            # Train
            train_metrics = self.train_epoch(model, train_loader, optimizer, criterion)

            # Validate
            val_metrics = self.validate(model, test_loader, criterion)

            # Update learning rate
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            self.logger.info(
                f"Epoch {epoch}: "
                f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                f"Val Acc: {val_metrics['accuracy']:.2f}%, "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"LR: {current_lr:.6f}"
            )

            # Track best accuracy and convergence
            if val_metrics["accuracy"] > best_accuracy:
                best_accuracy = val_metrics["accuracy"]

            # Check for convergence (accuracy not improving for several epochs)
            if convergence_epoch is None and epoch > 5:
                recent_accuracies = [
                    val_metrics["accuracy"]
                ]  # In real implementation, would track history
                if len(recent_accuracies) >= 3 and all(
                    acc >= val_metrics["accuracy"] - 0.5 for acc in recent_accuracies
                ):
                    convergence_epoch = epoch

            # Log per-epoch metrics
            if results_manager:
                results_manager.log_training_epoch(
                    experiment_id, run_id, epoch, train_metrics, val_metrics, current_lr
                )

        total_time = time.time() - start_time

        # Final evaluation
        final_val_metrics = self.validate(model, test_loader, criterion)

        results = {
            "final_accuracy": final_val_metrics["accuracy"],
            "best_accuracy": best_accuracy,
            "final_loss": final_val_metrics["loss"],
            "training_time": total_time,
            "convergence_epoch": convergence_epoch,
            "model_checkpoint": None,
        }

        # Save final model checkpoint
        if results_manager:
            experiment_config = {
                "experiment_id": experiment_id,
                "run_id": run_id,
                "data_ordering": "random",
                "num_epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
            }
            checkpoint_path = results_manager.save_final_model(
                model, optimizer, results, experiment_id, run_id, experiment_config
            )
            results["model_checkpoint"] = checkpoint_path

        self.logger.info(
            f"Training completed in {total_time:.2f}s. "
            f"Final accuracy: {final_val_metrics['accuracy']:.2f}%"
        )

        return results


def run_baseline_experiment(
    num_runs=3, num_epochs=20, batch_size=64, learning_rate=0.001
):
    """Run baseline experiment with multiple runs for statistical significance.

    Args:
        num_runs: Number of independent runs
        num_epochs: Training epochs per run
        batch_size: Batch size
        learning_rate: Learning rate
    """
    # Initialize results manager
    results_manager = ResultsManager(experiment_name="fashion_mnist_baseline")

    all_results = []

    for run in range(1, num_runs + 1):
        print(f"\n{'='*50}")
        print(f"BASELINE RUN {run}/{num_runs}")
        print(f"{'='*50}")

        # Initialize trainer
        trainer = BaselineTrainer(
            batch_size=batch_size, learning_rate=learning_rate, num_epochs=num_epochs
        )

        # Run training
        experiment_config = {
            "experiment_id": 1,
            "run_id": run,
            "data_ordering": "random",
            "num_clusters": None,
            "curriculum_strategy": None,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
        }

        results = trainer.train_and_evaluate(
            experiment_id=1, run_id=run, results_manager=results_manager
        )

        # Log results
        results_manager.log_experiment_result(experiment_config, results)
        all_results.append(results)

    # Generate summary
    accuracies = [r["final_accuracy"] for r in all_results]
    times = [r["training_time"] for r in all_results]

    print(f"\n{'='*50}")
    print(f"BASELINE SUMMARY ({num_runs} runs)")
    print(f"{'='*50}")
    print(f"Accuracy: {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}%")
    print(f"Training Time: {np.mean(times):.1f}s ± {np.std(times):.1f}s")
    print(f"Best Run: {max(accuracies):.2f}%")

    # Generate plots and reports
    results_manager.plot_training_curves()
    report_path = results_manager.generate_experiment_report()

    print(f"Results saved to: {results_manager.results_dir}")
    print(f"Report: {report_path}")

    return all_results, results_manager


if __name__ == "__main__":
    import numpy as np

    # Run baseline experiment
    results, manager = run_baseline_experiment(
        num_runs=3, num_epochs=10, batch_size=32, learning_rate=0.005
    )
