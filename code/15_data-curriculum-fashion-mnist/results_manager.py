import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging

# Fashion-MNIST class names for visualization
FASHION_CLASSES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]


class ResultsManager:
    """Manages experiment results, logging, visualization, and model checkpoints."""
    
    def __init__(self, experiment_name="curriculum_learning", base_dir=None):
        self.experiment_name = experiment_name
        # Default to results directory within the current project
        if base_dir is None:
            base_dir = "results"
        self.base_dir = Path(base_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory structure
        self.results_dir = self.base_dir / f"{experiment_name}_{self.timestamp}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoints_dir = self.results_dir / "model_checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        self.visualizations_dir = self.results_dir / "visualizations"
        self.visualizations_dir.mkdir(exist_ok=True)
        
        # Initialize CSV file for results logging
        self.results_file = self.results_dir / "experiment_results.csv"
        self.training_logs_file = self.results_dir / "training_logs.csv"
        
        self._init_results_csv()
        self._init_training_logs_csv()
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure logging for experiment tracking."""
        log_file = self.results_dir / "experiment.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Started experiment: {self.experiment_name}")
        
    def _init_results_csv(self):
        """Initialize CSV file for experiment results."""
        headers = [
            'experiment_id', 'run_id', 'data_ordering', 'num_clusters', 
            'curriculum_strategy', 'batch_size', 'learning_rate', 'num_epochs',
            'final_accuracy', 'best_accuracy', 'training_time', 'convergence_epoch',
            'final_loss', 'model_checkpoint'
        ]
        
        with open(self.results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
    def _init_training_logs_csv(self):
        """Initialize CSV file for per-epoch training logs."""
        headers = [
            'experiment_id', 'run_id', 'epoch', 'train_loss', 'train_accuracy',
            'val_loss', 'val_accuracy', 'learning_rate'
        ]
        
        with open(self.training_logs_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
    def log_experiment_result(self, experiment_config, results):
        """Log final experiment results to CSV.
        
        Args:
            experiment_config: Dict with experiment parameters
            results: Dict with final results (accuracy, loss, etc.)
        """
        row = [
            experiment_config.get('experiment_id', 0),
            experiment_config.get('run_id', 0),
            experiment_config.get('data_ordering', 'random'),
            experiment_config.get('num_clusters', None),
            experiment_config.get('curriculum_strategy', None),
            experiment_config.get('batch_size', 64),
            experiment_config.get('learning_rate', 0.001),
            experiment_config.get('num_epochs', 10),
            results.get('final_accuracy', 0.0),
            results.get('best_accuracy', 0.0),
            results.get('training_time', 0.0),
            results.get('convergence_epoch', None),
            results.get('final_loss', 0.0),
            results.get('model_checkpoint', '')
        ]
        
        with open(self.results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
            
        self.logger.info(f"Logged experiment {experiment_config.get('experiment_id')} "
                        f"run {experiment_config.get('run_id')}: "
                        f"accuracy={results.get('final_accuracy', 0):.2f}%")
        
    def log_training_epoch(self, experiment_id, run_id, epoch, train_metrics, val_metrics, lr):
        """Log per-epoch training metrics.
        
        Args:
            experiment_id: Experiment identifier
            run_id: Run identifier  
            epoch: Epoch number
            train_metrics: Dict with train_loss, train_accuracy
            val_metrics: Dict with val_loss, val_accuracy
            lr: Current learning rate
        """
        row = [
            experiment_id, run_id, epoch,
            train_metrics.get('loss', 0.0),
            train_metrics.get('accuracy', 0.0),
            val_metrics.get('loss', 0.0),
            val_metrics.get('accuracy', 0.0),
            lr
        ]
        
        with open(self.training_logs_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
            
    def save_final_model(self, model, optimizer, final_metrics, experiment_id, run_id, experiment_config):
        """Save final model after training completion.
        
        Args:
            model: PyTorch model to save
            optimizer: Optimizer state
            final_metrics: Dict with final accuracy, loss, etc.
            experiment_id: Experiment identifier
            run_id: Run identifier
            experiment_config: Dict with experiment configuration
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_name = f"final_model_exp{experiment_id}_run{run_id}_acc{final_metrics['final_accuracy']:.2f}.pth"
        checkpoint_path = self.checkpoints_dir / checkpoint_name
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'final_metrics': final_metrics,
            'experiment_id': experiment_id,
            'run_id': run_id,
            'experiment_config': experiment_config,
            'timestamp': self.timestamp
        }, checkpoint_path)
        
        self.logger.info(f"Saved final model: {checkpoint_name}")
        return str(checkpoint_path)
        
    def plot_training_curves(self, experiment_ids=None, save=True):
        """Plot training curves comparing different experiments.
        
        Args:
            experiment_ids: List of experiment IDs to plot. If None, plots all.
            save: Whether to save plot to file
        """
        # Read training logs
        df = pd.read_csv(self.training_logs_file)
        
        if experiment_ids:
            df = df[df['experiment_id'].isin(experiment_ids)]
            
        # Group by experiment and compute mean across runs
        grouped = df.groupby(['experiment_id', 'epoch']).agg({
            'train_accuracy': ['mean', 'std'],
            'val_accuracy': ['mean', 'std'],
            'train_loss': ['mean', 'std'],
            'val_loss': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        grouped.columns = ['_'.join(col).strip() if col[1] else col[0] for col in grouped.columns.values]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot accuracy curves
        for exp_id in grouped['experiment_id'].unique():
            exp_data = grouped[grouped['experiment_id'] == exp_id]
            
            ax1.plot(exp_data['epoch'], exp_data['train_accuracy_mean'], 
                    label=f'Exp {exp_id} Train', alpha=0.8)
            ax1.fill_between(exp_data['epoch'],
                           exp_data['train_accuracy_mean'] - exp_data['train_accuracy_std'],
                           exp_data['train_accuracy_mean'] + exp_data['train_accuracy_std'],
                           alpha=0.2)
            
            ax2.plot(exp_data['epoch'], exp_data['val_accuracy_mean'],
                    label=f'Exp {exp_id} Val', alpha=0.8)
            ax2.fill_between(exp_data['epoch'],
                           exp_data['val_accuracy_mean'] - exp_data['val_accuracy_std'],
                           exp_data['val_accuracy_mean'] + exp_data['val_accuracy_std'],
                           alpha=0.2)
            
            ax3.plot(exp_data['epoch'], exp_data['train_loss_mean'],
                    label=f'Exp {exp_id} Train', alpha=0.8)
            ax3.fill_between(exp_data['epoch'],
                           exp_data['train_loss_mean'] - exp_data['train_loss_std'],
                           exp_data['train_loss_mean'] + exp_data['train_loss_std'],
                           alpha=0.2)
                           
            ax4.plot(exp_data['epoch'], exp_data['val_loss_mean'],
                    label=f'Exp {exp_id} Val', alpha=0.8)
            ax4.fill_between(exp_data['epoch'],
                           exp_data['val_loss_mean'] - exp_data['val_loss_std'],
                           exp_data['val_loss_mean'] + exp_data['val_loss_std'],
                           alpha=0.2)
        
        ax1.set_title('Training Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.legend()
        ax1.grid(True)
        
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        ax3.set_title('Training Loss')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.legend()
        ax3.grid(True)
        
        ax4.set_title('Validation Loss')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save:
            plot_path = self.plots_dir / "training_curves.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved training curves plot: {plot_path}")
            
        return fig
        
    def save_cluster_visualizations(self, images, labels, cluster_info, strategy_name):
        """Save visualization of cluster examples.
        
        Args:
            images: Tensor of images [N, 1, 28, 28] (sample images, not full dataset)
            labels: Tensor of Fashion-MNIST labels [N] (corresponding labels)
            cluster_info: Dict with cluster assignments and distances
            strategy_name: Name of the clustering strategy
        """
        fig, axes = plt.subplots(2, 10, figsize=(20, 8))
        
        # Get medoid and edge indices (these are from full 60k dataset)
        medoid_indices = cluster_info.get('medoid_indices', [])
        edge_indices = cluster_info.get('edge_indices', [])
        
        # We need to find medoids/edges that are within our sample images
        # For visualization, we'll just show the first 10 available images from each category
        sample_size = len(images)
        
        # Show medoids (filter to available sample range)
        available_medoids = [idx for idx in medoid_indices if idx < sample_size]
        for i in range(min(10, len(available_medoids))):
            if i < len(available_medoids):
                idx = available_medoids[i]
                img = images[idx].squeeze().numpy()
                label = labels[idx].item()
                
                axes[0, i].imshow(img, cmap='gray')
                axes[0, i].set_title(f'Medoid\n{FASHION_CLASSES[label]}')
            else:
                # Show placeholder if not enough medoids in sample
                axes[0, i].imshow(np.zeros((32, 32)), cmap='gray')
                axes[0, i].set_title('N/A')
            axes[0, i].axis('off')
            
        # Show edge cases (filter to available sample range)  
        available_edges = [idx for idx in edge_indices if idx < sample_size]
        for i in range(min(10, len(available_edges))):
            if i < len(available_edges):
                idx = available_edges[i]
                img = images[idx].squeeze().numpy()
                label = labels[idx].item()
                
                axes[1, i].imshow(img, cmap='gray')
                axes[1, i].set_title(f'Edge Case\n{FASHION_CLASSES[label]}')
            else:
                # Show placeholder if not enough edges in sample
                axes[1, i].imshow(np.zeros((32, 32)), cmap='gray')
                axes[1, i].set_title('N/A')
            axes[1, i].axis('off')
            
        plt.suptitle(f'Cluster Examples: {strategy_name}', fontsize=16)
        plt.tight_layout()
        
        plot_path = self.visualizations_dir / f"cluster_examples_{strategy_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Saved cluster visualization: {plot_path}")
        plt.close()
        
        return str(plot_path)
        
    def generate_experiment_report(self):
        """Generate a comprehensive experiment report."""
        # Read results
        results_df = pd.read_csv(self.results_file)
        
        # Generate summary statistics
        summary = results_df.groupby(['data_ordering', 'curriculum_strategy']).agg({
            'final_accuracy': ['mean', 'std', 'count'],
            'training_time': ['mean', 'std'],
            'convergence_epoch': ['mean', 'std']
        }).round(3)
        
        # Save report
        report_path = self.results_dir / "experiment_report.md"
        with open(report_path, 'w') as f:
            f.write(f"# {self.experiment_name.title()} Experiment Report\n\n")
            f.write(f"**Experiment Date:** {self.timestamp}\n\n")
            f.write(f"## Summary Statistics\n\n")
            f.write(summary.to_markdown())
            f.write(f"\n\n## Results Files\n\n")
            f.write(f"- Detailed results: `experiment_results.csv`\n")
            f.write(f"- Training logs: `training_logs.csv`\n")
            f.write(f"- Model checkpoints: `model_checkpoints/`\n")
            f.write(f"- Visualizations: `visualizations/`\n")
            f.write(f"- Training curves: `plots/training_curves.png`\n")
            
        self.logger.info(f"Generated experiment report: {report_path}")
        return str(report_path)
        
    def get_results_summary(self):
        """Get a summary of all experiment results."""
        if not self.results_file.exists():
            return "No results available yet."
            
        df = pd.read_csv(self.results_file)
        return df.describe()