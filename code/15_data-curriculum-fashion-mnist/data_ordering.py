import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from typing import List, Optional, Dict, Tuple
import logging

from clustering import FAISSClustering, cluster_fashion_mnist


class OrderedDataset(Dataset):
    """Dataset wrapper that returns data in a specific order."""

    def __init__(self, base_dataset: Dataset, ordered_indices: np.ndarray):
        """Initialize ordered dataset.

        Args:
            base_dataset: Base Fashion-MNIST dataset
            ordered_indices: Array of indices defining the order
        """
        self.base_dataset = base_dataset
        self.ordered_indices = ordered_indices

    def __len__(self):
        return len(self.ordered_indices)

    def __getitem__(self, idx):
        # Map to the original dataset index
        original_idx = self.ordered_indices[idx]
        return self.base_dataset[original_idx]


class DataOrdering:
    """Manages different data ordering strategies for curriculum learning."""

    def __init__(
        self, batch_size: int = 64, num_workers: int = 2, random_state: int = 42
    ):
        """Initialize data ordering manager.

        Args:
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
            random_state: Random seed for reproducibility
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_state = random_state

        self.logger = logging.getLogger(__name__)

        # Data transforms - resize to 32x32 for LeNet5 compatibility
        self.transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,)),  # Fashion-MNIST statistics
            ]
        )

    def get_fashion_mnist_datasets(
        self, data_dir: str = "data"
    ) -> Tuple[Dataset, Dataset]:
        """Load Fashion-MNIST datasets.

        Args:
            data_dir: Directory to store/load data

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        train_dataset = datasets.FashionMNIST(
            root=data_dir, train=True, download=True, transform=self.transform
        )

        test_dataset = datasets.FashionMNIST(
            root=data_dir, train=False, download=True, transform=self.transform
        )

        self.logger.info(
            f"Loaded Fashion-MNIST: {len(train_dataset)} train, {len(test_dataset)} test"
        )
        return train_dataset, test_dataset

    def get_random_ordering(self, dataset_size: int) -> np.ndarray:
        """Get random ordering of dataset indices.

        Args:
            dataset_size: Size of the dataset

        Returns:
            Randomly ordered indices
        """
        np.random.seed(self.random_state)
        return np.random.permutation(dataset_size)

    def get_curriculum_ordering(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        strategy: str = "medoid_first",
        n_clusters: int = 50,
        pca_components: int = 50,
        use_pca: bool = True,
    ) -> Tuple[np.ndarray, Dict]:
        """Get curriculum learning ordering using clustering.

        Args:
            images: Fashion-MNIST images [N, C, H, W]
            labels: Fashion-MNIST labels [N]
            strategy: Curriculum strategy
                - "medoid_first": Medoids → Q1 → Q2 → Q3 → Q4
                - "quartiles_only": Q1 → Q2 → Q3 → Q4
            n_clusters: Number of clusters for analysis
            pca_components: Number of PCA components
            use_pca: Whether to apply PCA before clustering

        Returns:
            Tuple of (ordered_indices, cluster_info)
        """
        self.logger.info(f"Generating curriculum ordering with strategy: {strategy}")

        # Perform clustering
        clusterer, cluster_info = cluster_fashion_mnist(
            images=images,
            labels=labels,
            n_clusters=n_clusters,
            pca_components=pca_components,
            use_pca=use_pca,
            random_state=self.random_state,
        )

        # Get curriculum ordering
        ordered_indices = clusterer.get_curriculum_ordering(strategy=strategy)

        # Add ordering info to cluster_info
        cluster_info["ordering_strategy"] = strategy
        cluster_info["ordered_indices"] = ordered_indices

        self.logger.info(f"Generated curriculum with {len(ordered_indices)} samples")
        return ordered_indices, cluster_info

    def create_data_loaders(
        self,
        ordering_strategy: str = "random",
        curriculum_params: Optional[Dict] = None,
        data_dir: str = "data",
    ) -> Tuple[DataLoader, DataLoader, Optional[Dict]]:
        """Create data loaders with specified ordering strategy.

        Args:
            ordering_strategy: Data ordering strategy
                - "random": Random shuffling (baseline)
                - "curriculum": Curriculum learning ordering
            curriculum_params: Parameters for curriculum learning (if applicable)
                - strategy: "medoid_first" or "quartiles_only"
                - n_clusters: number of clusters
                - pca_components: PCA dimensions
                - use_pca: whether to use PCA
            data_dir: Directory for Fashion-MNIST data

        Returns:
            Tuple of (train_loader, test_loader, cluster_info)
        """
        # Load datasets
        train_dataset, test_dataset = self.get_fashion_mnist_datasets(data_dir)

        cluster_info = None

        if ordering_strategy == "random":
            # Random baseline - just use normal DataLoader with shuffling
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )
            self.logger.info("Created random ordering data loader")

        elif ordering_strategy == "curriculum":
            # Curriculum learning - need to determine order first
            if curriculum_params is None:
                curriculum_params = {
                    "strategy": "medoid_first",
                    "n_clusters": 50,
                    "pca_components": 50,
                    "use_pca": True,
                }

            # Extract all training images and labels for clustering
            self.logger.info("Extracting training data for clustering...")
            all_images = []
            all_labels = []

            # Use a temporary loader to extract all data
            temp_loader = DataLoader(train_dataset, batch_size=1000, shuffle=False)
            for batch_images, batch_labels in temp_loader:
                all_images.append(batch_images)
                all_labels.append(batch_labels)

            all_images = torch.cat(all_images, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            self.logger.info(
                f"Extracted {len(all_images)} training samples for clustering"
            )

            # Get curriculum ordering
            ordered_indices, cluster_info = self.get_curriculum_ordering(
                images=all_images, labels=all_labels, **curriculum_params
            )

            # Create ordered dataset
            ordered_dataset = OrderedDataset(train_dataset, ordered_indices)

            # Create data loader (no shuffling since we want our specific order)
            train_loader = DataLoader(
                ordered_dataset,
                batch_size=self.batch_size,
                shuffle=False,  # Important: maintain curriculum order
                num_workers=self.num_workers,
            )

            self.logger.info(
                f"Created curriculum ordering data loader with {len(ordered_dataset)} samples"
            )

        else:
            raise ValueError(f"Unknown ordering strategy: {ordering_strategy}")

        # Test loader is always the same (no special ordering needed)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return train_loader, test_loader, cluster_info

    def create_curriculum_phases(
        self,
        train_dataset: Dataset,
        ordered_indices: np.ndarray,
        phase_sizes: List[int],
    ) -> List[DataLoader]:
        """Create data loaders for different curriculum phases.

        Useful for incremental curriculum learning where you gradually
        add more data as training progresses.

        Args:
            train_dataset: Base training dataset
            ordered_indices: Curriculum-ordered indices
            phase_sizes: List of cumulative sizes for each phase

        Returns:
            List of DataLoaders for each phase
        """
        phase_loaders = []

        for i, size in enumerate(phase_sizes):
            if size > len(ordered_indices):
                size = len(ordered_indices)

            phase_indices = ordered_indices[:size]
            phase_dataset = OrderedDataset(train_dataset, phase_indices)

            phase_loader = DataLoader(
                phase_dataset,
                batch_size=self.batch_size,
                shuffle=False,  # Maintain curriculum order
                num_workers=self.num_workers,
            )

            phase_loaders.append(phase_loader)
            self.logger.info(
                f"Created curriculum phase {i+1} with {len(phase_dataset)} samples"
            )

        return phase_loaders


def get_data_loaders(
    ordering_strategy: str = "random",
    curriculum_params: Optional[Dict] = None,
    batch_size: int = 64,
    num_workers: int = 2,
    data_dir: str = "data",
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader, Optional[Dict]]:
    """Convenience function to create data loaders with specified ordering.

    Args:
        ordering_strategy: "random" or "curriculum"
        curriculum_params: Parameters for curriculum learning
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        data_dir: Directory for Fashion-MNIST data
        random_state: Random seed

    Returns:
        Tuple of (train_loader, test_loader, cluster_info)
    """
    data_orderer = DataOrdering(
        batch_size=batch_size, num_workers=num_workers, random_state=random_state
    )

    return data_orderer.create_data_loaders(
        ordering_strategy=ordering_strategy,
        curriculum_params=curriculum_params,
        data_dir=data_dir,
    )
