import numpy as np


def prune_indices(train_dataset, method="none", **kwargs):
    """
    Args:
        train_dataset: The original MNIST Dataset (with transforms).
        method: A string to pick pruning approach (duplicates, cluster, etc.).
        **kwargs: Additional parameters for your pruning logic.

    Returns:
        A list (or array) of indices that you want to keep in the final subset.
    """
    if method == "none":
        print("Using pruning method: none, keeping all data")
        # Keep all indices
        return list(range(len(train_dataset)))

    elif method == "duplicates":
        # Example (naive) duplicates approach:
        # 1. Convert each image to a flattened vector (in memory) to check duplicates
        # 2. This might be expensive for big datasets, but MNIST is small enough.

        all_data = train_dataset.data  # shape [60000, 28, 28]
        # Flatten into [60000, 784]
        all_data_flat = all_data.view(len(train_dataset), -1).numpy()

        # Use a set or a dictionary to track unique rows
        # For example, we can use numpy's unique or just a dict
        _, unique_indices = np.unique(all_data_flat, axis=0, return_index=True)

        # Sort so the subset is in ascending index
        pruned_indices = sorted(unique_indices.tolist())
        return pruned_indices

    elif method == "cluster":
        # Example cluster approach with k-means (placeholder):
        # We'll cluster the flattened images, then pick 1 or more reps per cluster.

        all_data = train_dataset.data  # [N, 28, 28]
        # Flatten
        all_data_flat = all_data.view(len(train_dataset), -1).float()  # [N, 784]

        k = kwargs.get("k", 5000)  # number of cluster centroids you want
        # ... run k-means on all_data_flat ...
        # ... pick representative indices for each cluster ...

        # For demonstration, let's just pick the first k
        pruned_indices = list(range(min(k, len(all_data_flat))))
        return pruned_indices

    else:
        raise ValueError(f"Unsupported pruning method: {method}")
