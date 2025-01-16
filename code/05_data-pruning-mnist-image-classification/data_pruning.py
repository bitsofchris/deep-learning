import numpy as np
from sklearn.cluster import KMeans


def _cluster_prune_indices(train_dataset, k=10000, random_state=42):
    """
    Clusters the raw pixel data of MNIST into k clusters
    and picks the single sample closest to each cluster's centroid.

    Args:
        train_dataset: A torchvision MNIST dataset object (with .data, .targets).
        k: Number of clusters to form. For large k, you'll keep more samples.
        random_state: For reproducible clustering.

    Returns:
        pruned_indices: A list of training-set indices,
                        one representative per cluster
                        (k or fewer if some clusters are empty).
    """

    # 1) Extract raw pixel data from dataset [N, 28, 28] -> flatten to [N, 784]
    X = train_dataset.data  # shape (60000, 28, 28) by default
    X_flat = X.float().view(-1, 784).numpy()  # shape (60000, 784)

    # 2) Run k-means on the flattened pixel vectors
    print(f"Running k-means on {len(X_flat)} samples with k={k} ...")
    kmeans = KMeans(n_clusters=k, random_state=random_state).fit(X_flat)
    labels = kmeans.labels_  # cluster assignment for each sample [N]
    centroids = kmeans.cluster_centers_  # shape (k, 784)

    # 3) For each cluster, pick the sample closest to that centroid
    pruned_indices = []
    for cluster_id in range(k):
        # Grab all sample indices that belong to cluster_id
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            # This cluster got no samples (can happen if k is large)
            continue

        # Subset the flattened data for this cluster
        subX = X_flat[cluster_indices]  # shape (num_in_cluster, 784)
        center = centroids[cluster_id]  # shape (784,)

        # Compute squared distances from each sample to the centroid
        distances = np.sum((subX - center) ** 2, axis=1)  # shape (num_in_cluster,)

        # Find the sample with the minimum distance
        min_idx_local = np.argmin(distances)
        min_idx_global = cluster_indices[min_idx_local]

        # Keep that global index
        pruned_indices.append(min_idx_global)

    print(f"Found {len(pruned_indices)} representative samples for {k} clusters.")
    return pruned_indices


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

    elif method == "random":
        # Take a random percentage of the indices
        percentage = kwargs.get("percentage", 0.5)  # default to 50%
        print(f"Using pruning method: random, with {percentage*100}% of data.")
        num_indices = len(train_dataset)
        num_indices_to_keep = int(num_indices * percentage)
        pruned_indices = np.random.choice(
            num_indices, num_indices_to_keep, replace=False
        )
        return pruned_indices.tolist()

    elif method == "cluster":
        k = kwargs.get("k", 10000)
        pruned_indices = _cluster_prune_indices(train_dataset, k=k, random_state=37)
        return pruned_indices

    else:
        raise ValueError(f"Unsupported pruning method: {method}")
