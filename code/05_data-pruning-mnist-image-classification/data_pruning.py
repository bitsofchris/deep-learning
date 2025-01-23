import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA


def _cluster_prune_indices(
    train_dataset,
    k=50,  # number of clusters
    target_size=10000,  # final desired number of samples
    selection_strategy="closest",  # "closest", "furthest", "random", or "hybrid"
    n_closest=1,  # used if strategy="hybrid"
    n_furthest=1,  # used if strategy="hybrid"
    apply_pca=False,
    pca_components=50,
    random_state=37,
    subsample_size=60000,  # how many samples to use for partial fitting in k-means
):
    """
    Performs K-means clustering on raw (or PCA-reduced) pixel data,
    then selects samples from each cluster according to selection_strategy.

    Args:
        train_dataset: A torchvision MNIST dataset object (with .data, .targets).
        k:             Number of clusters to form in K-means.
        target_size:   Desired final # of samples across entire dataset.
        selection_strategy: "closest", "furthest", "random", or "hybrid" (both).
        n_closest, n_furthest: For "hybrid", how many from each side of centroid distance.
        apply_pca:     Whether to run PCA on flattened images first.
        pca_components:# of PCA components if apply_pca=True.
        random_state:  Random seed for reproducibility.
        subsample_size: Subsample count for faster MiniBatchKMeans.

    Returns:
        pruned_indices: A list of sample indices to keep (up to target_size).
    """

    # 1) Flatten MNIST images to [N, 784]
    X = train_dataset.data  # shape [N, 28, 28]
    X_flat = X.float().view(-1, 784).numpy()  # shape [N, 784]
    N = len(X_flat)

    # 2) Optionally apply PCA
    if apply_pca:
        print(f"[INFO] Applying PCA to reduce to {pca_components}D before clustering.")
        pca = PCA(n_components=pca_components, random_state=random_state)
        X_flat = pca.fit_transform(X_flat)  # shape [N, pca_components]

    d = X_flat.shape[1]  # dimension after PCA (or 784 if no PCA)

    # 3) Determine how many samples per cluster
    #    - If 'hybrid', we pick (n_closest + n_furthest) from each cluster =>
    #           ~k*(n_closest + n_furthest) total
    #    - Otherwise, samples_per_cluster = target_size // k
    if selection_strategy == "hybrid":
        per_cluster = n_closest + n_furthest
        print(
            f"[INFO] Hybrid strategy: picking {n_closest} closest + {n_furthest}"
            f"furthest per cluster => {per_cluster} total each."
        )
    else:
        # Basic approach: integer division
        per_cluster = max(1, target_size // k)  # ensure at least 1 if k>target_size
        print(
            f"[INFO] Strategy={selection_strategy}, using {per_cluster} samples "
            "per cluster."
        )

    print(
        f"[INFO] K-means with k={k}, aiming for ~{k * per_cluster} samples (truncating "
        f"at {target_size})."
    )

    # 4) Subsample the data for faster K-means (if subsample_size < N)
    if subsample_size < N:
        sample_indices_for_clustering = np.random.choice(
            N, subsample_size, replace=False
        )
        X_sample = X_flat[sample_indices_for_clustering]
    else:
        X_sample = X_flat

    # 5) Fit MiniBatchKMeans
    print(f"[INFO] Running k-means on {len(X_sample)} samples, k={k} ...")
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        batch_size=1024,
        max_iter=20,
        max_no_improvement=5,
        random_state=random_state,
    )
    kmeans.fit(X_sample)
    centroids = kmeans.cluster_centers_  # shape (k, d)

    # 6) Assign each point in the FULL dataset to its nearest cluster
    labels = kmeans.predict(X_flat)  # shape [N,]

    pruned_indices = []
    rng = np.random.default_rng(random_state)  # for random draws

    # 7) For each cluster, pick data
    for cluster_id in range(k):
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue  # empty cluster

        subX = X_flat[cluster_indices]  # shape [X_in_that_cluster, d]
        center = centroids[cluster_id]

        # Distances to centroid
        distances = np.sum((subX - center) ** 2, axis=1)
        sort_idx = np.argsort(distances)  # ascending order => closest first

        if selection_strategy == "closest":
            # pick the top 'per_cluster' from the front
            chosen_local_idx = sort_idx[:per_cluster]

        elif selection_strategy == "furthest":
            # pick the top 'per_cluster' from the end
            chosen_local_idx = sort_idx[-per_cluster:]

        elif selection_strategy == "random":
            # pick 'per_cluster' random indices from cluster
            n_to_pick = min(per_cluster, len(cluster_indices))
            chosen_local_idx = rng.choice(
                len(cluster_indices), size=n_to_pick, replace=False
            )

        elif selection_strategy == "hybrid":
            # pick n_closest from the front, n_furthest from the end
            c = min(n_closest, len(cluster_indices))
            chosen_closest = sort_idx[:c]

            f = min(n_furthest, len(cluster_indices) - c)
            chosen_furthest = sort_idx[-f:] if f > 0 else []

            chosen_local_idx = np.concatenate([chosen_closest, chosen_furthest])
        else:
            raise ValueError(f"Unknown selection_strategy: {selection_strategy}")

        # Convert local -> global indices
        chosen_global_idx = cluster_indices[chosen_local_idx]
        pruned_indices.extend(chosen_global_idx.tolist())

    # 8) Truncate if we exceed target_size
    if len(pruned_indices) > target_size:
        # Naive approach: just take the first 'target_size'
        pruned_indices = pruned_indices[:target_size]

    print(
        f"[INFO] Final pruned set size = {len(pruned_indices)} (target={target_size})."
    )
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
        pruned_indices = _cluster_prune_indices(train_dataset, **kwargs)
        return pruned_indices

    else:
        raise ValueError(f"Unsupported pruning method: {method}")
