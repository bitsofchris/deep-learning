import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, List, Tuple, Optional


class FAISSClustering:
    """MiniBatchKMeans clustering with PCA dimensionality reduction for Fashion-MNIST."""
    
    def __init__(self, 
                 n_clusters: int = 50,
                 pca_components: int = 50,
                 use_pca: bool = True,
                 random_state: int = 42):
        """Initialize FAISS clustering.
        
        Args:
            n_clusters: Number of clusters to create
            pca_components: Number of PCA components (if use_pca=True)
            use_pca: Whether to apply PCA before clustering
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.pca_components = pca_components
        self.use_pca = use_pca
        self.random_state = random_state
        
        # Initialize components
        self.pca = None
        self.scaler = StandardScaler()
        self.kmeans = None
        self.cluster_centers = None
        self.labels = None
        
        # Cluster analysis results
        self.medoid_indices = []
        self.edge_indices = []
        self.quartile_indices = {'q1': [], 'q2': [], 'q3': [], 'q4': []}
        
        self.logger = logging.getLogger(__name__)
        
    def _prepare_features(self, images: torch.Tensor) -> np.ndarray:
        """Prepare image features for clustering.
        
        Args:
            images: Fashion-MNIST images tensor [N, C, H, W]
            
        Returns:
            Processed features array [N, features]
        """
        # Convert to numpy and flatten images
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
            
        # Flatten images to [N, H*W*C]
        features = images.reshape(len(images), -1)
        
        # Standardize features
        features = self.scaler.fit_transform(features)
        
        # Apply PCA if requested
        if self.use_pca:
            self.logger.info(f"Applying PCA to reduce dimensionality to {self.pca_components}")
            self.pca = PCA(n_components=self.pca_components, random_state=self.random_state)
            features = self.pca.fit_transform(features)
            self.logger.info(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        return features.astype(np.float32)
        
    def fit_predict(self, images: torch.Tensor) -> np.ndarray:
        """Fit FAISS clustering and predict cluster assignments.
        
        Args:
            images: Fashion-MNIST images tensor [N, C, H, W]
            
        Returns:
            Cluster labels array [N]
        """
        self.logger.info(f"Starting FAISS clustering with {self.n_clusters} clusters")
        
        # Prepare features
        features = self._prepare_features(images)
        n_samples, n_features = features.shape
        
        # Subsample if too large to avoid memory issues
        max_samples = 10000  # Limit for FAISS clustering
        if n_samples > max_samples:
            self.logger.info(f"Subsampling {max_samples} from {n_samples} samples for clustering")
            subsample_indices = np.random.RandomState(self.random_state).choice(
                n_samples, max_samples, replace=False
            )
            features_for_clustering = features[subsample_indices]
        else:
            features_for_clustering = features
            subsample_indices = None
        
        self.logger.info(f"Feature matrix shape for clustering: {features_for_clustering.shape}")
        self.logger.info(f"Original feature matrix shape: {features.shape}")
        
        # Initialize MiniBatchKMeans (stable on all platforms)
        self.logger.info("Using MiniBatchKMeans clustering")
        kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=1024,
            max_iter=20,
            max_no_improvement=5,
            random_state=self.random_state,
            n_init=3
        )
        
        # Fit clustering on subsampled data
        kmeans.fit(features_for_clustering)
        
        # Get cluster assignments for all samples
        labels = kmeans.predict(features)
        
        # Store results
        self.kmeans = kmeans
        self.cluster_centers = kmeans.cluster_centers_
        self.labels = labels
        
        self.logger.info(f"Clustering completed. Found {len(np.unique(labels))} clusters")
        
        # Analyze clusters for curriculum ordering
        self._analyze_clusters(features, labels)
        
        return labels
        
    def _analyze_clusters(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Analyze clusters to identify medoids, edge cases, and quartiles.
        
        Args:
            features: Feature matrix [N, features]  
            labels: Cluster assignments [N]
        """
        self.logger.info("Analyzing clusters for curriculum ordering")
        
        self.medoid_indices = []
        self.edge_indices = []
        self.quartile_indices = {'q1': [], 'q2': [], 'q3': [], 'q4': []}
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                continue
                
            # Get cluster features and centroid
            cluster_features = features[cluster_mask]
            centroid = self.cluster_centers[cluster_id]
            
            # Calculate distances to centroid
            distances = np.sum((cluster_features - centroid) ** 2, axis=1)
            
            # Sort by distance (closest to furthest)
            sorted_indices = np.argsort(distances)
            sorted_global_indices = cluster_indices[sorted_indices]
            
            # Store medoid (closest to centroid)
            self.medoid_indices.append(sorted_global_indices[0])
            
            # Store edge case (furthest from centroid)
            self.edge_indices.append(sorted_global_indices[-1])
            
            # Divide into quartiles for curriculum learning (excluding medoid to avoid duplication)
            remaining_indices = sorted_global_indices[1:]  # Skip medoid (first element)
            n_samples = len(remaining_indices)
            
            if n_samples >= 3:
                q1_end = n_samples // 4
                q2_end = n_samples // 2
                q3_end = 3 * n_samples // 4
                
                self.quartile_indices['q1'].extend(remaining_indices[:q1_end])
                self.quartile_indices['q2'].extend(remaining_indices[q1_end:q2_end])
                self.quartile_indices['q3'].extend(remaining_indices[q2_end:q3_end])
                self.quartile_indices['q4'].extend(remaining_indices[q3_end:])
            elif n_samples > 0:
                # For small clusters, put remaining samples in q1
                self.quartile_indices['q1'].extend(remaining_indices)
        
        # Convert to numpy arrays
        self.medoid_indices = np.array(self.medoid_indices)
        self.edge_indices = np.array(self.edge_indices)
        for q in self.quartile_indices:
            self.quartile_indices[q] = np.array(self.quartile_indices[q])
            
        self.logger.info(f"Identified {len(self.medoid_indices)} medoids")
        self.logger.info(f"Quartile sizes - Q1: {len(self.quartile_indices['q1'])}, "
                        f"Q2: {len(self.quartile_indices['q2'])}, "
                        f"Q3: {len(self.quartile_indices['q3'])}, "
                        f"Q4: {len(self.quartile_indices['q4'])}")
        
    def get_curriculum_ordering(self, strategy: str = "medoid_first") -> np.ndarray:
        """Get curriculum learning ordering of samples.
        
        Args:
            strategy: Ordering strategy
                - "medoid_first": Medoids → Q1 → Q2 → Q3 → Q4 (easy→hard)
                - "quartiles_only": Q1 → Q2 → Q3 → Q4 (easy→hard, no separate medoids)
                - "inverse": Q4 → Q3 → Q2 → Q1 → Medoids (hard→easy)
                - "random": Random ordering (baseline)
                
        Returns:
            Ordered indices array
        """
        if self.labels is None:
            raise ValueError("Must run fit_predict first")
            
        if strategy == "medoid_first":
            # Start with medoids (most prototypical), then quartiles
            ordered_indices = np.concatenate([
                self.medoid_indices,
                self.quartile_indices['q1'],
                self.quartile_indices['q2'], 
                self.quartile_indices['q3'],
                self.quartile_indices['q4']
            ])
            
        elif strategy == "quartiles_only":
            # Just quartile ordering without separate medoids (easy→hard)
            ordered_indices = np.concatenate([
                self.quartile_indices['q1'],
                self.quartile_indices['q2'],
                self.quartile_indices['q3'],
                self.quartile_indices['q4']
            ])
            
        elif strategy == "inverse":
            # Inverse curriculum: hard→easy (opposite of medoid_first)
            ordered_indices = np.concatenate([
                self.quartile_indices['q4'],  # Hardest first
                self.quartile_indices['q3'],
                self.quartile_indices['q2'],
                self.quartile_indices['q1'],
                self.medoid_indices           # Easiest (prototypes) last
            ])
            
        elif strategy == "round_robin":
            # Maximum diversity: round-robin sampling from all clusters
            ordered_indices = self._create_round_robin_ordering()
            
        elif strategy == "random":
            # Random baseline
            n_samples = len(self.labels)
            ordered_indices = np.random.RandomState(self.random_state).permutation(n_samples)
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            
        self.logger.info(f"Generated curriculum ordering with strategy '{strategy}': {len(ordered_indices)} samples")
        return ordered_indices
        
    def get_cluster_info(self) -> Dict:
        """Get comprehensive cluster analysis information.
        
        Returns:
            Dictionary with cluster analysis results
        """
        if self.labels is None:
            raise ValueError("Must run fit_predict first")
            
        cluster_sizes = [np.sum(self.labels == i) for i in range(self.n_clusters)]
        
        return {
            'n_clusters': self.n_clusters,
            'cluster_sizes': cluster_sizes,
            'total_samples': len(self.labels),
            'medoid_indices': self.medoid_indices,
            'edge_indices': self.edge_indices,
            'quartile_indices': self.quartile_indices,
            'use_pca': self.use_pca,
            'pca_components': self.pca_components if self.use_pca else None,
            'pca_variance_explained': self.pca.explained_variance_ratio_.sum() if self.pca else None
        }


def cluster_fashion_mnist(images: torch.Tensor, 
                         labels: torch.Tensor,
                         n_clusters: int = 50,
                         pca_components: int = 50,
                         use_pca: bool = True,
                         random_state: int = 42) -> Tuple[FAISSClustering, Dict]:
    """Convenience function to cluster Fashion-MNIST dataset.
    
    Args:
        images: Fashion-MNIST images [N, C, H, W]
        labels: Fashion-MNIST class labels [N]
        n_clusters: Number of clusters
        pca_components: Number of PCA components
        use_pca: Whether to use PCA
        random_state: Random seed
        
    Returns:
        Tuple of (clustering object, cluster info dict)
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting Fashion-MNIST clustering pipeline")
    
    # Initialize and fit clustering
    clusterer = FAISSClustering(
        n_clusters=n_clusters,
        pca_components=pca_components,
        use_pca=use_pca,
        random_state=random_state
    )
    
    # Fit clustering
    cluster_labels = clusterer.fit_predict(images)
    
    # Get comprehensive analysis
    cluster_info = clusterer.get_cluster_info()
    cluster_info['fashion_labels'] = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    cluster_info['cluster_assignments'] = cluster_labels
    
    logger.info("Fashion-MNIST clustering completed successfully")
    
    return clusterer, cluster_info