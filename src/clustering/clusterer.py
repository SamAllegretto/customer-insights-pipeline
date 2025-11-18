# src/clustering/cluster.py
import numpy as np
import random
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from typing import List, Dict, Any, Optional, Tuple

class Clusterer:
    """Clustering functionality for embeddings with LLM-based cluster labeling."""
    
    def __init__(self, algorithm: str = "kmeans", n_clusters: Optional[int] = None, 
                 distance_threshold: Optional[float] = None, eps: Optional[float] = None,
                 min_samples: int = 5, **kwargs):
        """
        Args:
            algorithm: 'kmeans', 'dbscan', 'agglomerative', 'agglomerative_distance'
            n_clusters: Number of clusters (kmeans, agglomerative)
            distance_threshold: Max distance for agglomerative_distance mode
            eps: Max distance between samples for DBSCAN
            min_samples: Min samples in neighborhood for DBSCAN core points
        """
        self.algorithm = algorithm.lower()
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold
        self.eps = eps
        self.min_samples = min_samples
        self.kwargs = kwargs
        self.model = self._init_model()
        self.embeddings_ = None
        self.labels_ = None

    def _init_model(self):
        if self.algorithm == "kmeans":
            if self.n_clusters is None:
                raise ValueError("n_clusters required for kmeans")
            return KMeans(n_clusters=self.n_clusters, random_state=42, **self.kwargs)
        
        elif self.algorithm == "dbscan":
            if self.eps is None:
                raise ValueError("eps (distance threshold) required for DBSCAN")
            return DBSCAN(eps=self.eps, min_samples=self.min_samples, **self.kwargs)
        
        elif self.algorithm == "agglomerative":
            if self.n_clusters is None:
                raise ValueError("n_clusters required for agglomerative")
            return AgglomerativeClustering(n_clusters=self.n_clusters, **self.kwargs)
        
        elif self.algorithm == "agglomerative_distance":
            if self.distance_threshold is None:
                raise ValueError("distance_threshold required for agglomerative_distance")
            return AgglomerativeClustering(
                n_clusters=None, 
                distance_threshold=self.distance_threshold,
                **self.kwargs
            )
        
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def fit_predict(self, embeddings: List[List[float]]) -> List[int]:
        X = np.array(embeddings)
        self.embeddings_ = X
        labels = self.model.fit_predict(X)
        self.labels_ = labels
        return labels.tolist()

    def get_cluster_centers(self) -> Optional[np.ndarray]:
        if hasattr(self.model, "cluster_centers_"):
            return self.model.cluster_centers_
        return None
    
    def _compute_cluster_centers(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict[int, np.ndarray]:
        centers = {}
        for cluster_id in np.unique(labels):
            if cluster_id == -1:
                continue
            cluster_mask = labels == cluster_id
            centers[cluster_id] = embeddings[cluster_mask].mean(axis=0)
        return centers

    def calculate_intra_cluster_distances(self, embeddings: Optional[List[List[float]]] = None, 
                                         labels: Optional[List[int]] = None) -> Dict[int, Dict[str, float]]:
        X = np.array(embeddings) if embeddings is not None else self.embeddings_
        lbls = np.array(labels) if labels is not None else self.labels_
        
        if X is None or lbls is None:
            raise ValueError("Must call fit_predict first or provide embeddings and labels")
        
        if hasattr(self.model, "cluster_centers_"):
            centers = {i: center for i, center in enumerate(self.model.cluster_centers_)}
        else:
            centers = self._compute_cluster_centers(X, lbls)
        
        cluster_stats = {}
        
        for cluster_id in np.unique(lbls):
            if cluster_id == -1:
                continue
                
            cluster_mask = lbls == cluster_id
            cluster_points = X[cluster_mask]
            center = centers[cluster_id]
            
            distances = np.linalg.norm(cluster_points - center, axis=1)
            
            cluster_stats[int(cluster_id)] = {
                'mean_distance': float(np.mean(distances)),
                'max_distance': float(np.max(distances)),
                'std_distance': float(np.std(distances)),
                'variance': float(np.var(distances)),
                'cluster_size': int(np.sum(cluster_mask))
            }
        
        return cluster_stats

    def calculate_inter_cluster_distances(self, embeddings: Optional[List[List[float]]] = None,
                                         labels: Optional[List[int]] = None) -> Dict[Tuple[int, int], float]:
        X = np.array(embeddings) if embeddings is not None else self.embeddings_
        lbls = np.array(labels) if labels is not None else self.labels_
        
        if X is None or lbls is None:
            raise ValueError("Must call fit_predict first or provide embeddings and labels")
        
        if hasattr(self.model, "cluster_centers_"):
            centers = {i: center for i, center in enumerate(self.model.cluster_centers_)}
        else:
            centers = self._compute_cluster_centers(X, lbls)
        
        inter_distances = {}
        cluster_ids = sorted(centers.keys())
        
        for i, id1 in enumerate(cluster_ids):
            for id2 in cluster_ids[i+1:]:
                distance = np.linalg.norm(centers[id1] - centers[id2])
                inter_distances[(id1, id2)] = float(distance)
        
        return inter_distances

    def calculate_silhouette_scores(self, embeddings: Optional[List[List[float]]] = None,
                                   labels: Optional[List[int]] = None) -> Dict[str, Any]:
        X = np.array(embeddings) if embeddings is not None else self.embeddings_
        lbls = np.array(labels) if labels is not None else self.labels_
        
        if X is None or lbls is None:
            raise ValueError("Must call fit_predict first or provide embeddings and labels")
        
        mask = lbls != -1
        if not np.any(mask):
            return {'overall_score': None, 'cluster_scores': {}}
        
        X_filtered = X[mask]
        lbls_filtered = lbls[mask]
        
        if len(np.unique(lbls_filtered)) < 2:
            return {'overall_score': None, 'cluster_scores': {}}
        
        overall_score = silhouette_score(X_filtered, lbls_filtered)
        sample_scores = silhouette_samples(X_filtered, lbls_filtered)
        
        cluster_scores = {}
        for cluster_id in np.unique(lbls_filtered):
            cluster_mask = lbls_filtered == cluster_id
            cluster_scores[int(cluster_id)] = float(np.mean(sample_scores[cluster_mask]))
        
        return {
            'overall_score': float(overall_score),
            'cluster_scores': cluster_scores
        }

    def get_distance_to_center(self, embedding: List[float], cluster_id: int) -> float:
        if self.embeddings_ is None or self.labels_ is None:
            raise ValueError("Must call fit_predict first")
        
        if hasattr(self.model, "cluster_centers_"):
            center = self.model.cluster_centers_[cluster_id]
        else:
            centers = self._compute_cluster_centers(self.embeddings_, self.labels_)
            center = centers[cluster_id]
        
        embedding_array = np.array(embedding)
        distance = np.linalg.norm(embedding_array - center)
        return float(distance)

    def get_clustering_quality_report(self, embeddings: Optional[List[List[float]]] = None,
                                     labels: Optional[List[int]] = None) -> Dict[str, Any]:
        intra_stats = self.calculate_intra_cluster_distances(embeddings, labels)
        inter_distances = self.calculate_inter_cluster_distances(embeddings, labels)
        silhouette = self.calculate_silhouette_scores(embeddings, labels)
        cluster_sizes = self.get_cluster_stats(labels if labels is not None else self.labels_.tolist())
        
        return {
            'intra_cluster_stats': intra_stats,
            'inter_cluster_distances': inter_distances,
            'silhouette_scores': silhouette,
            'cluster_sizes': cluster_sizes,
            'num_clusters': len(cluster_sizes),
            'total_points': sum(cluster_sizes.values())
        }

    def label_clusters_with_llm(
        self, 
        texts: List[str],
        labels: List[int], 
        agent,
        sample_size: int = 5, 
        iterations: int = 3
    ) -> Dict[int, str]:
        cluster_labels = {}
        unique_clusters = sorted(set(labels))
        
        for cluster_id in unique_clusters:
            if cluster_id == -1:
                cluster_labels[cluster_id] = "Noise/Outliers"
                continue
            
            cluster_texts = [texts[i] for i, label in enumerate(labels) if label == cluster_id]
            
            if len(cluster_texts) <= sample_size:
                label = agent.label_cluster(cluster_texts)
                cluster_labels[cluster_id] = label
                continue
            
            sampled_labels = []
            for _ in range(iterations):
                sample = random.sample(cluster_texts, sample_size)
                label = agent.label_cluster(sample)
                sampled_labels.append(label)
            
            most_common = max(set(sampled_labels), key=sampled_labels.count)
            cluster_labels[cluster_id] = most_common
        
        return cluster_labels

    def get_cluster_stats(self, labels: List[int]) -> Dict[int, int]:
        stats = {}
        for label in set(labels):
            stats[label] = labels.count(label)
        return stats

    @staticmethod
    def estimate_eps_from_data(embeddings: List[List[float]], percentile: float = 95) -> float:
        """
        Estimate DBSCAN eps parameter from data.
        Returns the distance at specified percentile of all pairwise distances.
        """
        from sklearn.neighbors import NearestNeighbors
        X = np.array(embeddings)
        neighbors = NearestNeighbors(n_neighbors=2)
        neighbors.fit(X)
        distances, _ = neighbors.kneighbors(X)
        distances = np.sort(distances[:, 1])
        return float(np.percentile(distances, percentile))