"""Unit tests for the Clusterer class."""
import pytest
import numpy as np
from src.clustering.clusterer import Clusterer


class TestClusterer:
    """Test Clusterer class."""
    
    def test_kmeans_initialization(self):
        """Test KMeans clusterer initialization."""
        clusterer = Clusterer(algorithm="kmeans", n_clusters=3)
        assert clusterer.algorithm == "kmeans"
        assert clusterer.n_clusters == 3
    
    def test_kmeans_requires_n_clusters(self):
        """Test that KMeans requires n_clusters parameter."""
        with pytest.raises(ValueError, match="n_clusters required for kmeans"):
            Clusterer(algorithm="kmeans")
    
    def test_dbscan_initialization(self):
        """Test DBSCAN clusterer initialization."""
        clusterer = Clusterer(algorithm="dbscan", eps=0.5, min_samples=5)
        assert clusterer.algorithm == "dbscan"
        assert clusterer.eps == 0.5
        assert clusterer.min_samples == 5
    
    def test_dbscan_requires_eps(self):
        """Test that DBSCAN requires eps parameter."""
        with pytest.raises(ValueError, match="eps .* required for DBSCAN"):
            Clusterer(algorithm="dbscan")
    
    def test_agglomerative_initialization(self):
        """Test agglomerative clusterer initialization."""
        clusterer = Clusterer(algorithm="agglomerative", n_clusters=4)
        assert clusterer.algorithm == "agglomerative"
        assert clusterer.n_clusters == 4
    
    def test_agglomerative_distance_initialization(self):
        """Test agglomerative with distance threshold initialization."""
        clusterer = Clusterer(algorithm="agglomerative_distance", distance_threshold=1.5)
        assert clusterer.algorithm == "agglomerative_distance"
        assert clusterer.distance_threshold == 1.5
    
    def test_unsupported_algorithm(self):
        """Test that unsupported algorithm raises error."""
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            Clusterer(algorithm="invalid_algorithm")
    
    def test_fit_predict_kmeans(self):
        """Test fit_predict with KMeans on simple data."""
        # Create simple 2D data with 3 clear clusters
        np.random.seed(42)
        embeddings = []
        # Cluster 1: around (0, 0)
        embeddings.extend([[0 + np.random.randn()*0.1, 0 + np.random.randn()*0.1] for _ in range(10)])
        # Cluster 2: around (5, 5)
        embeddings.extend([[5 + np.random.randn()*0.1, 5 + np.random.randn()*0.1] for _ in range(10)])
        # Cluster 3: around (10, 0)
        embeddings.extend([[10 + np.random.randn()*0.1, 0 + np.random.randn()*0.1] for _ in range(10)])
        
        clusterer = Clusterer(algorithm="kmeans", n_clusters=3)
        labels = clusterer.fit_predict(embeddings)
        
        assert len(labels) == 30
        assert len(set(labels)) == 3  # Should have 3 clusters
        assert all(isinstance(label, int) for label in labels)
    
    def test_get_cluster_centers_kmeans(self):
        """Test getting cluster centers for KMeans."""
        np.random.seed(42)
        embeddings = [[0, 0]] * 10 + [[5, 5]] * 10
        
        clusterer = Clusterer(algorithm="kmeans", n_clusters=2)
        clusterer.fit_predict(embeddings)
        centers = clusterer.get_cluster_centers()
        
        assert centers is not None
        assert centers.shape[0] == 2  # 2 clusters
        assert centers.shape[1] == 2  # 2 dimensions
    
    def test_get_cluster_stats(self):
        """Test getting cluster statistics."""
        clusterer = Clusterer(algorithm="kmeans", n_clusters=2)
        labels = [0, 0, 0, 1, 1, 1, 1, 0]
        stats = clusterer.get_cluster_stats(labels)
        
        assert stats[0] == 4
        assert stats[1] == 4
    
    def test_calculate_intra_cluster_distances(self):
        """Test calculating intra-cluster distances."""
        np.random.seed(42)
        embeddings = [[0, 0]] * 5 + [[10, 10]] * 5
        
        clusterer = Clusterer(algorithm="kmeans", n_clusters=2)
        labels = clusterer.fit_predict(embeddings)
        stats = clusterer.calculate_intra_cluster_distances()
        
        assert len(stats) == 2
        for cluster_id, cluster_stats in stats.items():
            assert 'mean_distance' in cluster_stats
            assert 'max_distance' in cluster_stats
            assert 'std_distance' in cluster_stats
            assert 'variance' in cluster_stats
            assert 'cluster_size' in cluster_stats
            assert cluster_stats['cluster_size'] == 5
    
    def test_calculate_silhouette_scores(self):
        """Test calculating silhouette scores."""
        np.random.seed(42)
        embeddings = [[0, 0]] * 10 + [[10, 10]] * 10
        
        clusterer = Clusterer(algorithm="kmeans", n_clusters=2)
        clusterer.fit_predict(embeddings)
        scores = clusterer.calculate_silhouette_scores()
        
        assert 'overall_score' in scores
        assert 'cluster_scores' in scores
        assert scores['overall_score'] is not None
        assert len(scores['cluster_scores']) == 2
    
    def test_estimate_eps_from_data(self):
        """Test estimating DBSCAN eps parameter."""
        np.random.seed(42)
        embeddings = [[i, i] for i in range(20)]
        
        eps = Clusterer.estimate_eps_from_data(embeddings, percentile=95)
        
        assert isinstance(eps, float)
        assert eps > 0
