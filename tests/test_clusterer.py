"""
Unit tests for the Clusterer class.
"""
import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clusterer import Clusterer


class TestClusterer(unittest.TestCase):
    """Tests for the Clusterer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample embeddings with clear cluster structure
        self.num_samples = 100
        self.dimension = 10
        self.n_clusters = 3
        
        # Generate clustered data (3 clusters)
        np.random.seed(42)  # For reproducibility
        
        # Create cluster centers
        centers = np.random.randn(self.n_clusters, self.dimension) * 10
        
        # Generate points around centers
        self.embeddings = np.vstack([
            centers[i] + np.random.randn(self.num_samples // self.n_clusters, self.dimension)
            for i in range(self.n_clusters)
        ])
        
        # Create documents
        self.documents = [{"id": i, "text": f"Document {i}"} for i in range(len(self.embeddings))]
    
    def test_kmeans_clustering(self):
        """Test KMeans clustering."""
        clusterer = Clusterer(algorithm="kmeans")
        labels = clusterer.fit(self.embeddings, self.documents, n_clusters=self.n_clusters)
        
        # Check that we have the right number of clusters
        self.assertEqual(len(set(labels)), self.n_clusters)
        
        # Check that all documents are assigned to a cluster
        self.assertEqual(len(labels), len(self.embeddings))
        
        # Check that the silhouette score is reasonable (> 0)
        score = clusterer.evaluate(metric="silhouette")
        self.assertGreater(score, 0)
    
    def test_dbscan_clustering(self):
        """Test DBSCAN clustering."""
        clusterer = Clusterer(algorithm="dbscan")
        labels = clusterer.fit(self.embeddings, self.documents, eps=3.0, min_samples=3)
        
        # Check that all documents are assigned to a cluster or noise (-1)
        self.assertEqual(len(labels), len(self.embeddings))
        
        # Check that we have at least one cluster
        self.assertGreaterEqual(len(set(labels)) - (1 if -1 in labels else 0), 1)
    
    def test_dimensionality_reduction(self):
        """Test dimensionality reduction."""
        clusterer = Clusterer(algorithm="kmeans")
        clusterer.fit(self.embeddings, self.documents, n_clusters=self.n_clusters)
        
        # Reduce to 2D
        reduced = clusterer.reduce_dimensions(n_components=2)
        
        # Check shape
        self.assertEqual(reduced.shape, (len(self.embeddings), 2))


if __name__ == "__main__":
    unittest.main()
