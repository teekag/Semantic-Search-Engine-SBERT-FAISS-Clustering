"""
Clusterer module for the Semantic Search Engine.

This module provides functionality to perform unsupervised clustering
on document embeddings to discover semantic groupings and structure.
"""

import os
import pickle
from typing import List, Dict, Union, Tuple, Optional, Any
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import umap
import hdbscan


class Clusterer:
    """
    A class for clustering document embeddings to discover semantic groupings.
    
    This class provides methods for applying various clustering algorithms
    to document embeddings and evaluating the resulting clusters.
    """
    
    def __init__(
        self, 
        algorithm: str = "kmeans",
        random_state: int = 42
    ):
        """
        Initialize the Clusterer with a specific clustering algorithm.
        
        Args:
            algorithm: Clustering algorithm to use ("kmeans", "dbscan", "hdbscan", "agglomerative")
            random_state: Random seed for reproducibility
        """
        self.algorithm = algorithm
        self.random_state = random_state
        self.model = None
        self.labels = None
        self.embeddings = None
        self.documents = None
        self.reduced_embeddings = None  # For visualization (UMAP)
        
    def fit(
        self, 
        embeddings: np.ndarray, 
        documents: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Fit the clustering model to the embeddings.
        
        Args:
            embeddings: Document embeddings to cluster
            documents: Optional list of document dictionaries
            **kwargs: Additional parameters for the clustering algorithm
            
        Returns:
            numpy.ndarray: Cluster labels for each document
        """
        self.embeddings = embeddings
        self.documents = documents if documents is not None else []
        
        # Create and fit the clustering model
        if self.algorithm == "kmeans":
            # Default parameters
            n_clusters = kwargs.get("n_clusters", 5)
            
            self.model = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init=10
            )
            
        elif self.algorithm == "dbscan":
            # Default parameters
            eps = kwargs.get("eps", 0.5)
            min_samples = kwargs.get("min_samples", 5)
            
            self.model = DBSCAN(
                eps=eps,
                min_samples=min_samples
            )
            
        elif self.algorithm == "hdbscan":
            # Default parameters
            min_cluster_size = kwargs.get("min_cluster_size", 5)
            min_samples = kwargs.get("min_samples", None)
            
            self.model = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                gen_min_span_tree=True,
                prediction_data=True
            )
            
        elif self.algorithm == "agglomerative":
            # Default parameters
            n_clusters = kwargs.get("n_clusters", 5)
            linkage = kwargs.get("linkage", "ward")
            
            self.model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage
            )
            
        else:
            raise ValueError(f"Unsupported clustering algorithm: {self.algorithm}")
        
        # Fit the model
        self.labels = self.model.fit_predict(embeddings)
        
        print(f"Clustered {len(embeddings)} documents into {len(set(self.labels))} clusters")
        return self.labels
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new embeddings.
        
        Args:
            embeddings: New document embeddings to cluster
            
        Returns:
            numpy.ndarray: Cluster labels for each document
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet")
            
        if self.algorithm == "kmeans" or self.algorithm == "agglomerative":
            # These algorithms support predict
            return self.model.predict(embeddings)
        elif self.algorithm == "dbscan":
            # DBSCAN doesn't have a predict method, so we use a nearest neighbors approach
            from sklearn.neighbors import NearestNeighbors
            
            # Find the nearest neighbor in the training set
            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(self.embeddings)
            
            # Get the indices of the nearest neighbors
            indices = nn.kneighbors(embeddings, return_distance=False)
            
            # Return the labels of the nearest neighbors
            return self.labels[indices.flatten()]
        elif self.algorithm == "hdbscan":
            # HDBSCAN has a predict method if prediction_data=True
            return hdbscan.approximate_predict(self.model, embeddings)[0]
        else:
            raise ValueError(f"Prediction not supported for {self.algorithm}")
    
    def evaluate(self, metric: str = "silhouette") -> float:
        """
        Evaluate the quality of the clustering.
        
        Args:
            metric: Evaluation metric to use ("silhouette")
            
        Returns:
            float: Evaluation score
        """
        if self.labels is None or self.embeddings is None:
            raise ValueError("Model has not been fitted yet")
            
        if metric == "silhouette":
            # Filter out noise points (label -1) for silhouette score
            mask = self.labels != -1
            if sum(mask) <= 1 or len(set(self.labels[mask])) <= 1:
                return 0.0  # Not enough valid clusters for silhouette score
                
            return silhouette_score(self.embeddings[mask], self.labels[mask])
        else:
            raise ValueError(f"Unsupported evaluation metric: {metric}")
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """
        Get information about the clusters.
        
        Returns:
            Dict: Information about the clusters
        """
        if self.labels is None:
            raise ValueError("Model has not been fitted yet")
            
        # Count documents per cluster
        unique_labels = set(self.labels)
        cluster_counts = {label: sum(self.labels == label) for label in unique_labels}
        
        # Calculate cluster centroids for algorithms that support it
        if self.algorithm == "kmeans":
            centroids = self.model.cluster_centers_
        else:
            # For other algorithms, calculate centroids manually
            centroids = {}
            for label in unique_labels:
                if label == -1:  # Skip noise points
                    continue
                mask = self.labels == label
                if sum(mask) > 0:
                    centroids[label] = np.mean(self.embeddings[mask], axis=0)
        
        return {
            "algorithm": self.algorithm,
            "num_clusters": len(unique_labels),
            "cluster_counts": cluster_counts,
            "centroids": centroids if isinstance(centroids, dict) else centroids.tolist() if hasattr(centroids, "tolist") else None
        }
    
    def get_documents_by_cluster(self) -> Dict[int, List[Dict[str, Any]]]:
        """
        Get documents grouped by cluster.
        
        Returns:
            Dict: Mapping from cluster labels to lists of documents
        """
        if self.labels is None or self.documents is None or len(self.documents) == 0:
            raise ValueError("Model has not been fitted or no documents provided")
            
        result = {}
        for i, label in enumerate(self.labels):
            if label not in result:
                result[label] = []
            if i < len(self.documents):
                result[label].append(self.documents[i])
        
        return result
    
    def reduce_dimensions(self, n_components: int = 2, **kwargs) -> np.ndarray:
        """
        Reduce the dimensionality of the embeddings for visualization.
        
        Args:
            n_components: Number of dimensions to reduce to
            **kwargs: Additional parameters for UMAP
            
        Returns:
            numpy.ndarray: Reduced embeddings
        """
        if self.embeddings is None:
            raise ValueError("No embeddings available")
            
        # Default UMAP parameters
        n_neighbors = kwargs.get("n_neighbors", 15)
        min_dist = kwargs.get("min_dist", 0.1)
        
        # Create and fit UMAP
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=self.random_state
        )
        
        self.reduced_embeddings = reducer.fit_transform(self.embeddings)
        return self.reduced_embeddings
    
    def save(self, directory: str, filename: str = "clusterer") -> str:
        """
        Save the clusterer to disk.
        
        Args:
            directory: Directory to save the files in
            filename: Base filename to use
            
        Returns:
            str: Path to the saved file
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save the model and metadata
        save_path = os.path.join(directory, f"{filename}.pkl")
        
        # Prepare data to save
        data = {
            "algorithm": self.algorithm,
            "random_state": self.random_state,
            "model": self.model,
            "labels": self.labels,
            "reduced_embeddings": self.reduced_embeddings
        }
        
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        
        print(f"Saved clusterer to {save_path}")
        return save_path
    
    @classmethod
    def load(cls, path: str) -> "Clusterer":
        """
        Load a clusterer from disk.
        
        Args:
            path: Path to the saved file
            
        Returns:
            Clusterer: Loaded clusterer
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        # Create a new instance
        clusterer = cls(
            algorithm=data["algorithm"],
            random_state=data["random_state"]
        )
        
        # Set the attributes
        clusterer.model = data["model"]
        clusterer.labels = data["labels"]
        clusterer.reduced_embeddings = data["reduced_embeddings"]
        
        print(f"Loaded clusterer from {path}")
        return clusterer


if __name__ == "__main__":
    # Simple demonstration of the Clusterer class
    import numpy as np
    
    # Create random embeddings for testing
    dimension = 384  # Typical for many SBERT models
    num_docs = 1000
    embeddings = np.random.rand(num_docs, dimension).astype(np.float32)
    
    # Create sample documents
    documents = [{"id": i, "text": f"Document {i}"} for i in range(num_docs)]
    
    # Create and fit the clusterer
    clusterer = Clusterer(algorithm="kmeans")
    labels = clusterer.fit(embeddings, documents, n_clusters=10)
    
    # Evaluate the clustering
    score = clusterer.evaluate()
    print(f"Silhouette score: {score:.4f}")
    
    # Get cluster info
    cluster_info = clusterer.get_cluster_info()
    print(f"\nCluster counts:")
    for label, count in cluster_info["cluster_counts"].items():
        print(f"Cluster {label}: {count} documents")
    
    # Reduce dimensions for visualization
    reduced = clusterer.reduce_dimensions(n_components=2)
    print(f"\nReduced embeddings shape: {reduced.shape}")
