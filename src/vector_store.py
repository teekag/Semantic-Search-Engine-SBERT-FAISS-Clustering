"""
Vector Store module for the Semantic Search Engine.

This module provides functionality to index and search vector embeddings
using FAISS (Facebook AI Similarity Search) for efficient similarity search.
"""

import os
import pickle
from typing import List, Dict, Union, Tuple, Optional, Any
import numpy as np
import faiss


class VectorStore:
    """
    A class for indexing and searching vector embeddings using FAISS.
    
    This class handles the creation, storage, and querying of FAISS indices
    for fast similarity search over document embeddings.
    """
    
    def __init__(
        self, 
        dimension: int,
        index_type: str = "Flat",
        metric: str = "cosine",
        nlist: int = 100,  # For IVF indices
        nprobe: int = 10,  # For IVF indices
        use_gpu: bool = False
    ):
        """
        Initialize the VectorStore with a specific FAISS index configuration.
        
        Args:
            dimension: Dimensionality of the vectors to be indexed
            index_type: Type of FAISS index to use ("Flat", "IVF", "HNSW", etc.)
            metric: Distance metric to use ("cosine", "l2", "ip")
            nlist: Number of clusters for IVF indices
            nprobe: Number of clusters to probe during search for IVF indices
            use_gpu: Whether to use GPU acceleration if available
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.nlist = nlist
        self.nprobe = nprobe
        self.use_gpu = use_gpu
        self.index = None
        self.documents = []
        
        # Create the index
        self._create_index()
        
    def _create_index(self):
        """Create the FAISS index based on the specified configuration."""
        # Determine the base index type based on the metric
        if self.metric == "cosine":
            # For cosine similarity, we normalize vectors and use inner product
            base_index = faiss.IndexFlatIP(self.dimension)
        elif self.metric == "l2":
            base_index = faiss.IndexFlatL2(self.dimension)
        elif self.metric == "ip":
            base_index = faiss.IndexFlatIP(self.dimension)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        
        # Create the specific index type
        if self.index_type == "Flat":
            self.index = base_index
        elif self.index_type == "IVF":
            # IVF requires training, so we create it but don't train yet
            self.index = faiss.IndexIVFFlat(base_index, self.dimension, self.nlist, 
                                           faiss.METRIC_INNER_PRODUCT if self.metric == "cosine" or self.metric == "ip" 
                                           else faiss.METRIC_L2)
            self.index.nprobe = self.nprobe
        elif self.index_type == "HNSW":
            # HNSW parameters: M (16 is default)
            M = 16
            self.index = faiss.IndexHNSWFlat(self.dimension, M, 
                                           faiss.METRIC_INNER_PRODUCT if self.metric == "cosine" or self.metric == "ip" 
                                           else faiss.METRIC_L2)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        # Use GPU if requested and available
        if self.use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
            print(f"Using GPU acceleration for FAISS index")
        
        print(f"Created {self.index_type} index with {self.metric} metric and dimension {self.dimension}")
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        Normalize vectors for cosine similarity.
        
        Args:
            vectors: Input vectors to normalize
            
        Returns:
            numpy.ndarray: Normalized vectors
        """
        if self.metric == "cosine":
            # L2 normalization for cosine similarity
            faiss.normalize_L2(vectors)
        return vectors
    
    def add(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]) -> None:
        """
        Add embeddings and their corresponding documents to the index.
        
        Args:
            embeddings: Array of embeddings with shape (n, dimension)
            documents: List of document dictionaries corresponding to the embeddings
        """
        if len(embeddings) != len(documents):
            raise ValueError("Number of embeddings must match number of documents")
        
        if len(embeddings) == 0:
            return
            
        # Store the current size for ID assignment
        current_size = len(self.documents)
        
        # Normalize vectors if using cosine similarity
        vectors = self._normalize_vectors(embeddings.copy())
        
        # Train the index if it's an IVF index and not yet trained
        if self.index_type == "IVF" and not self.index.is_trained:
            if len(vectors) < self.nlist:
                print(f"Warning: Training with {len(vectors)} vectors, which is less than nlist={self.nlist}")
            self.index.train(vectors)
        
        # Add vectors to the index
        self.index.add(vectors)
        
        # Store the documents
        self.documents.extend(documents)
        
        print(f"Added {len(embeddings)} vectors to index. Total vectors: {len(self.documents)}")
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> Dict[str, Union[List[Dict], List[float], List[int]]]:
        """
        Search for the k most similar documents to the query embedding.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            
        Returns:
            Dict containing the search results (documents, distances, indices)
        """
        if self.index is None or len(self.documents) == 0:
            return {"documents": [], "distances": [], "indices": []}
        
        # Ensure query is 2D array with shape (1, dimension)
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize query if using cosine similarity
        query_vector = self._normalize_vectors(query_embedding.copy())
        
        # Limit k to the number of documents
        k = min(k, len(self.documents))
        
        # Search the index
        distances, indices = self.index.search(query_vector, k)
        
        # Convert to lists for the first query
        distances = distances[0].tolist()
        indices = indices[0].tolist()
        
        # Get the corresponding documents
        result_documents = [self.documents[i] for i in indices if i >= 0 and i < len(self.documents)]
        
        # For cosine similarity, convert distances to similarities (1 - distance)
        if self.metric == "cosine" or self.metric == "ip":
            # For inner product with normalized vectors, similarity is already in [-1, 1]
            # Convert to [0, 1] range
            similarities = [(d + 1) / 2 for d in distances]
            return {
                "documents": result_documents,
                "similarities": similarities,
                "indices": indices
            }
        else:
            # For L2 distance, smaller is better
            return {
                "documents": result_documents,
                "distances": distances,
                "indices": indices
            }
    
    def save(self, directory: str, filename: str = "vector_store") -> str:
        """
        Save the vector store to disk.
        
        Args:
            directory: Directory to save the files in
            filename: Base filename to use
            
        Returns:
            str: Path to the saved index file
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save the index
        index_path = os.path.join(directory, f"{filename}.index")
        
        # If using GPU, convert back to CPU for saving
        if self.use_gpu and faiss.get_num_gpus() > 0:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, index_path)
        else:
            faiss.write_index(self.index, index_path)
        
        # Save the documents and metadata
        metadata = {
            "documents": self.documents,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "nlist": self.nlist,
            "nprobe": self.nprobe
        }
        
        metadata_path = os.path.join(directory, f"{filename}.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        
        print(f"Saved vector store to {index_path} and {metadata_path}")
        return index_path
    
    @classmethod
    def load(cls, directory: str, filename: str = "vector_store", use_gpu: bool = False) -> "VectorStore":
        """
        Load a vector store from disk.
        
        Args:
            directory: Directory containing the saved files
            filename: Base filename used when saving
            use_gpu: Whether to use GPU acceleration
            
        Returns:
            VectorStore: Loaded vector store
        """
        # Load metadata
        metadata_path = os.path.join(directory, f"{filename}.pkl")
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        
        # Create a new instance
        vector_store = cls(
            dimension=metadata["dimension"],
            index_type=metadata["index_type"],
            metric=metadata["metric"],
            nlist=metadata["nlist"],
            nprobe=metadata["nprobe"],
            use_gpu=use_gpu
        )
        
        # Load the index
        index_path = os.path.join(directory, f"{filename}.index")
        vector_store.index = faiss.read_index(index_path)
        
        # Move to GPU if requested
        if use_gpu and faiss.get_num_gpus() > 0:
            vector_store.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, vector_store.index)
        
        # Set the documents
        vector_store.documents = metadata["documents"]
        
        print(f"Loaded vector store from {index_path} with {len(vector_store.documents)} documents")
        return vector_store
    
    def get_index_info(self) -> Dict[str, str]:
        """
        Get information about the index.
        
        Returns:
            Dict: Information about the index
        """
        return {
            "index_type": self.index_type,
            "metric": self.metric,
            "dimension": str(self.dimension),
            "num_vectors": str(len(self.documents)),
            "nlist": str(self.nlist) if self.index_type == "IVF" else "N/A",
            "nprobe": str(self.nprobe) if self.index_type == "IVF" else "N/A",
            "using_gpu": str(self.use_gpu)
        }


if __name__ == "__main__":
    # Simple demonstration of the VectorStore class
    import numpy as np
    
    # Create random embeddings for testing
    dimension = 384  # Typical for many SBERT models
    num_docs = 1000
    embeddings = np.random.rand(num_docs, dimension).astype(np.float32)
    
    # Create sample documents
    documents = [{"id": i, "text": f"Document {i}"} for i in range(num_docs)]
    
    # Create and populate the vector store
    vector_store = VectorStore(dimension=dimension, index_type="Flat", metric="cosine")
    vector_store.add(embeddings, documents)
    
    # Create a query embedding
    query_embedding = np.random.rand(dimension).astype(np.float32)
    
    # Search
    results = vector_store.search(query_embedding, k=5)
    
    print("\nSearch results:")
    for i, (doc, sim) in enumerate(zip(results["documents"], results["similarities"])):
        print(f"{i+1}: {sim:.4f} - Document {doc['id']}")
