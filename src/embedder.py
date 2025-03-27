"""
Embedder module for the Semantic Search Engine.

This module provides functionality to convert text into embeddings using
Sentence-BERT (SBERT) models. These embeddings capture semantic meaning
and can be used for similarity search and clustering.
"""

import os
from typing import List, Dict, Union, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch


class Embedder:
    """
    A class for generating embeddings from text using SBERT models.
    
    This class handles the loading of pre-trained SBERT models and
    conversion of text documents into dense vector representations.
    """
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2", 
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the Embedder with a specific SBERT model.
        
        Args:
            model_name: Name of the SBERT model to use (default: "all-MiniLM-L6-v2")
            device: Device to run the model on ("cpu", "cuda", etc.). If None, will use CUDA if available.
            cache_dir: Directory to cache the downloaded models
        """
        self.model_name = model_name
        
        # Set device (use CUDA if available and not explicitly set to CPU)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Load the model
        self.model = SentenceTransformer(model_name, device=self.device, cache_folder=cache_dir)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        print(f"Loaded {model_name} with dimension {self.embedding_dim} on {self.device}")
        
    def embed_texts(self, texts: List[str], batch_size: int = 32, show_progress_bar: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of text documents.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing
            show_progress_bar: Whether to display a progress bar
            
        Returns:
            numpy.ndarray: Array of embeddings with shape (len(texts), embedding_dim)
        """
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True
        )
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query string.
        
        Args:
            query: Text string to embed
            
        Returns:
            numpy.ndarray: Query embedding vector
        """
        return self.model.encode(query, convert_to_numpy=True, show_progress_bar=False)
    
    def embed_documents(self, documents: List[Dict[str, str]], text_field: str = "text", 
                        batch_size: int = 32) -> Dict[str, Union[List[Dict], np.ndarray]]:
        """
        Generate embeddings for a list of document dictionaries.
        
        Args:
            documents: List of document dictionaries
            text_field: Key in the document dictionaries that contains the text to embed
            batch_size: Batch size for processing
            
        Returns:
            Dict containing the original documents and their embeddings
        """
        texts = [doc[text_field] for doc in documents]
        embeddings = self.embed_texts(texts, batch_size=batch_size)
        
        return {
            "documents": documents,
            "embeddings": embeddings
        }
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings produced by this model.
        
        Returns:
            int: Dimension of the embedding vectors
        """
        return self.embedding_dim
    
    def get_model_info(self) -> Dict[str, str]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict: Information about the model
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": str(self.embedding_dim),
            "device": self.device
        }


if __name__ == "__main__":
    # Simple demonstration of the Embedder class
    embedder = Embedder()
    
    # Example texts
    texts = [
        "This is a sample document about artificial intelligence.",
        "Machine learning models require large amounts of data.",
        "Natural language processing helps computers understand human language.",
        "Vector embeddings capture semantic meaning in dense representations."
    ]
    
    # Generate embeddings
    embeddings = embedder.embed_texts(texts)
    
    print(f"Generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
    print(f"Sample embedding (first 5 dimensions): {embeddings[0][:5]}")
    
    # Calculate cosine similarities between the first text and all others
    from sklearn.metrics.pairwise import cosine_similarity
    
    similarities = cosine_similarity([embeddings[0]], embeddings)[0]
    
    print("\nSimilarities to first text:")
    for i, (text, sim) in enumerate(zip(texts, similarities)):
        print(f"{i}: {sim:.4f} - {text}")
