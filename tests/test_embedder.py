"""
Unit tests for the Embedder class.
"""
import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedder import Embedder


class TestEmbedder(unittest.TestCase):
    """Tests for the Embedder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.embedder = Embedder(model_name="all-MiniLM-L6-v2")
        self.test_texts = [
            "This is a test document.",
            "Another document for testing."
        ]
    
    def test_embedding_shape(self):
        """Test that embeddings have the correct shape."""
        embeddings = self.embedder.embed_texts(self.test_texts)
        
        # Check shape
        self.assertEqual(embeddings.shape[0], len(self.test_texts))
        self.assertEqual(embeddings.shape[1], self.embedder.get_embedding_dimension())
    
    def test_query_embedding(self):
        """Test embedding a single query."""
        query = "Test query"
        embedding = self.embedder.embed_query(query)
        
        # Check shape
        self.assertEqual(embedding.shape[0], self.embedder.get_embedding_dimension())
    
    def test_model_info(self):
        """Test getting model info."""
        info = self.embedder.get_model_info()
        
        # Check keys
        self.assertIn("model_name", info)
        self.assertIn("embedding_dimension", info)
        self.assertIn("device", info)


if __name__ == "__main__":
    unittest.main()
