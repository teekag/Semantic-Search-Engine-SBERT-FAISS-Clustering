"""
Unit tests for the VectorStore class.
"""
import unittest
import numpy as np
import sys
import os
import tempfile
import shutil
import pickle

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vector_store import VectorStore


class TestVectorStore(unittest.TestCase):
    """Tests for the VectorStore class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dimension = 384  # Standard dimension for all-MiniLM-L6-v2
        self.vector_store = VectorStore(dimension=self.dimension, index_type="Flat", metric="cosine")
        
        # Create sample embeddings and documents
        self.num_samples = 10
        self.embeddings = np.random.random((self.num_samples, self.dimension)).astype(np.float32)
        self.documents = [{"id": i, "text": f"Document {i}"} for i in range(self.num_samples)]
        
        # Add to index
        self.vector_store.add(self.embeddings, self.documents)
        
        # Temp directory for save/load tests
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)
    
    def test_index_creation(self):
        """Test that the index is created correctly."""
        self.assertIsNotNone(self.vector_store.index)
        self.assertEqual(len(self.vector_store.documents), self.num_samples)
    
    def test_search(self):
        """Test search functionality."""
        # Create a query embedding (just use the first document's embedding)
        query_embedding = self.embeddings[0].copy()
        
        # Search
        results = self.vector_store.search(query_embedding, k=3)
        
        # Check results
        self.assertIn("documents", results)
        self.assertIn("similarities", results)
        self.assertIn("indices", results)
        self.assertEqual(len(results["documents"]), 3)
        
        # The first result should be the query document itself with high similarity
        self.assertEqual(results["documents"][0]["id"], 0)
        self.assertGreater(results["similarities"][0], 0.9)  # Should be close to 1.0
    
    def test_save_load(self):
        """Test saving and loading the index."""
        # Save the index
        filename = "test_vector_store"
        save_path = self.vector_store.save(self.temp_dir, filename=filename)
        
        # Verify that the files were created
        index_path = os.path.join(self.temp_dir, f"{filename}.index")
        metadata_path = os.path.join(self.temp_dir, f"{filename}.pkl")
        self.assertTrue(os.path.exists(index_path))
        self.assertTrue(os.path.exists(metadata_path))
        
        # Verify the metadata contains the documents
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        self.assertIn("documents", metadata)
        self.assertEqual(len(metadata["documents"]), self.num_samples)
        
        # Load the index directly into the existing vector store
        self.vector_store.load(self.temp_dir, filename=filename)
        
        # Check that the loaded index has the same documents
        self.assertEqual(len(self.vector_store.documents), self.num_samples)
        
        # Test search with the loaded index
        query_embedding = self.embeddings[0].copy()
        results = self.vector_store.search(query_embedding, k=1)
        
        # The first result should be the query document
        self.assertEqual(results["documents"][0]["id"], 0)


if __name__ == "__main__":
    unittest.main()
