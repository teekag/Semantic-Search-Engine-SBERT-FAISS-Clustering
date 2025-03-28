# Semantic Search Engine Documentation

## Overview

This document provides detailed information about the Semantic Search Engine project, including architecture, components, usage examples, and technical details.

## Table of Contents

1. [Architecture](#architecture)
2. [Core Components](#core-components)
3. [Installation and Setup](#installation-and-setup)
4. [Usage Examples](#usage-examples)
5. [API Reference](#api-reference)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)

## Architecture

The Semantic Search Engine follows a modular architecture with the following key components:

### Data Flow

1. **Document Ingestion**: Raw text documents are loaded from various sources.
2. **Embedding Generation**: Documents are converted to dense vector embeddings using SBERT.
3. **Vector Indexing**: Embeddings are indexed using FAISS for efficient similarity search.
4. **Query Processing**: User queries are embedded and searched against the index.
5. **Result Ranking**: Results are ranked by similarity and returned to the user.
6. **Clustering and Visualization**: Document embeddings are clustered and visualized for exploration.

### Technology Stack

- **Python 3.8+**: Core programming language
- **SBERT**: For generating semantic embeddings
- **FAISS**: For efficient vector similarity search
- **scikit-learn**: For clustering and evaluation metrics
- **FastAPI**: For REST API implementation
- **Matplotlib/Seaborn**: For visualization
- **NumPy/Pandas**: For data processing

## Core Components

### Embedder (`src/embedder.py`)

The Embedder class is responsible for converting text to vector embeddings using SBERT models.

#### Key Methods:

- `__init__(model_name="all-MiniLM-L6-v2", use_gpu=False)`: Initialize with a specific SBERT model
- `embed_texts(texts, batch_size=32, show_progress_bar=True)`: Embed a batch of texts
- `embed_query(query)`: Embed a single query text
- `get_embedding_dimension()`: Get the dimension of the embeddings
- `get_model_info()`: Get information about the loaded model

### Vector Store (`src/vector_store.py`)

The VectorStore class handles the creation, management, and querying of FAISS indices.

#### Key Methods:

- `__init__(dimension, index_type="Flat", metric="cosine", nlist=100, nprobe=10, use_gpu=False)`: Initialize with specific index configuration
- `add(embeddings, documents)`: Add embeddings and documents to the index
- `search(query_embedding, k=10)`: Search for similar documents
- `save(directory, filename="vector_store")`: Save the index to disk
- `load(directory, filename="vector_store", use_gpu=False)`: Load an index from disk
- `get_index_info()`: Get information about the index

### Clusterer (`src/clusterer.py`)

The Clusterer class provides functionality for clustering document embeddings and visualizing the results.

#### Key Methods:

- `__init__(algorithm="kmeans", n_clusters=5, random_state=42)`: Initialize with a specific clustering algorithm
- `fit(embeddings, documents=None, **kwargs)`: Fit the clustering model to embeddings
- `reduce_dimensions(embeddings, method="tsne", n_components=2, **kwargs)`: Reduce dimensionality for visualization
- `evaluate(embeddings, true_labels)`: Evaluate clustering quality
- `visualize(embeddings, labels=None, method="tsne", **kwargs)`: Visualize clusters

### API (`src/api.py`)

The FastAPI application that exposes the search engine functionality through a REST API.

#### Key Endpoints:

- `POST /search`: Search for documents similar to a query
- `GET /clusters`: Get cluster information and visualization data
- `GET /status`: Check API status
- `GET /categories`: List all document categories
- `GET /document/{document_id}`: Get a specific document

## Installation and Setup

### Prerequisites

- Python 3.8+
- pip
- Virtual environment (recommended)

### Step-by-Step Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/semantic-search-engine.git
cd semantic-search-engine
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Configuration

The project can be configured through various parameters in the component classes:

- **Embedder**: Choose different SBERT models by changing the `model_name` parameter
- **VectorStore**: Configure index type, metric, and other FAISS parameters
- **Clusterer**: Select different clustering algorithms and parameters

## Usage Examples

### Basic Search Pipeline

```python
from src.embedder import Embedder
from src.vector_store import VectorStore

# Initialize components
embedder = Embedder(model_name="all-MiniLM-L6-v2")
vector_store = VectorStore(dimension=embedder.get_embedding_dimension())

# Prepare documents
documents = [
    {"id": 1, "text": "This is a document about artificial intelligence."},
    {"id": 2, "text": "Machine learning is a subset of AI."},
    {"id": 3, "text": "Python is a popular programming language."}
]

# Embed documents
texts = [doc["text"] for doc in documents]
embeddings = embedder.embed_texts(texts)

# Add to index
vector_store.add(embeddings, documents)

# Search
query = "Tell me about AI"
query_embedding = embedder.embed_query(query)
results = vector_store.search(query_embedding, k=2)

# Print results
for i, doc in enumerate(results["documents"]):
    print(f"{i+1}. {doc['text']} (Similarity: {results['similarities'][i]:.4f})")
```

### Clustering Example

```python
from src.embedder import Embedder
from src.clusterer import Clusterer
import matplotlib.pyplot as plt

# Initialize components
embedder = Embedder()
clusterer = Clusterer(algorithm="kmeans", n_clusters=3)

# Prepare documents
documents = [
    {"id": 1, "text": "This is a document about artificial intelligence."},
    {"id": 2, "text": "Machine learning is a subset of AI."},
    {"id": 3, "text": "Python is a popular programming language."},
    {"id": 4, "text": "Java is another programming language."},
    {"id": 5, "text": "Deep learning is revolutionizing AI."}
]

# Embed documents
texts = [doc["text"] for doc in documents]
embeddings = embedder.embed_texts(texts)

# Cluster embeddings
labels = clusterer.fit(embeddings)

# Reduce dimensions for visualization
reduced_embeddings = clusterer.reduce_dimensions(embeddings, method="tsne")

# Visualize
plt.figure(figsize=(10, 8))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels)
plt.title("Document Clusters")
plt.savefig("clusters.png")
```

### Using the API

Start the API server:
```bash
uvicorn src.api:app --reload
```

Example API requests:

```python
import requests
import json

# Search endpoint
search_url = "http://localhost:8000/search"
search_payload = {
    "query": "How does machine learning work?",
    "top_k": 3
}
response = requests.post(search_url, json=search_payload)
results = response.json()
print(json.dumps(results, indent=2))

# Clusters endpoint
clusters_url = "http://localhost:8000/clusters?include_embeddings=true"
response = requests.get(clusters_url)
clusters = response.json()
print(f"Number of clusters: {clusters['num_clusters']}")
print(f"Silhouette score: {clusters['silhouette_score']}")
```

## API Reference

### Search Endpoint

```
POST /search
```

Request Body:
```json
{
  "query": "string",
  "top_k": integer (default: 5)
}
```

Response:
```json
{
  "query": "string",
  "results": [
    {
      "document_id": integer,
      "text": "string",
      "category": "string",
      "similarity": float
    }
  ]
}
```

### Clusters Endpoint

```
GET /clusters
```

Query Parameters:
- `include_embeddings` (boolean, optional): Whether to include reduced embeddings in the response

Response:
```json
{
  "num_clusters": integer,
  "silhouette_score": float,
  "cluster_counts": {
    "cluster_id": count
  },
  "cluster_categories": {
    "cluster_id": {
      "category": count
    }
  },
  "reduced_embeddings": [
    [x, y]
  ]
}
```

### Status Endpoint

```
GET /status
```

Response:
```json
{
  "status": "string",
  "version": "string",
  "model_info": {
    "name": "string",
    "dimension": integer
  },
  "index_info": {
    "index_type": "string",
    "metric": "string",
    "dimension": "string",
    "num_vectors": "string"
  }
}
```

### Categories Endpoint

```
GET /categories
```

Response:
```json
{
  "categories": [
    "string"
  ],
  "counts": {
    "category": count
  }
}
```

### Document Endpoint

```
GET /document/{document_id}
```

Path Parameters:
- `document_id` (integer, required): ID of the document to retrieve

Response:
```json
{
  "id": integer,
  "text": "string",
  "category": "string"
}
```

## Evaluation Metrics

The Semantic Search Engine includes several metrics for evaluating search quality:

### Precision@k

Measures the proportion of relevant documents among the top-k retrieved documents.

```python
def precision_at_k(results, k, relevant_category):
    """Calculate precision@k for a query."""
    relevant_count = sum(1 for doc in results["documents"][:k] 
                         if doc["category"] == relevant_category)
    return relevant_count / k
```

### Recall@k

Measures the proportion of relevant documents that are retrieved among the top-k results.

```python
def recall_at_k(results, k, relevant_category, total_relevant):
    """Calculate recall@k for a query."""
    relevant_count = sum(1 for doc in results["documents"][:k] 
                         if doc["category"] == relevant_category)
    return relevant_count / total_relevant
```

### Mean Reciprocal Rank (MRR)

Measures the rank of the first relevant document in the results.

```python
def mean_reciprocal_rank(results, relevant_category):
    """Calculate MRR for a query."""
    for i, doc in enumerate(results["documents"]):
        if doc["category"] == relevant_category:
            return 1.0 / (i + 1)
    return 0.0
```

## Performance Optimization

### Index Types

FAISS offers different index types with different performance characteristics:

- **Flat**: Exact search, highest accuracy but slower for large datasets
- **IVF**: Approximate search using inverted file index, faster but less accurate
- **HNSW**: Hierarchical Navigable Small World graphs, good balance of speed and accuracy

### GPU Acceleration

Both SBERT and FAISS support GPU acceleration for faster processing:

```python
# Enable GPU for embedder
embedder = Embedder(use_gpu=True)

# Enable GPU for vector store
vector_store = VectorStore(dimension=384, use_gpu=True)
```

### Batch Processing

For large document collections, use batch processing to reduce memory usage:

```python
batch_size = 1000
for i in range(0, len(documents), batch_size):
    batch_docs = documents[i:i+batch_size]
    batch_texts = [doc["text"] for doc in batch_docs]
    batch_embeddings = embedder.embed_texts(batch_texts)
    vector_store.add(batch_embeddings, batch_docs)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU instead
2. **Slow Search Performance**: Consider using an approximate index like IVF or HNSW
3. **Poor Search Results**: Try a different SBERT model or fine-tune on domain-specific data
4. **API Connection Issues**: Check that the server is running and the port is accessible

### Debugging Tips

- Enable verbose logging in the embedder: `embedder = Embedder(verbose=True)`
- Check FAISS index information: `print(vector_store.get_index_info())`
- Verify document embeddings: `print(embeddings.shape, np.isnan(embeddings).any())`
- Test API endpoints with curl:
  ```bash
  curl -X POST "http://localhost:8000/search" -H "Content-Type: application/json" -d '{"query": "test", "top_k": 3}'
  ```
