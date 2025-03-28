"""
FastAPI application for the Semantic Search Engine.

This module provides a REST API for the semantic search engine, allowing
users to search for documents and explore the embedding space.
"""

import os
import json
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
from sklearn.datasets import fetch_20newsgroups

# Import our modules - use relative imports for better compatibility
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.clusterer import Clusterer


# Define models for request/response
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    document_id: int
    text: str
    category: str
    similarity: float


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]


class ClusterResponse(BaseModel):
    num_clusters: int
    silhouette_score: float
    cluster_counts: Dict[str, int]
    cluster_categories: Dict[str, Dict[str, int]]
    reduced_embeddings: Optional[List[List[float]]] = None


# Initialize FastAPI app
app = FastAPI(
    title="Semantic Search API",
    description="API for semantic search using SBERT, FAISS, and clustering",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
embedder = None
vector_store = None
clusterer = None
documents = None
embeddings = None


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global embedder, vector_store, clusterer, documents, embeddings
    
    print("Loading models...")
    
    # Initialize embedder
    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    
    # Load dataset
    categories = ['comp.graphics', 'sci.med', 'rec.autos', 'talk.politics.guns', 
                  'sci.space', 'comp.sys.mac.hardware', 'rec.sport.baseball']
    newsgroups = fetch_20newsgroups(
        subset='train', 
        categories=categories, 
        remove=('headers', 'footers', 'quotes')
    )
    
    # Create document dictionaries
    documents = []
    for i, (text, target) in enumerate(zip(newsgroups.data, newsgroups.target)):
        text = text.strip()
        if len(text) > 100:  # Filter out very short documents
            documents.append({
                "id": i,
                "text": text,
                "category": newsgroups.target_names[target],
                "category_id": target
            })
    
    # Generate embeddings
    texts = [doc["text"] for doc in documents]
    embeddings = embedder.embed_texts(texts)
    
    # Initialize vector store
    vector_store = VectorStore(
        dimension=embedder.get_embedding_dimension(),
        index_type="Flat",
        metric="cosine"
    )
    
    # Add embeddings to vector store
    vector_store.add(embeddings, documents)
    
    # Initialize clusterer
    clusterer = Clusterer(algorithm="kmeans")
    clusterer.fit(embeddings, documents, n_clusters=len(categories))
    
    print(f"Models loaded. Indexed {len(documents)} documents.")


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to the Semantic Search API", 
            "docs_url": "/docs",
            "endpoints": ["/search", "/clusters", "/status", "/categories", "/document/{document_id}"]}


@app.get("/status")
async def status():
    """Get status of the API."""
    if embedder is None or vector_store is None:
        return {"status": "initializing"}
    
    return {
        "status": "ready",
        "documents_indexed": len(documents) if documents else 0,
        "embedding_model": embedder.model_name if embedder else None,
        "index_type": vector_store.index_type if vector_store else None,
        "clustering_algorithm": clusterer.algorithm if clusterer else None
    }


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search for documents similar to the query.
    
    Args:
        request: SearchRequest containing the query and top_k
        
    Returns:
        SearchResponse with query and results
    """
    if embedder is None or vector_store is None:
        raise HTTPException(status_code=503, detail="Service is initializing")
    
    # Embed the query
    query_embedding = embedder.embed_query(request.query)
    
    # Search for similar documents
    results = vector_store.search(query_embedding, k=request.top_k)
    
    # Format results
    search_results = []
    for i, (doc, sim) in enumerate(zip(results["documents"], results["similarities"])):
        search_results.append(
            SearchResult(
                document_id=doc["id"],
                text=doc["text"][:500] + "..." if len(doc["text"]) > 500 else doc["text"],
                category=doc["category"],
                similarity=sim
            )
        )
    
    return SearchResponse(query=request.query, results=search_results)


@app.get("/categories")
async def get_categories():
    """Get all document categories."""
    if documents is None:
        raise HTTPException(status_code=503, detail="Service is initializing")
    
    categories = set(doc["category"] for doc in documents)
    return {"categories": list(categories)}


@app.get("/document/{document_id}")
async def get_document(document_id: int):
    """Get a specific document by ID."""
    if documents is None:
        raise HTTPException(status_code=503, detail="Service is initializing")
    
    for doc in documents:
        if doc["id"] == document_id:
            return doc
    
    raise HTTPException(status_code=404, detail="Document not found")


@app.get("/clusters")
async def get_clusters(include_embeddings: bool = False):
    """
    Get cluster information and optionally reduced embeddings for visualization.
    
    Args:
        include_embeddings: Whether to include reduced embeddings in the response
        
    Returns:
        Cluster information and optionally reduced embeddings
    """
    if clusterer is None or clusterer.labels is None:
        raise HTTPException(status_code=503, detail="Clustering not available")
    
    # Get cluster info
    cluster_info = clusterer.get_cluster_info()
    
    # Get document counts per cluster
    cluster_counts = {}
    for label in set(clusterer.labels):
        if label == -1:  # Skip noise points
            continue
        cluster_counts[str(int(label))] = int(sum(clusterer.labels == label))
    
    # Get category distribution per cluster
    cluster_categories = {}
    for label in set(clusterer.labels):
        if label == -1:  # Skip noise points
            continue
        
        # Get documents in this cluster
        cluster_docs = [documents[i] for i, l in enumerate(clusterer.labels) if l == label]
        
        # Count categories
        category_counts = {}
        for doc in cluster_docs:
            category = doc["category"]
            category_counts[category] = category_counts.get(category, 0) + 1
        
        cluster_categories[str(int(label))] = category_counts
    
    response = {
        "num_clusters": len(set(clusterer.labels)) - (1 if -1 in clusterer.labels else 0),
        "silhouette_score": float(clusterer.evaluate()),
        "cluster_counts": cluster_counts,
        "cluster_categories": cluster_categories
    }
    
    # Include reduced embeddings if requested
    if include_embeddings:
        if clusterer.reduced_embeddings is None:
            # Reduce dimensions if not already done
            clusterer.reduce_dimensions(n_components=2)
        
        # Convert to list of lists for JSON serialization
        reduced_embeddings = clusterer.reduced_embeddings.tolist()
        response["reduced_embeddings"] = reduced_embeddings
        response["labels"] = clusterer.labels.tolist()
    
    return response


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle all exceptions."""
    return JSONResponse(
        status_code=500,
        content={"message": f"An error occurred: {str(exc)}"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
