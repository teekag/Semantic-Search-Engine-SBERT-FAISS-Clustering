"""
# Enhanced Semantic Search Pipeline

This script demonstrates the full capabilities of the semantic search engine:
1. Loading the 20 Newsgroups dataset
2. Embedding documents using SBERT
3. Indexing embeddings with FAISS
4. Searching for semantically similar documents
5. Clustering and visualizing the embedding space
6. Evaluating search quality with metrics
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

# Add the parent directory to the path to correctly import src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our modules
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.clusterer import Clusterer

# Create output directory if it doesn't exist
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))
os.makedirs(output_dir, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# 1. Load the 20 Newsgroups Dataset
print("## 1. Loading Dataset")
categories = ['comp.graphics', 'sci.med', 'rec.autos', 'talk.politics.guns', 
              'sci.space', 'comp.sys.mac.hardware', 'rec.sport.baseball']
newsgroups = fetch_20newsgroups(subset='train', categories=categories, 
                               remove=('headers', 'footers', 'quotes'))

# Create document dictionaries
documents = []
for i, (text, target) in enumerate(zip(newsgroups.data, newsgroups.target)):
    # Clean up the text a bit
    text = text.strip()
    if len(text) > 100:  # Filter out very short documents
        documents.append({
            "id": i,
            "text": text,
            "category": newsgroups.target_names[target],
            "category_id": target
        })

print(f"Loaded {len(documents)} documents from {len(categories)} categories")

# Display a sample document
sample_idx = np.random.randint(0, len(documents))
sample_doc = documents[sample_idx]
print(f"\nSample document (category: {sample_doc['category']}):\n")
print(sample_doc['text'][:500] + "...")

# 2. Embed Documents using SBERT
print("\n## 2. Embedding Documents")
embedder = Embedder(model_name="all-MiniLM-L6-v2")

# Extract the text from documents
texts = [doc["text"] for doc in documents]

# Generate embeddings
embeddings = embedder.embed_texts(texts)

print(f"Generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
print(f"Sample embedding (first 5 dimensions): {embeddings[0][:5]}")

# 3. Index Embeddings with FAISS
print("\n## 3. Indexing Embeddings")
vector_store = VectorStore(
    dimension=embedder.get_embedding_dimension(),
    index_type="Flat",  # Use Flat for exact search
    metric="cosine"
)

# Add the embeddings to the index
vector_store.add(embeddings, documents)

# Display index info
index_info = vector_store.get_index_info()
for key, value in index_info.items():
    print(f"{key}: {value}")

# 4. Search for Similar Documents
print("\n## 4. Semantic Search")
# Define some test queries
test_queries = [
    "How to treat a high fever",
    "The best sports cars on the market",
    "Rendering 3D graphics with ray tracing",
    "Second amendment rights and gun control laws",
    "Space shuttle design and NASA missions",
    "Baseball statistics and player performance"
]

# Function to perform a search and display results
def search_and_display(query, k=5):
    print(f"Query: {query}\n")
    
    # Embed the query
    query_embedding = embedder.embed_query(query)
    
    # Search for similar documents
    results = vector_store.search(query_embedding, k=k)
    
    # Display results
    for i, (doc, sim) in enumerate(zip(results["documents"], results["similarities"])):
        print(f"{i+1}. [{doc['category']}] (Similarity: {sim:.4f})")
        print(f"   {doc['text'][:200].replace(chr(10), ' ')}...\n")
    
    return results

# Test each query
all_results = {}
for query in test_queries:
    results = search_and_display(query)
    all_results[query] = results
    print("-" * 80)

# 5. Cluster Documents
print("\n## 5. Clustering and Visualization")
# Initialize the clusterer
clusterer = Clusterer(algorithm="kmeans")

# Fit the clusterer (use the number of actual categories)
labels = clusterer.fit(embeddings, documents, n_clusters=len(categories))

# Evaluate clustering
silhouette = clusterer.evaluate(metric="silhouette")
print(f"Silhouette score: {silhouette:.4f}")

# Reduce dimensions for visualization
reduced_embeddings = clusterer.reduce_dimensions(n_components=2)

# Create a DataFrame for plotting
plot_df = pd.DataFrame({
    'x': reduced_embeddings[:, 0],
    'y': reduced_embeddings[:, 1],
    'category': [doc['category'] for doc in documents],
    'cluster': labels
})

# Plot by true category
plt.figure(figsize=(12, 10))
sns.scatterplot(data=plot_df, x='x', y='y', hue='category', palette='tab10', alpha=0.7)
plt.title('Document Embedding Space by Category')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'embedding_by_category.png'), dpi=300)
plt.close()

# Plot by cluster
plt.figure(figsize=(12, 10))
sns.scatterplot(data=plot_df, x='x', y='y', hue='cluster', palette='tab10', alpha=0.7)
plt.title('Document Embedding Space by Cluster')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'embedding_by_cluster.png'), dpi=300)
plt.close()

# 6. Evaluation Metrics
print("\n## 6. Evaluation Metrics")

# Function to calculate precision@k and recall@k
def calculate_precision_recall_at_k(query, results, k=5, relevant_category=None):
    """
    Calculate precision@k and recall@k for a query.
    
    Args:
        query: The query string
        results: The search results
        k: The number of results to consider
        relevant_category: The category considered relevant
        
    Returns:
        precision@k, recall@k
    """
    if relevant_category is None:
        # Infer relevant category from query
        category_keywords = {
            'comp.graphics': ['graphics', 'rendering', '3d', 'animation'],
            'sci.med': ['medical', 'disease', 'treatment', 'fever', 'health'],
            'rec.autos': ['car', 'vehicle', 'engine', 'sports car'],
            'talk.politics.guns': ['gun', 'amendment', 'rights', 'control'],
            'sci.space': ['space', 'nasa', 'shuttle', 'mission', 'rocket'],
            'comp.sys.mac.hardware': ['mac', 'apple', 'hardware', 'computer'],
            'rec.sport.baseball': ['baseball', 'player', 'statistics', 'game']
        }
        
        # Find the category with the most keyword matches
        max_matches = 0
        relevant_category = None
        
        for category, keywords in category_keywords.items():
            matches = sum(1 for keyword in keywords if keyword.lower() in query.lower())
            if matches > max_matches:
                max_matches = matches
                relevant_category = category
    
    # Get the top k results
    top_k_results = results["documents"][:k]
    
    # Calculate precision@k
    relevant_results = [doc for doc in top_k_results if doc['category'] == relevant_category]
    precision_at_k = len(relevant_results) / k if k > 0 else 0
    
    # Calculate recall@k (assuming all documents in the relevant category are relevant)
    total_relevant = sum(1 for doc in documents if doc['category'] == relevant_category)
    recall_at_k = len(relevant_results) / total_relevant if total_relevant > 0 else 0
    
    return precision_at_k, recall_at_k

# Calculate precision@k and recall@k for each query
print("Precision@5 and Recall@5 for each query:")
for query, results in all_results.items():
    precision, recall = calculate_precision_recall_at_k(query, results, k=5)
    print(f"Query: {query}")
    print(f"  Precision@5: {precision:.4f}")
    print(f"  Recall@5: {recall:.4f}")
    print()

# Calculate Mean Reciprocal Rank (MRR)
def calculate_mrr(query, results, relevant_category=None):
    """
    Calculate Mean Reciprocal Rank for a query.
    
    Args:
        query: The query string
        results: The search results
        relevant_category: The category considered relevant
        
    Returns:
        MRR score
    """
    if relevant_category is None:
        # Infer relevant category from query (same as above)
        category_keywords = {
            'comp.graphics': ['graphics', 'rendering', '3d', 'animation'],
            'sci.med': ['medical', 'disease', 'treatment', 'fever', 'health'],
            'rec.autos': ['car', 'vehicle', 'engine', 'sports car'],
            'talk.politics.guns': ['gun', 'amendment', 'rights', 'control'],
            'sci.space': ['space', 'nasa', 'shuttle', 'mission', 'rocket'],
            'comp.sys.mac.hardware': ['mac', 'apple', 'hardware', 'computer'],
            'rec.sport.baseball': ['baseball', 'player', 'statistics', 'game']
        }
        
        max_matches = 0
        relevant_category = None
        
        for category, keywords in category_keywords.items():
            matches = sum(1 for keyword in keywords if keyword.lower() in query.lower())
            if matches > max_matches:
                max_matches = matches
                relevant_category = category
    
    # Find the first relevant result
    for i, doc in enumerate(results["documents"]):
        if doc['category'] == relevant_category:
            return 1.0 / (i + 1)  # Reciprocal rank
    
    return 0.0  # No relevant results found

# Calculate MRR for each query
print("Mean Reciprocal Rank (MRR) for each query:")
mrr_scores = []
for query, results in all_results.items():
    mrr = calculate_mrr(query, results)
    mrr_scores.append(mrr)
    print(f"Query: {query}")
    print(f"  MRR: {mrr:.4f}")
    print()

# Calculate overall MRR
overall_mrr = np.mean(mrr_scores)
print(f"Overall MRR: {overall_mrr:.4f}")

# Create a t-SNE visualization
print("\n## 7. t-SNE Visualization")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_embeddings = tsne.fit_transform(embeddings)

# Create a DataFrame for plotting
tsne_df = pd.DataFrame({
    'x': tsne_embeddings[:, 0],
    'y': tsne_embeddings[:, 1],
    'category': [doc['category'] for doc in documents],
    'cluster': labels
})

# Plot by true category
plt.figure(figsize=(12, 10))
sns.scatterplot(data=tsne_df, x='x', y='y', hue='category', palette='tab10', alpha=0.7)
plt.title('t-SNE Visualization of Document Embeddings by Category')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'tsne_by_category.png'), dpi=300)
plt.close()

# Plot by cluster
plt.figure(figsize=(12, 10))
sns.scatterplot(data=tsne_df, x='x', y='y', hue='cluster', palette='tab10', alpha=0.7)
plt.title('t-SNE Visualization of Document Embeddings by Cluster')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'tsne_by_cluster.png'), dpi=300)
plt.close()

# 8. Save Sample Query Results
print("\n## 8. Saving Sample Query Results")
# Save a sample query result to a JSON file
import json

# Choose a sample query (NASA mission)
sample_query = "NASA mission and space exploration"
print(f"Running sample query: {sample_query}")

# Embed the query
query_embedding = embedder.embed_query(sample_query)

# Search for similar documents
results = vector_store.search(query_embedding, k=5)

# Format results for JSON
formatted_results = {
    "query": sample_query,
    "results": []
}

for i, (doc, sim) in enumerate(zip(results["documents"], results["similarities"])):
    formatted_results["results"].append({
        "rank": i + 1,
        "document_id": doc["id"],
        "category": doc["category"],
        "similarity": float(sim),
        "text_snippet": doc["text"][:300].replace("\n", " ") + "..."
    })

# Save to JSON file
with open(os.path.join(output_dir, 'sample_query_result.json'), 'w') as f:
    json.dump(formatted_results, f, indent=2)

print(f"Saved sample query results to '{os.path.join(output_dir, 'sample_query_result.json')}'")

print("\nEnhanced Search Pipeline Complete!")
