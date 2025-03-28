#!/usr/bin/env python
"""
Interactive Demo for Semantic Search Engine.

This script provides a simple command-line interface to test the semantic search
functionality without needing to start the API server.
"""

import os
import sys
import argparse
from sklearn.datasets import fetch_20newsgroups
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import our modules
from src.embedder import Embedder
from src.vector_store import VectorStore

console = Console()

def setup_search_engine(use_gpu=False):
    """Set up the search engine components."""
    console.print(Panel("Loading Semantic Search Engine...", style="bold blue"))
    
    # Initialize embedder
    embedder = Embedder(model_name="all-MiniLM-L6-v2", use_gpu=use_gpu)
    console.print(f"[green]✓[/green] Loaded embedder: {embedder.get_model_info()['name']}")
    
    # Load dataset
    console.print("Loading 20 Newsgroups dataset...")
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
    
    console.print(f"[green]✓[/green] Loaded {len(documents)} documents from {len(categories)} categories")
    
    # Extract the text from documents
    texts = [doc["text"] for doc in documents]
    
    # Generate embeddings
    console.print("Generating embeddings...")
    embeddings = embedder.embed_texts(texts)
    console.print(f"[green]✓[/green] Generated {len(embeddings)} embeddings")
    
    # Create vector store
    vector_store = VectorStore(
        dimension=embedder.get_embedding_dimension(),
        index_type="Flat",  # Use Flat for exact search
        metric="cosine",
        use_gpu=use_gpu
    )
    
    # Add the embeddings to the index
    vector_store.add(embeddings, documents)
    console.print(f"[green]✓[/green] Indexed {len(documents)} documents")
    
    return embedder, vector_store, documents

def display_results(results, query):
    """Display search results in a nicely formatted table."""
    table = Table(title=f"Search Results for: '{query}'")
    table.add_column("Rank", style="dim")
    table.add_column("Category", style="green")
    table.add_column("Similarity", style="cyan")
    table.add_column("Text", style="white", no_wrap=False)
    
    for i, (doc, similarity) in enumerate(zip(results["documents"], results["similarities"])):
        # Truncate text for display
        text = doc["text"]
        if len(text) > 200:
            text = text[:200] + "..."
        
        table.add_row(
            str(i + 1),
            doc["category"],
            f"{similarity:.4f}",
            text
        )
    
    console.print(table)

def interactive_search(embedder, vector_store):
    """Run an interactive search loop."""
    while True:
        query = Prompt.ask("\n[bold blue]Enter a search query[/bold blue] (or 'q' to quit)")
        
        if query.lower() in ('q', 'quit', 'exit'):
            break
        
        # Embed the query
        query_embedding = embedder.embed_query(query)
        
        # Search
        results = vector_store.search(query_embedding, k=5)
        
        # Display results
        display_results(results, query)

def main():
    parser = argparse.ArgumentParser(description="Interactive Semantic Search Demo")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration if available")
    args = parser.parse_args()
    
    try:
        # Setup search engine
        embedder, vector_store, documents = setup_search_engine(use_gpu=args.gpu)
        
        # Print welcome message
        console.print(Panel(
            "[bold green]Semantic Search Engine Demo[/bold green]\n\n"
            "This demo allows you to search the 20 Newsgroups dataset using semantic search.\n"
            "Try queries like:\n"
            "  - How to treat a high fever\n"
            "  - The best sports cars on the market\n"
            "  - Problems with Windows operating system\n"
            "  - Space exploration and NASA missions\n\n"
            "Type 'q' to quit.",
            expand=False
        ))
        
        # Run interactive search
        interactive_search(embedder, vector_store)
        
    except KeyboardInterrupt:
        console.print("\n[bold red]Search demo terminated.[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        
    console.print("[bold green]Thank you for using the Semantic Search Engine Demo![/bold green]")

if __name__ == "__main__":
    main()
