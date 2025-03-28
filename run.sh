#!/bin/bash
# Run script for Semantic Search Engine

# Function to display help
show_help() {
    echo "Semantic Search Engine - Run Script"
    echo ""
    echo "Usage: ./run.sh [command]"
    echo ""
    echo "Commands:"
    echo "  api         - Start the FastAPI server"
    echo "  demo        - Run the interactive search demo"
    echo "  pipeline    - Run the enhanced search pipeline"
    echo "  test        - Run all unit tests"
    echo "  help        - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run.sh api         # Start the API server"
    echo "  ./run.sh demo        # Run the interactive demo"
    echo "  ./run.sh pipeline    # Run the enhanced search pipeline"
    echo "  ./run.sh test        # Run all unit tests"
}

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    source venv/bin/activate
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Process command
case "$1" in
    api)
        echo "Starting API server..."
        python run_api_server.py
        ;;
    demo)
        echo "Running interactive search demo..."
        python demo_search.py
        ;;
    pipeline)
        echo "Running enhanced search pipeline..."
        python notebooks/enhanced_search_pipeline.py
        ;;
    test)
        echo "Running unit tests..."
        python -m pytest tests/
        ;;
    help|*)
        show_help
        ;;
esac

# Deactivate virtual environment
deactivate
