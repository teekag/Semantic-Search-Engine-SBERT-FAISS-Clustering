#!/usr/bin/env python
"""
Run the Semantic Search Engine API server.

This script starts the FastAPI server and provides a simple interface
to test the search functionality.
"""

import uvicorn
import webbrowser
import time
import threading
import argparse

def open_browser(port):
    """Open the browser to the API documentation after a short delay."""
    time.sleep(2)  # Give the server time to start
    webbrowser.open(f"http://localhost:{port}/docs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Semantic Search Engine API server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the server on")
    parser.add_argument("--no-browser", action="store_true", help="Don't open the browser automatically")
    args = parser.parse_args()
    
    print(f"Starting Semantic Search Engine API server on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the server")
    
    if not args.no_browser:
        # Open browser in a separate thread
        threading.Thread(target=open_browser, args=(args.port,)).start()
    
    # Start the server
    uvicorn.run("src.api:app", host=args.host, port=args.port, reload=True)
