#!/usr/bin/env python3
"""Production server runner with both HTTP and gRPC support"""

import os
import sys
import asyncio
import signal
import multiprocessing
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_http_server():
    """Run the FastAPI HTTP server"""
    from app.main import start_server
    start_server()


def run_grpc_server():
    """Run the gRPC server"""
    from app.grpc_server import serve
    serve()


def main():
    """Run both HTTP and gRPC servers"""
    print("Starting Cortex Memory API Server...")
    print("=" * 50)
    
    # Compile protos first
    print("Compiling protocol buffers...")
    from compile_protos import compile_protos
    if not compile_protos():
        print("Failed to compile protos. Exiting.")
        sys.exit(1)
    
    # Create processes for each server
    http_process = multiprocessing.Process(target=run_http_server)
    grpc_process = multiprocessing.Process(target=run_grpc_server)
    
    # Start both servers
    http_process.start()
    grpc_process.start()
    
    print("=" * 50)
    print("Servers started:")
    print(f"  HTTP API: http://localhost:8080")
    print(f"  API Docs: http://localhost:8080/docs")
    print(f"  gRPC:     localhost:50051")
    print("=" * 50)
    
    try:
        # Wait for both processes
        http_process.join()
        grpc_process.join()
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        http_process.terminate()
        grpc_process.terminate()
        http_process.join()
        grpc_process.join()
        print("Servers stopped.")


if __name__ == "__main__":
    main()