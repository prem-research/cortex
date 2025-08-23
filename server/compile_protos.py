#!/usr/bin/env python3
"""Compile protobuf files for gRPC"""

import os
import sys
from pathlib import Path

def compile_protos():
    """Compile proto files to Python modules"""
    # Get paths
    current_dir = Path(__file__).parent
    proto_dir = current_dir / "protos"
    output_dir = current_dir / "app" / "generated"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create __init__.py in generated directory
    init_file = output_dir / "__init__.py"
    init_file.touch()
    
    # Compile proto files
    proto_file = proto_dir / "cortex.proto"
    
    if not proto_file.exists():
        print(f"Proto file not found: {proto_file}")
        return False
    
    # Run protoc compiler
    cmd = (
        f"python -m grpc_tools.protoc "
        f"-I{proto_dir} "
        f"--python_out={output_dir} "
        f"--grpc_python_out={output_dir} "
        f"--pyi_out={output_dir} "
        f"{proto_file}"
    )
    
    print(f"Compiling: {proto_file}")
    print(f"Command: {cmd}")
    
    result = os.system(cmd)
    
    if result == 0:
        print(f"Successfully compiled protobuf files to {output_dir}")
        
        # Fix imports in generated files
        fix_imports(output_dir)
        return True
    else:
        print(f"Failed to compile protobuf files (exit code: {result})")
        return False


def fix_imports(output_dir: Path):
    """Fix relative imports in generated files"""
    # Fix imports in cortex_pb2_grpc.py
    grpc_file = output_dir / "cortex_pb2_grpc.py"
    if grpc_file.exists():
        content = grpc_file.read_text()
        # Change import from absolute to relative
        content = content.replace("import cortex_pb2", "from . import cortex_pb2")
        grpc_file.write_text(content)
        print(f"Fixed imports in {grpc_file}")


if __name__ == "__main__":
    success = compile_protos()
    sys.exit(0 if success else 1)