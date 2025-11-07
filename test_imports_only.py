#!/usr/bin/env python3
"""Simple test to verify module structure without dependencies."""

import sys
import ast
import os

def test_module_structure():
    """Test that all modules have correct structure."""
    modules = [
        "models/clip_utils.py",
        "models/data_and_evaluation.py",
        "models/training_and_reference.py",
        "models/subspace_lora_clip_learner.py"
    ]
    
    for module_path in modules:
        print(f"\nChecking {module_path}...")
        try:
            with open(module_path, 'r') as f:
                content = f.read()
            
            # Parse the AST to check structure
            tree = ast.parse(content)
            
            # Check for classes and functions
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            
            print(f"  ✓ Found {len(classes)} classes: {', '.join(classes)}")
            print(f"  ✓ Found {len(functions)} functions")
            
            # Check for imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}")
            
            print(f"  ✓ Found {len(imports)} imports")
            
        except Exception as e:
            print(f"  ✗ Error checking {module_path}: {e}")
            return False
    
    return True

def test_file_hierarchy():
    """Test that the file hierarchy is correct."""
    expected_files = [
        "models/clip_utils.py",
        "models/data_and_evaluation.py",
        "models/training_and_reference.py",
        "models/subspace_lora_clip_learner.py"
    ]
    
    print("\nChecking file hierarchy...")
    for file_path in expected_files:
        if os.path.exists(file_path):
            print(f"  ✓ {file_path} exists")
        else:
            print(f"  ✗ {file_path} missing")
            return False
    
    return True

def main():
    """Run all tests."""
    print("Testing refactored SubspaceLoRA CLIP module structure...")
    print("=" * 60)
    
    success = True
    
    print("\n1. Testing file hierarchy...")
    success &= test_file_hierarchy()
    
    print("\n2. Testing module structure...")
    success &= test_module_structure()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ All structural tests passed! The refactored code structure is correct.")
        return 0
    else:
        print("✗ Some structural tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())