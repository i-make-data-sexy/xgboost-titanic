"""
Generates a visual representation of the project structure.
"""

# ========================================================================
#   Imports
# ========================================================================

import os
from pathlib import Path

# ========================================================================
#   Configuration
# ========================================================================

# Directories and files to ignore
IGNORE = {
    "__pycache__", ".git", ".venv", "venv", 
    ".pytest_cache", ".idea", ".vscode",
    "*.pyc", "*.pyo", ".DS_Store"
}

# ========================================================================
#   Structure Generation
# ========================================================================

def generate_structure(path, prefix="", max_depth=None, current_depth=0):
    """
    Recursively generates project structure.
    
    Args:
        path (Path): Directory path to analyze
        prefix (str): Prefix for tree branches
        max_depth (int): Maximum depth to traverse
        current_depth (int): Current recursion depth
    """
    
    # Check depth limit
    if max_depth and current_depth >= max_depth:
        return
    
    # Get all items in directory
    items = sorted(Path(path).iterdir(), key=lambda x: (x.is_file(), x.name))
    
    # Filter out ignored items
    items = [
        item for item in items 
        if not any(
            ignored in str(item) 
            for ignored in IGNORE
        )
    ]
    
    for i, item in enumerate(items):
        # Determine if last item
        is_last = i == len(items) - 1
        
        # Print current item
        current_prefix = "└── " if is_last else "├── "
        print(f"{prefix}{current_prefix}{item.name}")
        
        # Recurse for directories
        if item.is_dir():
            next_prefix = prefix + ("    " if is_last else "│   ")
            generate_structure(
                item, next_prefix, max_depth, current_depth + 1
            )

# ========================================================================
#   Run
# ========================================================================

if __name__ == "__main__":
    print("Project Structure:")
    print("=" * 40)
    print(os.path.basename(os.getcwd()) + "/")
    generate_structure(".", max_depth=3)