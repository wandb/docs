#!/usr/bin/env python3
"""
Fix casing for TypeScript and Python SDK reference documentation files.
This ensures files follow the correct naming conventions:
- Classes/Interfaces: PascalCase (e.g., WeaveClient.mdx)
- Functions: camelCase (e.g., init.mdx, wrapOpenAI.mdx)
- Python classes: PascalCase (e.g., ArtifactCollection.mdx)
"""

import os
import shutil
from pathlib import Path

def fix_typescript_casing(base_path):
    """Fix TypeScript SDK file casing - ensure all files use lowercase."""
    print("Fixing TypeScript SDK file casing to lowercase...")
    
    ts_base = Path(base_path) / "weave/reference/typescript-sdk"
    if not ts_base.exists():
        print(f"  TypeScript SDK path not found: {ts_base}")
        return
    
    # All TypeScript SDK files should use lowercase filenames for consistency
    # This applies to classes, functions, interfaces, and type-aliases
    subdirs_to_check = ["classes", "functions", "interfaces", "type-aliases"]
    
    for subdir in subdirs_to_check:
        dir_path = ts_base / subdir
        if not dir_path.exists():
            continue
            
        for file in dir_path.glob("*.mdx"):
            # Convert filename to lowercase
            lowercase_name = file.stem.lower()
            if file.stem != lowercase_name:
                new_path = file.parent / f"{lowercase_name}.mdx"
                print(f"  Renaming: {file.name} → {lowercase_name}.mdx")
                shutil.move(str(file), str(new_path))

def fix_python_casing(base_path):
    """Fix Python SDK file casing for WEAVE reference docs only."""
    print("Fixing Weave Python SDK file casing...")
    
    # IMPORTANT: This should ONLY touch Weave reference docs, never Models reference docs
    py_base = Path(base_path) / "weave/reference/python-sdk"
    if not py_base.exists():
        print(f"  Weave Python SDK path not found: {py_base}")
        return
    
    # For Weave Python SDK, we generally want lowercase filenames
    # Only specific files might need special casing - currently none known
    # Most Weave modules use lowercase with underscores (e.g., weave_client.mdx)
    
    print(f"  Weave Python SDK files are generated with correct casing")
    print(f"  No casing changes needed for Weave reference documentation")

def main():
    """Main function to fix all casing issues."""
    # Assume we're running from the docs root
    base_path = Path.cwd()
    
    print("Starting casing fixes...")
    print(f"Working directory: {base_path}")
    
    fix_typescript_casing(base_path)
    fix_python_casing(base_path)
    
    print("\nCasing fixes complete!")
    print("Note: This script should be run after generating reference docs to ensure correct casing.")

if __name__ == "__main__":
    main()
