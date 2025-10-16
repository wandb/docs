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
    """Fix TypeScript SDK file casing."""
    print("Fixing TypeScript SDK file casing...")
    
    ts_base = Path(base_path) / "weave/reference/typescript-sdk/weave"
    if not ts_base.exists():
        print(f"  TypeScript SDK path not found: {ts_base}")
        return
    
    # Define correct names for each directory
    casing_rules = {
        "classes": {
            "dataset": "Dataset",
            "evaluation": "Evaluation", 
            "weaveclient": "WeaveClient",
            "weaveobject": "WeaveObject",
        },
        "interfaces": {
            "callschema": "CallSchema",
            "callsfilter": "CallsFilter",
            "weaveaudio": "WeaveAudio",
            "weaveimage": "WeaveImage",
        },
        "functions": {
            # Functions should be lowercase/camelCase
            "init": "init",
            "login": "login",
            "op": "op",
            "requirecurrentcallstackentry": "requireCurrentCallStackEntry",
            "requirecurrentchildsummary": "requireCurrentChildSummary",
            "weaveaudio": "weaveAudio",
            "weaveimage": "weaveImage",
            "wrapopenai": "wrapOpenAI",
        },
        "type-aliases": {
            "op": "Op",  # Type alias Op is uppercase
            "opdecorator": "OpDecorator",
            "messagesprompt": "MessagesPrompt",
            "stringprompt": "StringPrompt",
        }
    }
    
    for dir_name, rules in casing_rules.items():
        dir_path = ts_base / dir_name
        if not dir_path.exists():
            continue
            
        for file in dir_path.glob("*.mdx"):
            basename = file.stem.lower()
            if basename in rules:
                correct_name = rules[basename]
                if file.stem != correct_name:
                    new_path = file.parent / f"{correct_name}.mdx"
                    print(f"  Renaming: {file.name} → {correct_name}.mdx")
                    shutil.move(str(file), str(new_path))

def fix_python_casing(base_path):
    """Fix Python SDK file casing."""
    print("Fixing Python SDK file casing...")
    
    py_base = Path(base_path) / "models/ref/python/public-api"
    if not py_base.exists():
        print(f"  Python SDK path not found: {py_base}")
        return
    
    # Python class files that should be uppercase
    uppercase_files = {
        "artifactcollection": "ArtifactCollection",
        "artifactcollections": "ArtifactCollections",
        "artifactfiles": "ArtifactFiles",
        "artifacttype": "ArtifactType",
        "artifacttypes": "ArtifactTypes",
        "betareport": "BetaReport",
        "file": "File",
        "member": "Member",
        "project": "Project",
        "registry": "Registry",
        "run": "Run",
        "runartifacts": "RunArtifacts",
        "sweep": "Sweep",
        "team": "Team",
        "user": "User",
    }
    
    # Files that should remain lowercase
    lowercase_files = ["api", "artifacts", "automations", "files", "projects", 
                       "reports", "runs", "sweeps", "_index"]
    
    for file in py_base.glob("*.mdx"):
        basename = file.stem.lower()
        
        if basename in uppercase_files:
            correct_name = uppercase_files[basename]
            if file.stem != correct_name:
                new_path = file.parent / f"{correct_name}.mdx"
                print(f"  Renaming: {file.name} → {correct_name}.mdx")
                shutil.move(str(file), str(new_path))
        elif basename in lowercase_files:
            # Ensure these stay lowercase
            if file.stem != basename:
                new_path = file.parent / f"{basename}.mdx"
                print(f"  Renaming: {file.name} → {basename}.mdx")
                shutil.move(str(file), str(new_path))

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
