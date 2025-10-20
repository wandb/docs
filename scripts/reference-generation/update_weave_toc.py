#!/usr/bin/env python3
"""
Update docs.json TOC with newly generated Weave Python SDK reference documentation.
"""

import json
import os
from pathlib import Path


def get_generated_modules():
    """Scan the weave/reference/python-sdk directory to find all generated modules."""
    sdk_dir = Path("weave/reference/python-sdk")
    modules = {}
    
    if not sdk_dir.exists():
        return modules
    
    # Walk through the directory structure
    for mdx_file in sdk_dir.rglob("*.mdx"):
        # Skip index files
        if mdx_file.name == "index.mdx":
            continue
            
        # Get relative path from sdk_dir
        rel_path = mdx_file.relative_to(sdk_dir)
        
        # Convert to module path for TOC
        parts = list(rel_path.parts)
        parts[-1] = parts[-1].replace('.mdx', '')
        
        # Determine the group based on path
        if len(parts) >= 2:
            if parts[0] == "weave":
                if parts[1] == "trace":
                    group = "Core"
                elif parts[1] == "trace_server":
                    group = "Trace Server"
                elif parts[1] == "trace_server_bindings":
                    group = "Trace Server Bindings"
                else:
                    group = "Other"
            else:
                group = "Other"
        else:
            group = "Root"
        
        if group not in modules:
            modules[group] = []
        
        # Create the TOC path
        toc_path = "weave/reference/python-sdk/" + str(rel_path).replace('.mdx', '')
        modules[group].append(toc_path)
    
    # Sort modules within each group
    for group in modules:
        modules[group].sort()
    
    return modules


def update_docs_json(modules):
    """Update the docs.json file with the new module structure."""
    
    # Read current docs.json
    with open("docs.json", "r") as f:
        docs = json.load(f)
    
    # Find the Weave Python SDK section in the navigation
    # This is a nested structure, so we need to traverse it
    def find_and_update_python_sdk(nav_item):
        """Recursively find and update the Python SDK section."""
        if isinstance(nav_item, dict):
            # Check if this is the Python SDK group
            if nav_item.get("group") == "Python SDK":
                # Found it! Update the pages
                pages = nav_item.get("pages", [])
                
                # Keep the index page if it exists
                new_pages = []
                for page in pages:
                    if isinstance(page, str) and page.endswith("/index"):
                        new_pages.append(page)
                        break
                
                # Add the grouped modules
                if "Core" in modules and modules["Core"]:
                    new_pages.append({
                        "group": "Core",
                        "pages": modules["Core"]
                    })
                
                if "Trace Server" in modules and modules["Trace Server"]:
                    new_pages.append({
                        "group": "Trace Server",
                        "pages": modules["Trace Server"]
                    })
                
                if "Trace Server Bindings" in modules and modules["Trace Server Bindings"]:
                    new_pages.append({
                        "group": "Trace Server Bindings",
                        "pages": modules["Trace Server Bindings"]
                    })
                
                if "Other" in modules and modules["Other"]:
                    new_pages.append({
                        "group": "Other",
                        "pages": modules["Other"]
                    })
                
                nav_item["pages"] = new_pages
                return True
            
            # Recursively check pages
            if "pages" in nav_item:
                for page in nav_item["pages"]:
                    if find_and_update_python_sdk(page):
                        return True
        
        elif isinstance(nav_item, list):
            for item in nav_item:
                if find_and_update_python_sdk(item):
                    return True
        
        return False
    
    # Update the navigation
    if "navigation" in docs:
        for nav in docs["navigation"]:
            if "tabs" in nav:
                for tab in nav["tabs"]:
                    if tab.get("tab") == "W&B Weave":
                        if find_and_update_python_sdk(tab):
                            break
    
    # Write updated docs.json
    with open("docs.json", "w") as f:
        json.dump(docs, f, indent=2)
    
    print("✓ Updated docs.json with new Python SDK modules")


def main():
    """Main function."""
    print("Updating Weave Python SDK TOC in docs.json...")
    
    # Get the generated modules
    modules = get_generated_modules()
    
    if not modules:
        print("No Python SDK modules found to add to TOC")
        return
    
    print(f"Found modules in {len(modules)} groups:")
    for group, pages in modules.items():
        print(f"  {group}: {len(pages)} modules")
        for page in pages[:3]:  # Show first 3 as examples
            print(f"    - {page}")
        if len(pages) > 3:
            print(f"    ... and {len(pages) - 3} more")
    
    # Update docs.json
    update_docs_json(modules)
    
    print("\n✓ TOC update complete!")


if __name__ == "__main__":
    main()
