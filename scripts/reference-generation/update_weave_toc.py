#!/usr/bin/env python3
"""
Update docs.json TOC with newly generated Weave reference documentation.
Updates Python SDK, TypeScript SDK, and Service API sections.
"""

import json
import os
from pathlib import Path


def get_generated_python_modules():
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


def get_typescript_items():
    """Scan the weave/reference/typescript-sdk directory for generated items."""
    ts_dir = Path("weave/reference/typescript-sdk/weave")
    items = {}
    
    if not ts_dir.exists():
        return items
    
    # Check for different subdirectories
    for subdir in ["classes", "functions", "interfaces", "type-aliases"]:
        subdir_path = ts_dir / subdir
        if subdir_path.exists():
            items[subdir] = []
            for mdx_file in subdir_path.glob("*.mdx"):
                name = mdx_file.stem
                toc_path = f"weave/reference/typescript-sdk/weave/{subdir}/{name}"
                items[subdir].append(toc_path)
            items[subdir].sort()
    
    return items


def get_service_api_endpoints():
    """Scan the weave/reference/service-api directory for generated endpoints."""
    api_dir = Path("weave/reference/service-api")
    endpoints = []
    
    if not api_dir.exists():
        return endpoints
    
    for mdx_file in api_dir.glob("*.mdx"):
        # Skip index files
        if mdx_file.name == "index.mdx":
            continue
        
        name = mdx_file.stem
        toc_path = f"weave/reference/service-api/{name}"
        endpoints.append(toc_path)
    
    return sorted(endpoints)


def update_docs_json(python_modules, typescript_items, service_endpoints):
    """Update the docs.json file with all reference documentation."""
    
    # Read current docs.json
    with open("docs.json", "r") as f:
        docs = json.load(f)
    
    # Navigate through the navigation structure
    # The structure is: navigation.languages[0].tabs[x].tab="W&B Weave"
    navigation = docs.get("navigation", {})
    languages = navigation.get("languages", [])
    if not languages:
        print("⚠️  No languages found in navigation")
        return
    
    updated = False
    # Look for English navigation (first language)
    for lang in languages:
        if lang.get("language") == "en" and "tabs" in lang:
            for tab in lang["tabs"]:
                if tab.get("tab") == "W&B Weave":
                    # Find the Reference group
                    for group in tab.get("pages", []):
                        if isinstance(group, dict) and group.get("group") == "Reference":
                            reference_pages = group.get("pages", [])
                            
                            # Update Python SDK
                            for page in reference_pages:
                                if isinstance(page, dict) and page.get("group") == "Python SDK":
                                    # Keep the index page if it exists
                                    new_pages = []
                                    for existing_page in page.get("pages", []):
                                        if isinstance(existing_page, str) and existing_page.endswith("/index"):
                                            new_pages.append(existing_page)
                                            break
                                    
                                    # Add the grouped modules
                                    if "Core" in python_modules and python_modules["Core"]:
                                        new_pages.append({
                                            "group": "Core",
                                            "pages": python_modules["Core"]
                                        })
                                    
                                    if "Trace Server" in python_modules and python_modules["Trace Server"]:
                                        new_pages.append({
                                            "group": "Trace Server",
                                            "pages": python_modules["Trace Server"]
                                        })
                                    
                                    if "Trace Server Bindings" in python_modules and python_modules["Trace Server Bindings"]:
                                        new_pages.append({
                                            "group": "Trace Server Bindings",
                                            "pages": python_modules["Trace Server Bindings"]
                                        })
                                    
                                    if "Other" in python_modules and python_modules["Other"]:
                                        new_pages.append({
                                            "group": "Other",
                                            "pages": python_modules["Other"]
                                        })
                                    
                                    page["pages"] = new_pages
                                    print(f"✓ Updated Python SDK with {sum(len(m) for m in python_modules.values())} modules")
                                    updated = True
                            
                            # Update TypeScript SDK
                            for page in reference_pages:
                                if isinstance(page, dict) and page.get("group") == "TypeScript SDK":
                                    # Keep the index and README if they exist
                                    new_pages = []
                                    for existing_page in page.get("pages", []):
                                        if isinstance(existing_page, str) and (existing_page.endswith("/index") or existing_page.endswith("/README")):
                                            new_pages.append(existing_page)
                                    
                                    # Add the categorized items with proper casing
                                    if "classes" in typescript_items and typescript_items["classes"]:
                                        new_pages.append({
                                            "group": "Classes",
                                            "pages": typescript_items["classes"]
                                        })
                                    
                                    if "functions" in typescript_items and typescript_items["functions"]:
                                        new_pages.append({
                                            "group": "Functions",
                                            "pages": typescript_items["functions"]
                                        })
                                    
                                    if "interfaces" in typescript_items and typescript_items["interfaces"]:
                                        new_pages.append({
                                            "group": "Interfaces",
                                            "pages": typescript_items["interfaces"]
                                        })
                                    
                                    if "type-aliases" in typescript_items and typescript_items["type-aliases"]:
                                        new_pages.append({
                                            "group": "Type Aliases",
                                            "pages": typescript_items["type-aliases"]
                                        })
                                    
                                    page["pages"] = new_pages
                                    print(f"✓ Updated TypeScript SDK with {sum(len(items) for items in typescript_items.values())} items")
                                    updated = True
                            
                            # Update Service API
                            for page in reference_pages:
                                if isinstance(page, dict) and page.get("group") == "Service API":
                                    # Point to the versioned OpenAPI spec in the Weave repo
                                    # This spec already has the necessary filters applied
                                    import os
                                    version = os.environ.get('WEAVE_VERSION', 'latest')
                                    
                                    # The version should already be resolved by the Python SDK generation
                                    # which converts "latest" to an actual version number
                                    if version and version != "latest":
                                        # Use the specific version tag
                                        openapi_url = f"https://raw.githubusercontent.com/wandb/weave/v{version}/sdks/node/weave.openapi.json"
                                    else:
                                        # Fallback to master if somehow we don't have a version
                                        openapi_url = "https://raw.githubusercontent.com/wandb/weave/master/sdks/node/weave.openapi.json"
                                    
                                    page["openapi"] = openapi_url
                                    print(f"✓ Updated Service API to use versioned OpenAPI spec: {openapi_url}")
                                    updated = True
                            
                            break
                    break
    
    if updated:
        # Write updated docs.json
        with open("docs.json", "w") as f:
            json.dump(docs, f, indent=2)
        print("✓ Updated docs.json with all reference documentation")
    else:
        print("⚠️  Could not find Reference sections to update in docs.json")


def main():
    """Main function."""
    print("Updating Weave reference documentation TOC in docs.json...\n")
    
    # Get Python SDK modules
    python_modules = get_generated_python_modules()
    if python_modules:
        print(f"Found Python SDK modules in {len(python_modules)} groups:")
        for group, pages in python_modules.items():
            print(f"  {group}: {len(pages)} modules")
    else:
        print("No Python SDK modules found")
    
    # Get TypeScript SDK items
    typescript_items = get_typescript_items()
    if typescript_items:
        print(f"\nFound TypeScript SDK items:")
        for category, items in typescript_items.items():
            print(f"  {category}: {len(items)} items")
    else:
        print("\nNo TypeScript SDK items found")
    
    # Get Service API endpoints
    service_endpoints = get_service_api_endpoints()
    if service_endpoints:
        print(f"\nFound {len(service_endpoints)} Service API endpoints")
    else:
        print("\nNo Service API endpoints found")
    
    # Update docs.json
    if python_modules or typescript_items or service_endpoints:
        print("\nUpdating docs.json...")
        update_docs_json(python_modules, typescript_items, service_endpoints)
        print("\n✓ TOC update complete!")
    else:
        print("\n⚠️  No reference documentation found to update")


if __name__ == "__main__":
    main()
