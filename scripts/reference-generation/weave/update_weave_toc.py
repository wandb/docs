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
        
        # Determine the group based on path structure
        # Paths are relative to python-sdk/, so parts[0] is the module directory
        if len(parts) >= 1:
            if parts[0] == "trace":
                group = "Core"
            elif parts[0] == "trace_server":
                group = "Trace Server"
            elif parts[0] == "trace_server_bindings":
                group = "Trace Server Bindings"
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
    ts_dir = Path("weave/reference/typescript-sdk")
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
                toc_path = f"weave/reference/typescript-sdk/{subdir}/{name}"
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


def _iter_nav_dicts(node):
    """Yield every dict found while walking a navigation subtree.

    Walks into both 'pages' and 'tabs' arrays, which Mintlify nests
    at different depths depending on the navigation shape.
    """
    if isinstance(node, dict):
        yield node
        for child in node.get("pages", []):
            yield from _iter_nav_dicts(child)
        for child in node.get("tabs", []):
            yield from _iter_nav_dicts(child)
    elif isinstance(node, list):
        for child in node:
            yield from _iter_nav_dicts(child)


def _find_english_group(docs, group_name):
    """Return the first dict with group == group_name inside the English nav.

    Resilient to changes in how `tab` and `group` levels are nested under
    `navigation.languages[lang=en]`, so the regen workflow keeps working
    even if the Reference / W&B Weave / SDK hierarchy is restructured.
    """
    languages = docs.get("navigation", {}).get("languages", [])
    en_lang = next(
        (lang for lang in languages if lang.get("language") == "en"),
        None,
    )
    if en_lang is None:
        return None
    for node in _iter_nav_dicts(en_lang):
        if isinstance(node, dict) and node.get("group") == group_name:
            return node
    return None


def update_docs_json(python_modules, typescript_items, service_endpoints):
    """Update the docs.json file with all reference documentation."""

    # Read current docs.json
    with open("docs.json", "r") as f:
        docs = json.load(f)

    if not docs.get("navigation", {}).get("languages"):
        print("⚠️  No languages found in navigation")
        return

    updated = False

    # Update Python SDK
    python_group = _find_english_group(docs, "Python SDK")
    if python_group is not None:
        new_pages = ["weave/reference/python-sdk"]

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

        python_group["pages"] = new_pages
        print(f"✓ Updated Python SDK with {sum(len(m) for m in python_modules.values())} modules")
        updated = True
    else:
        print("⚠️  Could not find 'Python SDK' group in English navigation")

    # Update TypeScript SDK
    typescript_group = _find_english_group(docs, "TypeScript SDK")
    if typescript_group is not None:
        new_pages = ["weave/reference/typescript-sdk"]

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

        typescript_group["pages"] = new_pages
        print(f"✓ Updated TypeScript SDK with {sum(len(items) for items in typescript_items.values())} items")
        updated = True
    else:
        print("⚠️  Could not find 'TypeScript SDK' group in English navigation")

    # Note: Service API OpenAPI configuration is managed by sync_openapi_spec.py
    # We don't modify it here to preserve the local vs remote spec choice

    if updated:
        # Write updated docs.json. Use ensure_ascii=False so we don't
        # mangle non-ASCII characters in localized navigation entries
        # (French, Japanese, Korean) into \uXXXX escape sequences.
        with open("docs.json", "w") as f:
            json.dump(docs, f, indent=2, ensure_ascii=False)
        print("✓ Updated docs.json with all reference documentation")
    else:
        print("⚠️  No SDK groups were updated in docs.json. The navigation "
              "structure may have changed - inspect docs.json and update "
              "this script's group lookup if needed.")


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
