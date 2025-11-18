#!/usr/bin/env python3
"""
Sync the OpenAPI spec from the remote service and optionally use local copy.

This script:
1. Downloads the latest OpenAPI spec from the service
2. Compares it with the local copy (if exists)
3. Updates the local copy if changed
4. Can optionally update docs.json to use local spec for builds
"""

import json
import hashlib
from pathlib import Path
import requests
import sys
from typing import Optional, Tuple


def fetch_remote_spec(url: str = "https://trace.wandb.ai/openapi.json") -> dict:
    """Fetch the OpenAPI spec from the remote service."""
    print(f"  Fetching remote spec from {url}...")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"  ✗ Failed to fetch remote spec: {e}")
        return None


def load_local_spec(path: Path) -> Optional[dict]:
    """Load the local OpenAPI spec if it exists."""
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return None


def spec_hash(spec: dict) -> str:
    """Generate a hash of the spec for comparison."""
    # Sort keys for consistent hashing
    spec_str = json.dumps(spec, sort_keys=True)
    return hashlib.sha256(spec_str.encode()).hexdigest()


def validate_spec(spec: dict) -> list:
    """
    Validate the OpenAPI spec for potential issues.
    Returns list of warning messages about spec issues.
    """
    warnings = []
    
    paths = spec.get("paths", {})
    
    # Track endpoint definitions to detect duplicates
    endpoint_map = {}  # (method, path) -> [operation_ids]
    tag_endpoint_map = {}  # tag -> [(method, path)]
    
    for path, path_item in paths.items():
        for method in ["get", "post", "put", "delete", "patch"]:
            if method not in path_item:
                continue
                
            operation = path_item[method]
            operation_id = operation.get("operationId", "")
            tags = operation.get("tags", [])
            
            # Check for duplicate endpoint definitions
            endpoint_key = (method.upper(), path)
            if endpoint_key not in endpoint_map:
                endpoint_map[endpoint_key] = []
            endpoint_map[endpoint_key].append(operation_id)
            
            # Track endpoints by tag to detect if endpoints appear in multiple tags
            for tag in tags:
                if tag not in tag_endpoint_map:
                    tag_endpoint_map[tag] = []
                tag_endpoint_map[tag].append(endpoint_key)
    
    # Check for actual duplicates (same endpoint with different operation IDs)
    for endpoint_key, operation_ids in endpoint_map.items():
        if len(operation_ids) > 1:
            method, path = endpoint_key
            warnings.append(f"  ⚠ Duplicate endpoint: {method} {path} defined {len(operation_ids)} times with operation IDs: {operation_ids}")
    
    # Check for endpoints appearing in multiple categories (tags)
    endpoint_tag_count = {}
    for tag, endpoints in tag_endpoint_map.items():
        for endpoint in endpoints:
            if endpoint not in endpoint_tag_count:
                endpoint_tag_count[endpoint] = []
            endpoint_tag_count[endpoint].append(tag)
    
    for endpoint, tags in endpoint_tag_count.items():
        if len(tags) > 1:
            method, path = endpoint
            warnings.append(f"  ℹ Endpoint {method} {path} appears in multiple categories: {tags}")
    
    return warnings


def compare_specs(local_spec: dict, remote_spec: dict) -> Tuple[bool, list]:
    """
    Compare local and remote specs.
    Returns (is_different, changes_summary)
    """
    if spec_hash(local_spec) == spec_hash(remote_spec):
        return False, []
    
    changes = []
    
    # Compare paths
    local_paths = set(local_spec.get("paths", {}).keys())
    remote_paths = set(remote_spec.get("paths", {}).keys())
    
    added = remote_paths - local_paths
    removed = local_paths - remote_paths
    
    if added:
        changes.append(f"  + Added {len(added)} endpoints: {', '.join(sorted(added)[:3])}{'...' if len(added) > 3 else ''}")
    if removed:
        changes.append(f"  - Removed {len(removed)} endpoints: {', '.join(sorted(removed)[:3])}{'...' if len(removed) > 3 else ''}")
    
    # Check for modified endpoints
    common_paths = local_paths & remote_paths
    modified = 0
    for path in common_paths:
        if local_spec["paths"][path] != remote_spec["paths"][path]:
            modified += 1
    
    if modified:
        changes.append(f"  ~ Modified {modified} endpoints")
    
    return True, changes


def update_docs_json(use_local: bool = False):
    """Update docs.json to use local or remote OpenAPI spec."""
    docs_json_path = Path("docs.json")
    
    with open(docs_json_path, 'r') as f:
        docs_config = json.load(f)
    
    # Find the Service API openapi configuration
    for nav_item in docs_config.get("navigation", []):
        if nav_item.get("group") == "Weave":
            for page in nav_item.get("pages", []):
                if isinstance(page, dict) and "Weave Reference" in page.get("group", ""):
                    for ref_page in page.get("pages", []):
                        if isinstance(ref_page, dict) and ref_page.get("group") == "Service API":
                            if "openapi" in ref_page:
                                if use_local:
                                    # Use local spec
                                    ref_page["openapi"] = "weave/reference/service-api/openapi.json"
                                    print("  ✓ Updated docs.json to use local OpenAPI spec")
                                else:
                                    # Use remote spec
                                    ref_page["openapi"] = {"source": "https://trace.wandb.ai/openapi.json"}
                                    print("  ✓ Updated docs.json to use remote OpenAPI spec")
                                
                                with open(docs_json_path, 'w') as f:
                                    json.dump(docs_config, f, indent=2)
                                    f.write('\n')
                                return True
    
    print("  ✗ Could not find Service API configuration in docs.json")
    return False


def main():
    """Main function."""
    print("Syncing OpenAPI specification...")
    
    local_spec_path = Path("weave/reference/service-api/openapi.json")
    remote_url = "https://trace.wandb.ai/openapi.json"
    
    # Fetch remote spec
    remote_spec = fetch_remote_spec(remote_url)
    if not remote_spec:
        # If can't fetch remote, ensure we're using local
        if local_spec_path.exists():
            print("  ⚠ Using existing local spec due to remote fetch failure")
            update_docs_json(use_local=True)
            return 0
        else:
            print("  ✗ No local spec and couldn't fetch remote spec")
            return 1
    
    # Validate the remote spec for issues
    print("\n  Validating OpenAPI spec...")
    spec_warnings = validate_spec(remote_spec)
    if spec_warnings:
        print("  ⚠ OpenAPI spec validation warnings:")
        for warning in spec_warnings:
            print(warning)
        if any("Duplicate endpoint" in w for w in spec_warnings):
            print("\n  ⚠ CRITICAL: Duplicate endpoint definitions found!")
            print("     This may indicate an issue in the upstream OpenAPI spec.")
            print("     Consider reporting this to the Weave team: https://github.com/wandb/weave/issues")
    else:
        print("  ✓ OpenAPI spec validation passed")
    
    # Load local spec
    local_spec = load_local_spec(local_spec_path)
    
    if local_spec:
        # Compare specs
        is_different, changes = compare_specs(local_spec, remote_spec)
        
        if is_different:
            print("  ⚠ OpenAPI spec has changed:")
            for change in changes:
                print(change)
            
            # Save the new spec
            local_spec_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_spec_path, 'w') as f:
                json.dump(remote_spec, f, indent=2)
                f.write('\n')
            print(f"  ✓ Updated local spec at {local_spec_path}")
        else:
            print("  ✓ Local spec is up to date")
    else:
        # No local spec, save the remote one
        local_spec_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_spec_path, 'w') as f:
            json.dump(remote_spec, f, indent=2)
            f.write('\n')
        print(f"  ✓ Created local spec at {local_spec_path}")
    
    # Parse command line arguments
    use_local = "--use-local" in sys.argv
    use_remote = "--use-remote" in sys.argv
    
    if use_local:
        update_docs_json(use_local=True)
    elif use_remote:
        update_docs_json(use_local=False)
    else:
        # Default: Check current configuration
        docs_json_path = Path("docs.json")
        with open(docs_json_path, 'r') as f:
            docs_config = json.load(f)
        
        # Check if currently using local or remote by searching for the openapi config
        # The structure is complex, so let's just search for the pattern in the JSON string
        json_str = json.dumps(docs_config)
        
        # Check if we have the local path or remote config
        using_local = '"openapi": "weave/reference/service-api/openapi.json"' in json_str or '"openapi": "openapi.json"' in json_str
        
        if using_local:
            print(f"\n  ℹ Currently using local OpenAPI spec ({local_spec_path})")
        else:
            print("\n  ℹ Currently using remote OpenAPI spec (https://trace.wandb.ai/openapi.json)")
        
        print("\n  Tip: Use --use-local to configure docs.json to use the local spec")
        print("       Use --use-remote to configure docs.json to use the remote spec")
    
    print("✓ OpenAPI spec sync complete!")
    return 0


if __name__ == "__main__":
    exit(main())
