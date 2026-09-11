#!/usr/bin/env python3
"""
Sync the OpenAPI spec for the Weave Service API from the remote service.

This script:
1. Downloads the latest OpenAPI spec from the service
2. Validates it and compares it with the local copy (if exists)
3. Updates the local copy if changed
4. Maintains English-language entries in docs.json. 

Reference pages are generated from the committed spec by
scripts/reference-generation/common/generate_openapi_stubs.py, which owns both the stub
pages and the navigation entries that list them.
"""

import json
import hashlib
import os
from pathlib import Path
import requests
from typing import Optional, Tuple

# Remote OpenAPI spec URL
# Primary: GitHub raw URL (if available in wandb/core repo - more stable, version-controlled)
# Fallback: Live service URL (may change frequently)
GITHUB_SPEC_URL = "https://raw.githubusercontent.com/wandb/core/master/services/weave-trace/openapi.json"
LIVE_SPEC_URL = "https://trace.wandb.ai/openapi.json"
REMOTE_SPEC_URL = GITHUB_SPEC_URL  # Try GitHub first, fallback to live service in fetch_remote_spec


def fetch_remote_spec(url: str = None) -> dict:
    """Fetch the OpenAPI spec from the remote service.
    
    Tries GitHub first (more stable), falls back to live service if GitHub fails.
    For private repos, uses GITHUB_TOKEN or GITHUB_PAT from environment if available.
    """
    if url is None:
        # Try GitHub first (preferred - version controlled)
        print(f"  Fetching remote spec from {GITHUB_SPEC_URL}...")
        
        # Check for GitHub authentication token (for private repos)
        github_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_PAT")
        headers = {}
        if github_token:
            headers["Authorization"] = f"token {github_token}"
            print("  Using GitHub authentication token")
        
        try:
            response = requests.get(GITHUB_SPEC_URL, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            if response.status_code == 404 and not github_token:
                print(f"  ⚠ GitHub spec not available (404 - may require authentication for private repo): {e}")
            else:
                print(f"  ⚠ GitHub spec not available: {e}")
            print(f"  Falling back to live service: {LIVE_SPEC_URL}...")
            url = LIVE_SPEC_URL
    
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


def main():
    """Main function."""
    print("Syncing OpenAPI specification...")
    
    local_spec_path = Path("weave/reference/service-api/openapi.json")
    
    # Fetch remote spec (tries GitHub first, falls back to live service)
    remote_spec = fetch_remote_spec()
    if not remote_spec:
        # If can't fetch remote, ensure we're using local
        if local_spec_path.exists():
            print("  ⚠ Using existing local spec due to remote fetch failure")
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
    
    # Pages are generated from this spec by
    # scripts/reference-generation/common/generate_openapi_stubs.py, which the workflow
    # runs next. docs.json no longer carries an `openapi` source to point at, so there is
    # nothing here to reconfigure.
    print(f"\n  ℹ Reference pages are generated from {local_spec_path}")
    print("✓ OpenAPI spec sync complete!")
    return 0


if __name__ == "__main__":
    exit(main())
