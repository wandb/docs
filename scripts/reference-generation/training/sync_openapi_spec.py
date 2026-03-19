#!/usr/bin/env python3
"""
Sync the Training API OpenAPI spec from the remote service.

This script:
1. Downloads the latest OpenAPI spec from the Training service
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


def fetch_remote_spec(url: str = "https://api.training.wandb.ai/openapi.json") -> dict:
    """Fetch the OpenAPI spec from the remote Training service."""
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


def patch_spec(spec: dict) -> dict:
    """
    Apply patches to the OpenAPI spec to fix known issues.
    
    This is a workaround until the upstream Training API is fixed.
    """
    patched = False
    
    # Fix missing tags on health endpoints
    health_endpoints = ["/v1/health", "/v1/system-check"]
    for path in health_endpoints:
        if path in spec.get("paths", {}):
            for method in ["get", "post", "put", "delete", "patch"]:
                if method in spec["paths"][path]:
                    if "tags" not in spec["paths"][path][method]:
                        spec["paths"][path][method]["tags"] = ["health"]
                        print(f"  ✓ Added missing 'health' tag to {method.upper()} {path}")
                        patched = True
    
    if patched:
        print("  ℹ Applied patches to fix upstream OpenAPI spec issues")
    
    return spec


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
    
    # Find the Training API openapi configuration
    # The structure is complex, so let's search for it
    json_str = json.dumps(docs_config)
    
    # Check if we can find the Training API configuration
    if '"group": "API Reference"' in json_str and 'training/api-reference' in json_str:
        # Read the file again to modify it
        with open(docs_json_path, 'r') as f:
            content = f.read()
        
        if use_local:
            # Replace remote URL with local path
            content = content.replace(
                '"openapi": "https://api.training.wandb.ai/openapi.json"',
                '"openapi": "training/api-reference/openapi.json"'
            )
            print("  ✓ Updated docs.json to use local Training API spec")
        else:
            # Replace local path with remote URL
            content = content.replace(
                '"openapi": "training/api-reference/openapi.json"',
                '"openapi": "https://api.training.wandb.ai/openapi.json"'
            )
            print("  ✓ Updated docs.json to use remote Training API spec")
        
        with open(docs_json_path, 'w') as f:
            f.write(content)
        return True
    
    print("  ✗ Could not find Training API configuration in docs.json")
    return False


def main():
    """Main function."""
    print("Syncing Training API OpenAPI specification...")
    
    local_spec_path = Path("training/api-reference/openapi.json")
    remote_url = "https://api.training.wandb.ai/openapi.json"
    
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
    
    # Apply patches to fix known issues
    remote_spec = patch_spec(remote_spec)
    
    # Load local spec
    local_spec = load_local_spec(local_spec_path)
    
    if local_spec:
        # Compare specs
        is_different, changes = compare_specs(local_spec, remote_spec)
        
        if is_different:
            print("  ⚠ Training API spec has changed:")
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
        
        # Check if currently using local or remote
        json_str = json.dumps(docs_config)
        using_local = '"openapi": "training/api-reference/openapi.json"' in json_str
        
        if using_local:
            print(f"\n  ℹ Currently using local Training API spec ({local_spec_path})")
        else:
            print("\n  ℹ Currently using remote Training API spec (https://api.training.wandb.ai/openapi.json)")
        
        print("\n  Tip: Use --use-local to configure docs.json to use the local spec")
        print("       Use --use-remote to configure docs.json to use the remote spec")
    
    print("✓ Training API spec sync complete!")
    return 0


if __name__ == "__main__":
    exit(main())
