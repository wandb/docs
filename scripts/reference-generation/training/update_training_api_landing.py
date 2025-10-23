#!/usr/bin/env python3
"""
Update the Training API landing page with the current endpoints from the OpenAPI spec.

This script fetches the OpenAPI spec (either from local file or remote URL) and
updates the Available Endpoints section in the training/api-reference.mdx landing page.
"""

import json
import re
from pathlib import Path
import requests
from typing import Dict, List, Tuple


def fetch_openapi_spec() -> dict:
    """Fetch OpenAPI spec from local file or remote URL."""
    # First try local file
    local_spec = Path("training/api-reference/openapi.json")
    if local_spec.exists():
        print(f"  Using local OpenAPI spec: {local_spec}")
        with open(local_spec, 'r') as f:
            return json.load(f)
    
    # Fallback to remote
    print("  Fetching remote OpenAPI spec from https://api.training.wandb.ai/openapi.json")
    response = requests.get("https://api.training.wandb.ai/openapi.json")
    response.raise_for_status()
    return response.json()


def parse_endpoints(spec: dict) -> Dict[str, List[Tuple[str, str, str, str]]]:
    """
    Parse endpoints from OpenAPI spec and group by category.
    Returns dict of category -> list of (method, path, operation_id, summary)
    """
    endpoints_by_tag = {}
    
    for path, path_item in spec.get("paths", {}).items():
        for method, operation in path_item.items():
            if method in ["get", "post", "put", "delete", "patch"]:
                tags = operation.get("tags", ["Uncategorized"])
                summary = operation.get("summary", "")
                operation_id = operation.get("operationId", "")
                
                for tag in tags:
                    if tag not in endpoints_by_tag:
                        endpoints_by_tag[tag] = []
                    endpoints_by_tag[tag].append((method.upper(), path, operation_id, summary))
    
    # Sort endpoints within each tag
    for tag in endpoints_by_tag:
        endpoints_by_tag[tag].sort(key=lambda x: (x[1], x[0]))  # Sort by path, then method
    
    return endpoints_by_tag


def generate_endpoint_url(operation_id: str, tag: str, path: str) -> str:
    """Generate the Mintlify URL for an endpoint based on its operation ID and tag."""
    # Map tags to URL segments
    tag_mapping = {
        "Chat": "chat",
        "Completions": "completions",
        "Models": "models",
        "Jobs": "jobs",
        "Training": "training",
        "Inference": "inference",
        "Health": "health",
        "Service": "service",
    }
    
    tag_segment = tag_mapping.get(tag, tag.lower().replace(" ", "-"))
    
    # Convert operation_id to URL slug
    # Remove the suffix like _get, _post, etc.
    url_slug = re.sub(r'_(get|post|put|delete|patch)$', '', operation_id)
    # Replace underscores with hyphens
    url_slug = url_slug.replace('_', '-')
    
    # If no operation_id, use the path to generate a slug
    if not url_slug:
        # Convert path like /v1/chat/completions to chat-completions
        url_slug = path.strip('/').replace('/v1/', '').replace('/', '-')
    
    return f"https://docs.wandb.ai/training/api-reference/{tag_segment}/{url_slug}"


def generate_endpoints_section(endpoints: Dict[str, List[Tuple[str, str, str, str]]]) -> str:
    """Generate the markdown for the Available Endpoints section."""
    lines = ["## Available Endpoints\n\n"]
    
    # Define the order of categories
    category_order = ["Chat", "Completions", "Models", "Jobs", "Training", "Health", "Service"]
    
    # Add any categories not in the predefined order
    for tag in endpoints:
        if tag not in category_order:
            category_order.append(tag)
    
    for category in category_order:
        if category not in endpoints:
            continue
            
        lines.append(f"\n### {category}\n\n")
        
        for method, path, operation_id, summary in endpoints[category]:
            url = generate_endpoint_url(operation_id, category, path)
            lines.append(f"- **[{method} {path}]({url})** - {summary}\n")
    
    return "".join(lines)


def update_landing_page(endpoints_section: str):
    """Update the training/api-reference.mdx landing page with new endpoints section."""
    landing_page = Path("training/api-reference.mdx")
    
    if not landing_page.exists():
        print(f"  ✗ Landing page not found at {landing_page}")
        return False
    
    with open(landing_page, 'r') as f:
        content = f.read()
    
    # Check if there's already an Available Endpoints section
    if "## Available Endpoints" in content:
        # Replace existing section
        pattern = r'## Available Endpoints\n.*?(?=\n##|\Z)'
        new_content = re.sub(pattern, endpoints_section.rstrip(), content, flags=re.DOTALL)
    else:
        # Add the section before "## Related Resources" if it exists
        if "## Related Resources" in content:
            new_content = content.replace("## Related Resources", f"{endpoints_section}\n## Related Resources")
        else:
            # Otherwise add it at the end
            new_content = content + "\n\n" + endpoints_section
    
    if new_content != content:
        with open(landing_page, 'w') as f:
            f.write(new_content)
        print(f"  ✓ Updated {landing_page}")
        return True
    else:
        print(f"  ✓ No changes needed for {landing_page}")
        return False


def main():
    """Main function."""
    print("Updating Training API landing page...")
    
    try:
        # Fetch OpenAPI spec
        spec = fetch_openapi_spec()
        
        # Parse endpoints
        endpoints = parse_endpoints(spec)
        
        # Count total endpoints
        total = sum(len(eps) for eps in endpoints.values())
        print(f"  Found {total} endpoints in {len(endpoints)} categories")
        
        # Generate endpoints section
        endpoints_section = generate_endpoints_section(endpoints)
        
        # Update landing page
        if update_landing_page(endpoints_section):
            print("✓ Training API landing page updated successfully!")
        else:
            print("✓ Training API landing page is already up to date")
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
